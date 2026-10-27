"""固定题集的基础模型评估；规则分数不等同于综合回答质量。"""
import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import set_seed

from inference import GENERATION_SETTINGS, MODEL_ID, MODEL_REVISION, QwenChat


def grade(answer, check):
    """整段解析，不移除代码块；格式要求是题目的一部分。"""
    answer = answer.strip().replace('\r\n', '\n')
    if check['type'] == 'exact':
        return answer == check['expected']
    if check['type'] == 'json_equal':
        try:
            actual = json.loads(answer)
        except (ValueError, TypeError):
            return False
        # 同时检查类型，避免把 true 或 3.0 当成整数 3。
        def same(a, b):
            if type(a) is not type(b):
                return False
            if isinstance(b, dict):
                return a.keys() == b.keys() and all(same(a[k], b[k]) for k in b)
            if isinstance(b, list):
                return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
            return a == b
        return same(actual, check['expected'])
    raise ValueError(f"不支持的规则：{check['type']}")


def load_cases(path):
    raw = Path(path).read_bytes()
    cases = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if not cases or len({c['id'] for c in cases}) != len(cases):
        raise ValueError('测试题为空或 ID 重复')
    for case in cases:
        assert case['prompts'] and all(isinstance(p, str) and p.strip() for p in case['prompts'])
        if case['check']['type'] not in ('exact', 'json_equal'):
            raise ValueError('未知评分规则')
    return cases, hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/eval.jsonl')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--adapter-run', default=None, help='包含 run-config.json 和 adapter/ 的训练目录')
    args = parser.parse_args()
    cases, digest = load_cases(args.dataset)
    target = Path(args.output_dir or ('outputs/baseline-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    # 不覆盖既有评估，保留失败运行和负面结果。
    target.mkdir(parents=True, exist_ok=False)
    config = dict(timestamp=datetime.now().astimezone().isoformat(), model=MODEL_ID,
                  revision=MODEL_REVISION, dataset_sha256=digest, dataset=args.dataset,
                  case_count=len(cases), seed=42, generation_settings=GENERATION_SETTINGS,
                  enable_thinking=False, dtype='bfloat16', torch=torch.__version__,
                  transformers=transformers.__version__, python=platform.python_version(),
                  gpu=torch.cuda.get_device_name(0), scope='每题重置历史；多轮题使用本模型实际前轮输出；只评分最后一轮。')
    config['evaluation_script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config['inference_script_sha256'] = hashlib.sha256(Path('inference.py').read_bytes()).hexdigest()
    if args.adapter_run:
        from peft import PeftConfig
        root = Path(args.adapter_run)
        train_config = json.loads((root/'run-config.json').read_text(encoding='utf-8'))
        if train_config['base_model'] != MODEL_ID or train_config['revision'] != MODEL_REVISION:
            raise ValueError('Adapter 基础模型版本不匹配')
        if train_config['data']['sha256']['eval'] != digest:
            raise ValueError('测试题指纹与训练记录不一致')
        adapter_config = PeftConfig.from_pretrained(root/'adapter')
        if adapter_config.base_model_name_or_path != MODEL_ID:
            raise ValueError('Adapter 元数据中的基础模型不匹配')
        config.update(adapter_run=str(root), adapter_sha256=hashlib.sha256(
            (root/'adapter'/'adapter_model.safetensors').read_bytes()).hexdigest())
    (target / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    engine = QwenChat()
    if args.adapter_run:
        from peft import PeftModel
        engine.model = PeftModel.from_pretrained(engine.model, Path(args.adapter_run)/'adapter',
                                                 is_trainable=False).eval()
    set_seed(42)
    engine.reply([{'role': 'user', 'content': '只回复：你好'}])
    print('Warmup completed; starting fixed cases.', flush=True)
    records = []
    started = time.perf_counter()
    with (target / 'results.jsonl').open('w', encoding='utf-8') as log:
        for case in cases:
            set_seed(42)
            messages, turns = [], []
            error = None
            try:
                for prompt in case['prompts']:
                    messages.append({'role': 'user', 'content': prompt})
                    answer, metrics = engine.reply(messages)
                    messages.append({'role': 'assistant', 'content': answer})
                    turns.append(dict(prompt=prompt, answer=answer, **metrics))
                passed = grade(turns[-1]['answer'], case['check'])
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                passed = False
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
            result = dict(id=case['id'], category=case['category'], check=case['check'],
                          turns=turns, passed=passed, error=error)
            log.write(json.dumps(result, ensure_ascii=False) + '\n')
            log.flush()
            records.append(result)
            print(f"{case['id']}: {'PASS' if passed else 'FAIL'}" + (f' ({error})' if error else ''), flush=True)
    groups = defaultdict(list)
    for result in records:
        groups[result['category']].append(result)
    turns = [turn for r in records for turn in r['turns']]
    summary = dict(total=len(records), passed=sum(r['passed'] for r in records),
                   errors=sum(r['error'] is not None for r in records),
                   categories={k: dict(total=len(v), passed=sum(r['passed'] for r in v)) for k, v in groups.items()},
                   measured_turns=len(turns), elapsed_seconds=time.perf_counter() - started,
                   median_generation_seconds=statistics.median(t['seconds'] for t in turns) if turns else None,
                   aggregate_output_tokens_per_second=sum(t['output_tokens'] for t in turns) / sum(t['seconds'] for t in turns) if turns else None,
                   peak_allocated_gib=max((t['peak_allocated_gib'] for t in turns), default=None),
                   peak_reserved_gib=max((t['peak_reserved_gib'] for t in turns), default=None),
                   truncated_turns=sum(not t['ended_with_eos'] for t in turns),
                   caveat='小规模自编题；严格格式规则通过率，不是语义质量/通用能力。预热后单次运行，速度包含提示词处理，显存仅本进程 PyTorch。')
    (target / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f'Results: {target}', flush=True)


if __name__ == '__main__':
    main()
