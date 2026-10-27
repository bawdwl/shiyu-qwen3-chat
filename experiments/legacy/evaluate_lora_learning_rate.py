"""Compare two LCCC LoRA learning-rate runs on fixed development prompts.

This is an exposed diagnostic run, not a blind or independent quality score.
"""
import argparse
import gc
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch
from transformers import set_seed

from characters import conversation_messages
from inference import GENERATION_SETTINGS, QwenChat


CONTROLLED_KEYS = (
    'base_model', 'revision', 'seed', 'epochs', 'micro_batch', 'gradient_accumulation',
    'max_length', 'r', 'lora_alpha', 'lora_dropout', 'target_modules', 'dtype',
    'gradient_checkpointing', 'optimizer', 'weight_decay', 'max_grad_norm', 'schedule',
)


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def load_and_validate_runs(old_root, candidate_root):
    old = json.loads((old_root / 'run-config.json').read_text(encoding='utf-8'))
    candidate = json.loads((candidate_root / 'run-config.json').read_text(encoding='utf-8'))
    differences = [key for key in CONTROLLED_KEYS if old.get(key) != candidate.get(key)]
    if old.get('data', {}).get('sha256') != candidate.get('data', {}).get('sha256'):
        differences.append('data.sha256')
    if differences:
        raise ValueError('除学习率外训练条件也有差异：' + ', '.join(differences))
    if old.get('learning_rate') == candidate.get('learning_rate'):
        raise ValueError('两个 run 的学习率相同')
    for name, run in [('old', old), ('candidate', candidate)]:
        conditioning = run.get('conditioning') or {}
        if conditioning.get('system_prompt'):
            raise ValueError(f'{name} run 含训练 system prompt，无法与本次单因素对照匹配')
        if run.get('model_profile') != 'qwen3-1.7b':
            raise ValueError(f'{name} run 不是 Qwen3-1.7B')
    return old, candidate


def summarize(records):
    result = {}
    for variant in ('base', 'old_lora', 'candidate_lora'):
        rows = [row for row in records if row['variant'] == variant]
        if rows:
            result[variant] = dict(
                replies=len(rows),
                median_output_tokens=statistics.median(row['metrics']['output_tokens'] for row in rows),
                replies_le5_tokens=sum(row['metrics']['output_tokens'] <= 5 for row in rows),
                truncated=sum(not row['metrics']['ended_with_eos'] for row in rows),
            )
    return result


def render(records, cases):
    variants = ('base', 'old_lora', 'candidate_lora')
    names = {'base': 'Base', 'old_lora': '旧 LoRA（2e-4）', 'candidate_lora': '候选 LoRA（2e-5）'}
    lines = [
        '# 学习率单因素对照：开发回答', '',
        '这些回答来自已公开给开发者的诊断场景，不是盲评或独立质量测试。',
        '每种模型收到相同上下文和当前 App 角色提示；输出长度不是质量分。', '',
    ]
    for case in cases:
        lines.extend(['## ' + case['id'], ''])
        for message in case['messages']:
            lines.append(f"- {message['role']}：{message['content']}")
        lines.append('')
        if case.get('rubric'):
            lines.extend(['**检查点：** ' + case['rubric'], ''])
        for variant in variants:
            rows = [row for row in records if row['case_id'] == case['id'] and row['variant'] == variant]
            for row in rows:
                lines.append(f"**{names[variant]} / seed={row['seed']}：** {row['answer']}")
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-run', default='outputs/lora-lccc-base-v1-pilot')
    parser.add_argument('--candidate-run', default='outputs/lora-lccc-base-v3-lr2e-5')
    parser.add_argument('--cases', default='data/im_chat_development_v2.json')
    parser.add_argument('--case-kind', choices=['chat', 'capability_regression'])
    parser.add_argument('--output-dir', default='outputs/lccc-learning-rate-v3-development')
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261026, 20261027])
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be nonempty and unique')
    old_root, candidate_root = Path(args.old_run), Path(args.candidate_run)
    old, candidate = load_and_validate_runs(old_root, candidate_root)
    cases_bytes = Path(args.cases).read_bytes()
    source = json.loads(cases_bytes)
    cases = source['cases']
    if args.case_kind:
        cases = [case for case in cases if case.get('kind', 'chat') == args.case_kind]
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('开发场景为空或 ID 重复')

    prompts = {}
    messages_by_case = {}
    for case in cases:
        messages = case['messages']
        if not messages or messages[-1]['role'] != 'user' or any(
            message['role'] != ('user' if i % 2 == 0 else 'assistant') or not message['content'].strip()
            for i, message in enumerate(messages)
        ):
            raise ValueError('固定上下文格式错误：' + case['id'])
        full = conversation_messages(dict(character_id='companion', messages=messages))
        system = next((message['content'] for message in full if message['role'] == 'system'), None)
        prompts[case['id']] = dict(system_prompt=system,
                                   sha256=sha256_bytes(system.encode('utf-8')) if system else None)
        messages_by_case[case['id']] = full

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'cases.snapshot.json').write_bytes(cases_bytes)
    config = dict(
        timestamp=datetime.now().astimezone().isoformat(),
        old_learning_rate=old['learning_rate'], candidate_learning_rate=candidate['learning_rate'],
        changed_factor='AdamW learning rate only; 10x lower in candidate',
        model_id=candidate['base_model'], revision=candidate['revision'], seeds=args.seeds,
        data_sha256=candidate['data']['sha256'],
        development_cases_sha256=sha256_bytes(cases_bytes),
        selected_case_ids=[case['id'] for case in cases], case_kind=args.case_kind,
        prompt_by_case=prompts, generation_settings=GENERATION_SETTINGS,
        adapters={
            'old_lora': sha256_bytes((old_root / 'adapter' / 'adapter_model.safetensors').read_bytes()),
            'candidate_lora': sha256_bytes((candidate_root / 'adapter' / 'adapter_model.safetensors').read_bytes()),
        },
        note='开发诊断；记录原始回答及长度/截断，不自动评分自然度、事实正确率或用户偏好。',
    )
    (root / 'run-config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    records = []
    base_answers = {}
    try:
        for run_name, run_root, lora_name in (
            ('old', old_root, 'old_lora'), ('candidate', candidate_root, 'candidate_lora')
        ):
            engine = QwenChat(adapter_run=run_root, model_profile='qwen3-1.7b', trim_history=True)
            for variant, mode in (('base', 'base'), (lora_name, 'lora')):
                for case in cases:
                    for seed in args.seeds:
                        set_seed(seed)
                        answer, metrics = engine.reply(messages_by_case[case['id']], model_mode=mode)
                        record = dict(case_id=case['id'], kind=case.get('kind', 'chat'),
                                      variant=variant, seed=seed, answer=answer, metrics=metrics)
                        records.append(record)
                        if variant == 'base':
                            base_answers[(case['id'], seed, run_name)] = answer
                        (root / 'results.json').write_text(
                            json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
                        print(f"Recorded {variant} {case['id']} seed={seed}", flush=True)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        base_match = sum(
            base_answers[(case['id'], seed, 'old')] == base_answers[(case['id'], seed, 'candidate')]
            for case in cases for seed in args.seeds
        )
        summary = dict(
            outputs=summarize(records),
            duplicated_base_answers_matched=base_match,
            duplicated_base_answers_total=len(cases) * len(args.seeds),
            note='长度与截断统计仅描述生成，不代表回答质量。',
        )
        (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        (root / 'answers.md').write_text(render(records, cases), encoding='utf-8')
        (root / 'status.json').write_text(json.dumps(dict(status='completed', replies=len(records)), ensure_ascii=False, indent=2), encoding='utf-8')
    except Exception as exc:
        (root / 'status.json').write_text(json.dumps(dict(status='failed', error=f'{type(exc).__name__}: {exc}'), ensure_ascii=False, indent=2), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
