"""Isolated CPU MNN smoke run; upstream quantized weights, no LoRA or speedup claim."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def worker(output):
    sys.path.insert(0, str(ROOT / 'outputs/mnn-runtime'))
    import MNN.llm as mnnllm
    from transformers import AutoTokenizer

    directory = ROOT / 'outputs/models/qwen3-1.7b-mnn'
    source = json.loads((directory / 'source.json').read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'outputs/models/qwen3-1.7b', local_files_only=True)
    engine = mnnllm.create(str(directory / 'config.json'))
    config = dict(backend_type='cpu', thread_num=4, max_new_tokens=64,
                  sampler_type='greedy', reuse_kv=False, use_template=False)
    if not engine.set_config(config):
        raise RuntimeError('MNN rejected the requested CPU configuration')
    start = time.perf_counter()
    engine.load()
    load_seconds = time.perf_counter() - start
    records = []
    cases = ['请只回复：你好。', '今天有点累，不想听建议，陪我随便聊两句。',
             '用一个简单例子说明Python列表和元组的区别，两句话以内。']
    for index, question in enumerate([cases[0]] + cases):
        engine.reset()
        engine.generate_init()
        messages = [dict(role='system', content='你是中文AI聊天助手。简短、自然、准确地回应，不虚构经历。'),
                    dict(role='user', content=question)]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_ids = engine.tokenizer_encode(prompt)
        if input_ids != tokenizer.encode(prompt, add_special_tokens=False):
            raise ValueError('MNN tokenizer differs from the fixed local Qwen tokenizer')
        start = time.perf_counter()
        # The installed Windows wheel's generate(ids) returned an empty list.
        # response(raw_prompt) works; disable its extra template and verify token counts.
        raw_response = engine.response(prompt, False)
        seconds = time.perf_counter() - start
        context = engine.context
        tokens = list(context.output_tokens)
        if context.status not in (mnnllm.LlmStatus.NORMAL_FINISHED, mnnllm.LlmStatus.MAX_TOKENS_FINISHED):
            print(json.dumps(dict(status=str(context.status),returned_tokens=tokens,
                                  context_tokens=context.output_tokens,log=engine.get_log()),ensure_ascii=False),flush=True)
            raise RuntimeError('MNN generation status: ' + str(context.status))
        if context.prompt_len != len(input_ids) or context.gen_seq_len != len(tokens):
            raise ValueError('Prompt/output token counts cannot be verified')
        answer = tokenizer.decode(tokens, skip_special_tokens=True)
        record = dict(warmup=index == 0, question=question, answer=answer, raw_response=raw_response,
                      input_tokens=len(input_ids), output_tokens=len(tokens), generation_seconds=seconds,
                      output_tokens_per_second=len(tokens) / seconds, status=str(context.status),
                      prefill_us=context.prefill_us, decode_us=context.decode_us,
                      context_output_tokens=list(context.output_tokens))
        if not answer.strip():
            raise ValueError('Empty MNN answer')
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    report = dict(runtime='MNN 3.6.1 Windows CPU', platform=platform.platform(), config=config,
                  source_model=source['model_id'], source_revision=source['revision'],
                  tokenizer='local fixed Qwen3-1.7B; MNN input IDs verified equal',
                  load_seconds=load_seconds, cases=records,
                  limitations='Upstream export, not our conversion or training. CPU smoke only; not a fair GPU speed comparison, concurrency test or MNN Chat mobile validation.')
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--output', default='docs/mnn-cpu-smoke.json')
    args = parser.parse_args()
    if args.worker:
        worker(args.output)
        return
    # Native runtime failures or hangs stay outside the active chat service.
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker', '--output', args.output]
    env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}
    result = subprocess.run(command, cwd=ROOT, env=env, timeout=600)
    if result.returncode:
        raise SystemExit(f'MNN worker failed (exit {result.returncode}); active chat backend was not changed.')


if __name__ == '__main__':
    main()
