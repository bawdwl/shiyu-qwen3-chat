"""独立进程、预热、重复测量；不修改线上推理默认值。"""
import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROMPTS = [
    {'id': 'short', 'content': '请解释数据库事务为什么需要提交和回滚。'},
    {'id': 'long', 'content': '以下是聊天系统的设计背景：' +
     '网页通过HTTP调用本地模型服务，完整问答保存到SQLite，停止时不应保存半轮消息。' * 12 +
     '请说明取消生成与数据库提交之间如何保持状态一致。'},
]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def summarize(rows):
    groups = []
    for workload in ('natural', 'fixed_tokens'):
        for prompt in PROMPTS:
            selected = [r for r in rows if r['workload'] == workload and r['prompt_id'] == prompt['id']]
            if not selected:
                continue
            result = dict(workload=workload, prompt_id=prompt['id'], count=len(selected))
            for key in ('seconds', 'first_visible_text_seconds', 'output_tokens',
                        'tokens_per_second', 'peak_allocated_gib', 'peak_reserved_gib'):
                values = [r[key] for r in selected if r[key] is not None]
                result[key + '_median'] = statistics.median(values) if values else None
            result['input_tokens'] = sorted({r['input_tokens'] for r in selected})
            groups.append(result)
    return groups


def worker(args):
    import threading
    import torch
    import transformers
    from transformers import set_seed
    from inference import QwenChat, GENERATION_SETTINGS, MODEL_ID, MODEL_REVISION

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'status.json', {'status': 'running'})
    try:
        original_settings = dict(GENERATION_SETTINGS)
        config = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(), mode=args.worker,
                      model=MODEL_ID, revision=MODEL_REVISION, adapter_run=args.adapter_run,
                      repeats=args.repeats, seed=42, prompts=PROMPTS,
                      generation_settings=original_settings, output_budget=args.tokens,
                      python=platform.python_version(), torch=torch.__version__,
                      transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
                      script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      inference_sha256=hashlib.sha256(Path('inference.py').read_bytes()).hexdigest(),
                      adapter_sha256=hashlib.sha256((Path(args.adapter_run) / 'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
                      note='Both modes load the same PEFT wrapper. Base disables adapter. Local cached load, not OS cold-cache load. Timings are engine-side, not HTTP/UI.')
        write_json(root / 'config.json', config)
        started = time.perf_counter()
        engine = QwenChat(args.adapter_run)
        torch.cuda.synchronize()
        load_seconds = time.perf_counter() - started
        rows = []
        with (root / 'results.jsonl').open('x', encoding='utf-8') as raw:
            for workload in ('natural', 'fixed_tokens'):
                settings = dict(original_settings, max_new_tokens=args.tokens)
                if workload == 'fixed_tokens':
                    settings['min_new_tokens'] = args.tokens
                GENERATION_SETTINGS.clear()
                GENERATION_SETTINGS.update(settings)
                # Each workload has an excluded warm-up. No warm-up rows enter summaries.
                set_seed(42)
                engine.reply_stream([{'role': 'user', 'content': PROMPTS[0]['content']}],
                                    args.worker, threading.Event(), lambda text: None)
                for repeat in range(args.repeats):
                    order = PROMPTS if repeat % 2 == 0 else list(reversed(PROMPTS))
                    for prompt in order:
                        set_seed(42)
                        chunks = []
                        answer, metrics = engine.reply_stream(
                            [{'role': 'user', 'content': prompt['content']}],
                            args.worker, threading.Event(), chunks.append)
                        if workload == 'fixed_tokens' and metrics['output_tokens'] != args.tokens:
                            raise RuntimeError('Fixed output length was not achieved')
                        row = dict(mode=args.worker, workload=workload, prompt_id=prompt['id'],
                                   repeat=repeat + 1, generation_settings=settings,
                                   answer=answer, streamed_text=''.join(chunks), **metrics)
                        row['tokens_per_second'] = row['output_tokens'] / row['seconds']
                        raw.write(json.dumps(row, ensure_ascii=False) + '\n')
                        raw.flush()
                        rows.append(row)
                        print(f"{args.worker} {workload} {prompt['id']} {repeat+1}/{args.repeats}", flush=True)
        write_json(root / 'summary.json', dict(load_seconds=load_seconds, groups=summarize(rows)))
        write_json(root / 'status.json', {'status': 'completed', 'measured_runs': len(rows)})
    except Exception as exc:
        write_json(root / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter-run', default='outputs/lora-data-v2')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--worker', choices=['base', 'lora'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 3 or not 1 <= args.tokens <= 256:
        parser.error('repeats must be >=3; tokens must be in [1,256]')
    if args.worker:
        worker(args)
        return
    # A live model server competes for VRAM and invalidates the intended conditions.
    import httpx
    try:
        httpx.get('http://127.0.0.1:8000/health', timeout=2)
    except httpx.TransportError:
        pass
    else:
        raise RuntimeError('Stop the local backend before benchmarking')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / 'status.json', {'status': 'running', 'order': ['base', 'lora']})
    try:
        for mode in ('base', 'lora'):
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', mode,
                            '--adapter-run', args.adapter_run, '--output-dir', str(root / mode),
                            '--repeats', str(args.repeats), '--tokens', str(args.tokens)], check=True)
        summaries = {mode: json.loads((root / mode / 'summary.json').read_text(encoding='utf-8'))
                     for mode in ('base', 'lora')}
        write_json(root / 'comparison.json', summaries)
        write_json(root / 'status.json', {'status': 'completed'})
    except Exception as exc:
        write_json(root / 'status.json', {'status': 'failed', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
