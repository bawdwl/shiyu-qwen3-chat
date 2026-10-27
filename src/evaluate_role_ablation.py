"""Paired fixed-context development outputs: base, old LoRA, role-only LoRA.

No automated naturalness scores; this exposed developer set is not blind evidence.
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_control(old, new):
    keys = ('base_model', 'revision', 'seed', 'epochs', 'micro_batch', 'gradient_accumulation',
            'learning_rate', 'max_length', 'r', 'lora_alpha', 'lora_dropout', 'target_modules',
            'dtype', 'gradient_checkpointing', 'optimizer', 'weight_decay', 'max_grad_norm', 'schedule')
    differences = [key for key in keys if old[key] != new[key]]
    if old['data']['sha256'] != new['data']['sha256']:
        differences.append('data.sha256')
    if differences:
        raise ValueError('Uncontrolled training differences: ' + ', '.join(differences))
    if old.get('conditioning', {}).get('system_prompt'):
        raise ValueError('Old run must be the no-system condition')
    prompt = new.get('conditioning', {}).get('system_prompt')
    if not prompt or new['conditioning']['sha256'] != hashlib.sha256(prompt.encode('utf-8')).hexdigest():
        raise ValueError('New run has no valid system-prompt snapshot')
    return list(keys) + ['data.sha256']


def summarize(records):
    summary = {}
    for variant in ('base', 'old_lora', 'role_lora'):
        for kind in ('chat', 'capability_regression'):
            rows = [row for row in records if row['variant'] == variant and row.get('kind', 'chat') == kind]
            if rows:
                summary.setdefault(variant, {})[kind] = dict(replies=len(rows),
                                        median_output_tokens=statistics.median(r['metrics']['output_tokens'] for r in rows),
                                        replies_le5_tokens=sum(r['metrics']['output_tokens'] <= 5 for r in rows),
                                        truncated=sum(not r['metrics']['ended_with_eos'] for r in rows))
    return summary


def render(records, cases):
    lines = ['# 角色条件单因素实验：开发回答对照', '',
             '这些是原创、已公开给开发者的诊断场景，不是训练数据或独立盲评。长度不是自然度分数。',
             '每种模型使用相同固定上下文；不把各自生成的上一轮答案带入另一个比较。', '']
    for case in cases:
        lines += ['## ' + case['id'], '']
        for message in case['messages']:
            lines += [f"- {message['role']}：{message['content']}"]
        if case.get('rubric'):
            lines += ['', '**参考检查点（不是模型回答）：** ' + case['rubric']]
        for variant in ('base', 'old_lora', 'role_lora'):
            rows = [r for r in records if r['case_id'] == case['id'] and r['variant'] == variant]
            for row in rows:
                lines += ['', f"**{variant} / seed={row['seed']}：** {row['answer']}"]
        lines.append('')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-run', default='outputs/lora-lccc-base-v1-pilot')
    parser.add_argument('--new-run', default='outputs/lora-lccc-base-v2-role-only')
    parser.add_argument('--cases', default='data/im_chat_development_v2.json')
    parser.add_argument('--output-dir', default='outputs/lccc-role-v2-development')
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261023, 20261024])
    parser.add_argument('--case-kind', choices=['chat', 'capability_regression'], default=None)
    parser.add_argument('--use-saved-prompt', action='store_true', help='显式复现历史角色提示，而不是当前App提示')
    args = parser.parse_args()
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error('Seeds must be nonempty and unique')
    old_root, new_root = Path(args.old_run), Path(args.new_run)
    old = json.loads((old_root / 'run-config.json').read_text(encoding='utf-8'))
    new = json.loads((new_root / 'run-config.json').read_text(encoding='utf-8'))
    controlled = validate_control(old, new)
    system = new['conditioning']['system_prompt']
    current = conversation_messages(dict(character_id='companion', messages=[]))[0]['content']
    if system != current and not args.use_saved_prompt:
        raise ValueError('App prompt changed; use --use-saved-prompt to reproduce the historical snapshot')
    cases_bytes = Path(args.cases).read_bytes()
    cases = json.loads(cases_bytes)['cases']
    if args.case_kind:
        cases = [case for case in cases if case.get('kind', 'chat') == args.case_kind]
    if not cases or len({case['id'] for case in cases}) != len(cases):
        raise ValueError('Empty or duplicate development case IDs')
    for case in cases:
        messages = case['messages']
        if not messages or messages[-1]['role'] != 'user' or any(
            message['role'] != ('user' if i % 2 == 0 else 'assistant') or not message['content'].strip()
            for i, message in enumerate(messages)
        ):
            raise ValueError('Invalid fixed-context messages: ' + case['id'])
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'cases.snapshot.json').write_bytes(cases_bytes)
    config = dict(timestamp=datetime.now().astimezone().isoformat(), seeds=args.seeds,
                  model_id=new['base_model'], revision=new['revision'], controlled_keys=controlled,
                  changed_factor='system prompt in training/validation only',
                  system_prompt_sha256=new['conditioning']['sha256'], system_prompt=system,
                  current_app_prompt_matches_snapshot=(system == current),
                  development_cases_sha256=hashlib.sha256(cases_bytes).hexdigest(),
                  selected_case_ids=[case['id'] for case in cases], case_kind=args.case_kind,
                  generation_settings=GENERATION_SETTINGS,
                  adapters={name: digest(path / 'adapter' / 'adapter_model.safetensors')
                            for name, path in [('old_lora', old_root), ('role_lora', new_root)]},
                  note='Exposed developer-authored cases, fixed contexts, two generation seeds; no human/AI quality scores or fair speed benchmark.')
    (root / 'run-config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    records = []
    try:
        for run, variants in [(old_root, [('base', 'base'), ('old_lora', 'lora')]),
                              (new_root, [('role_lora', 'lora')])]:
            engine = QwenChat(adapter_run=run, model_profile='qwen3-1.7b', trim_history=True)
            for variant, mode in variants:
                for case in cases:
                    messages = [dict(role='system', content=system)] + case['messages']
                    for seed in args.seeds:
                        set_seed(seed)
                        answer, metrics = engine.reply(messages, model_mode=mode)
                        records.append(dict(case_id=case['id'], kind=case.get('kind', 'chat'), variant=variant, seed=seed,
                                            answer=answer, metrics=metrics))
                        (root / 'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
                        print(f"Recorded {variant} {case['id']} seed={seed}", flush=True)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        (root / 'summary.json').write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2), encoding='utf-8')
        (root / 'answers.md').write_text(render(records, cases), encoding='utf-8')
        (root / 'status.json').write_text(json.dumps(dict(status='completed', replies=len(records))), encoding='utf-8')
    except Exception as exc:
        (root / 'status.json').write_text(json.dumps(dict(status='failed', error=str(exc)), ensure_ascii=False), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
