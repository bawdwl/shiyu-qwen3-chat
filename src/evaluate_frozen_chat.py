"""Generate a new source-hidden App-style packet from frozen real utterances."""
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

from transformers import set_seed

from characters import conversation_messages
from inference import GENERATION_SETTINGS, QwenChat


def assemble_review(cases, results):
    expected = {case['id'] for case in cases}
    if len(expected) != len(cases) or len(results) != len(cases) * 2:
        raise ValueError('Incomplete or duplicate review cases')
    grouped = {case_id: {} for case_id in expected}
    for row in results:
        if row['case_id'] not in expected or row['label'] not in ('A', 'B'):
            raise ValueError('Unknown case or label in generated results')
        rows = grouped[row['case_id']]
        if row['label'] in rows or not row['answer'].strip():
            raise ValueError('Duplicate label or empty answer')
        rows[row['label']] = row
    packet, key = [], {}
    for case in cases:
        rows = grouped[case['id']]
        if set(rows) != {'A', 'B'} or {row['mode'] for row in rows.values()} != {'base', 'lora'}:
            raise ValueError('Each case needs exactly one base and one LoRA answer')
        key[case['id']] = {label: rows[label]['mode'] for label in ('A', 'B')}
        packet.append(dict(case_id=case['id'], messages=case['messages'],
                           answers={label: rows[label]['answer'] for label in ('A', 'B')}))
    return packet, key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/lora-lccc-base-v2-role-only')
    parser.add_argument('--frozen-dir', default='outputs/lccc-real-chat-review-v2-frozen')
    parser.add_argument('--output-dir', default='outputs/lccc-real-chat-review-v2')
    parser.add_argument('--finalize-only', action='store_true',
                        help='只从已保存的完整回答恢复未完成的评分包，不重新生成，不覆盖已完成批次')
    parser.add_argument('--use-saved-prompt', action='store_true', help='显式使用训练时的历史提示，而不是当前App提示')
    args = parser.parse_args()
    frozen = Path(args.frozen_dir)
    freeze = json.loads((frozen / 'freeze-manifest.json').read_text(encoding='utf-8'))
    data = (frozen / 'cases.json').read_bytes()
    if hashlib.sha256(data).hexdigest() != freeze['cases_sha256']:
        raise ValueError('Frozen prompts were changed')
    cases = json.loads(data)
    run = Path(args.run_dir)
    config = json.loads((run / 'run-config.json').read_text(encoding='utf-8'))
    if config['data']['sha256'] != freeze['training_data_sha256']:
        raise ValueError('Training data differs from the overlap audit')
    system = config['conditioning']['system_prompt']
    current = conversation_messages(dict(character_id='companion', messages=[]))[0]['content']
    if system != current and not args.use_saved_prompt:
        raise ValueError('App role prompt changed; use --use-saved-prompt to reproduce historical conditions')
    root = Path(args.output_dir)
    if args.finalize_only and (not root.is_dir() or (root / 'run-config.json').exists()):
        raise ValueError('Finalize only supports an existing incomplete batch')
    root.mkdir(parents=True, exist_ok=args.finalize_only)
    packet_dir = root / 'reviewer-packet'
    packet_dir.mkdir(exist_ok=args.finalize_only)
    rng = random.SystemRandom()
    packet, key, results = [], {}, []
    seed = 20261025
    if args.finalize_only:
        results = json.loads((root / 'private-results.json').read_text(encoding='utf-8'))
    else:
        engine = QwenChat(adapter_run=run, model_profile=config['model_profile'], trim_history=True)
    for case in ([] if args.finalize_only else cases):
        modes = ['base', 'lora']
        rng.shuffle(modes)
        mapping = dict(zip(('A', 'B'), modes))
        answers = {}
        for label, mode in mapping.items():
            set_seed(seed)
            answer, metrics = engine.reply([dict(role='system', content=system)] + case['messages'], model_mode=mode)
            answers[label] = answer
            results.append(dict(case_id=case['id'], label=label, mode=mode, answer=answer, metrics=metrics))
        key[case['id']] = mapping
        packet.append(dict(case_id=case['id'], messages=case['messages'], answers=answers))
        # Keep completed pairs if a later generation fails; a partial batch is not complete evidence.
        (root / 'private-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Generated ' + case['id'], flush=True)
    packet, key = assemble_review(cases, results)
    lines = ['# 日常角色聊天回复盲评', '',
             '这次只看一句你可能发给角色的话。用户原话选自未用于之前盲评的真人公开对话，保留原话，仅规范空格；不需要扮演原对话中的某个人。',
             '这里评估单轮接话，不测试长期记忆。选题在生成本轮回答前完成。A/B逐题随机，两边都使用同一小满角色设定。',
             '每题勾一个结果。差不多表示难分高下，不代表都满意；如果两份都不想继续聊，请选两个都不自然。原因可不填。', '']
    for index, case in enumerate(packet, 1):
        lines += [f'## 第 {index} 题', '', '**你说：** ' + case['messages'][0]['content'], '',
                  '**回答 A：** ' + case['answers']['A'], '', '**回答 B：** ' + case['answers']['B'], '',
                  '- [ ] A 更自然', '- [ ] B 更自然', '- [ ] 差不多', '- [ ] 两个都不自然', '',
                  '原因（可不填）：', '', '---', '']
    (packet_dir / 'score-sheet.md').write_text('\n'.join(lines), encoding='utf-8')
    (packet_dir / 'scores.jsonl').write_text(''.join(json.dumps(dict(case_id=case['case_id'], preference=None, reason=None),
                                                             ensure_ascii=False) + '\n' for case in packet), encoding='utf-8')
    (root / 'organizer-key.json').write_text(json.dumps(dict(mapping=key, note='评分完成前请不要查看来源映射'), ensure_ascii=False, indent=2), encoding='utf-8')
    summary = dict(timestamp=datetime.now().astimezone().isoformat(), status='review-pending',
                   model=config['base_model'], revision=config['revision'],
                   adapter_sha256=hashlib.sha256((run / 'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),
                   system_prompt_sha256=config['conditioning']['sha256'], seed=seed,
                   current_app_prompt_matches_snapshot=(system == current),
                   generation_settings=GENERATION_SETTINGS, freeze_manifest=freeze,
                   case_ids=[case['id'] for case in cases], case_count=len(cases),
                   note='No human or AI scores. A curated 12-case local research pilot cannot establish general improvement.')
    (root / 'run-config.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='review-pending', case_count=len(cases), score_sheet=str(packet_dir / 'score-sheet.md')), ensure_ascii=False))


if __name__ == '__main__':
    main()
