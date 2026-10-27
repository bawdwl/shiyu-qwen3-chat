"""Run the author-approved natural-chat holdout on the fixed Qwen3-1.7B model."""
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

import torch
from transformers import set_seed

from characters import conversation_messages
from evaluate_role_prompt_baseline import PROFILE, SEED, sha256_bytes
from inference import GENERATION_SETTINGS, QwenChat
from role_knowledge import build_additive_retrieval_messages, retrieve_role_facts
from runtime_models import get_profile

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / 'data' / 'role-chat-c1-natural-holdout.json'
OUTPUT_DIR = ROOT / 'outputs' / 'role-chat-c1-natural-holdout'
MAX_NEW_TOKENS = 512
VARIANTS = ('role_card', 'role_card_plus_retrieval')
EXPECTED_RETRIEVAL = {
    'confirmed-role-facts': {'identity.age', 'identity.city', 'identity.occupation'},
    'unknown-education': {'unknown.education'},
    'unknown-hometown': {'unknown.hometown'},
    'unknown-walking-route': {'interest.walking'},
    'unknown-bookstore-story': {'unknown.work_experiences'},
}


def load_frozen_cases():
    case_bytes = CASES_PATH.read_bytes()
    data = json.loads(case_bytes)
    manifest_path = OUTPUT_DIR / 'freeze-manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError('缺少冻结清单；先完成题集冻结，再运行推理。')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('cases_sha256') != sha256_bytes(case_bytes):
        raise ValueError('冻结题集哈希不匹配，停止运行。')
    if data.get('dataset_id') != 'xiaoman-c1-natural-holdout-v1':
        raise ValueError('不是 C1 自然聊天保留评测集。')
    cases = data.get('cases')
    if not isinstance(cases, list) or len(cases) != 9:
        raise ValueError('C1 题集必须包含作者批准的 9 道题。')
    ids = [case.get('id') for case in cases]
    if len(set(ids)) != len(ids) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError('题目 id 必须唯一且非空。')
    for case in cases:
        turns = case.get('user_turns')
        if not turns or any(not isinstance(text, str) or not text.strip() for text in turns):
            raise ValueError('每个案例必须包含非空对话轮次：' + str(case.get('id')))
        facts = {row['id'] for text in turns for row in retrieve_role_facts(text)}
        expected = EXPECTED_RETRIEVAL.get(case['id'], set())
        if not expected <= facts:
            raise ValueError(f"{case['id']} 未命中预期角色资料：缺少 {sorted(expected - facts)}")
    return case_bytes, data, manifest


def make_score_sheet(cases, results, answer_map, output_path):
    by_id = {row['case_id']: row for row in results}
    lines = [
        '# 小满自然聊天：匿名作者体验对照', '',
        '每题选择：□ 左边更自然　□ 右边更自然　□ 差不多　□ 两边都不满意。',
        '只按你真实聊天时更愿意继续聊哪组来选；不需要推测哪组用了检索。若不满意，可写一句原因。',
        '左右条件名称和对应关系另存于本地 answer-map.json，不在本表显示。两组使用同一基础模型、角色卡和生成参数。',
        '题目只用于评测，没有加入训练数据。', ''
    ]
    for index, case in enumerate(cases, 1):
        left_variant, right_variant = answer_map[case['id']]
        row = by_id[case['id']]
        lines.extend([f"## {index}. {case['focus']}", ''])
        for turn_index, turn in enumerate(row['turns'], 1):
            if len(case['user_turns']) > 1:
                lines.append(f'### 第 {turn_index} 轮')
            lines.append('你：' + turn['user'])
            lines.append('左边：' + turn['conditions'][left_variant]['answer'])
            lines.append('右边：' + turn['conditions'][right_variant]['answer'])
            lines.append('')
        lines.extend([
            '选择：□ 左边更自然　□ 右边更自然　□ 差不多　□ 两边都不满意',
            '原因（可留空）：', ''
        ])
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError('先冻结题集并创建输出目录：' + str(OUTPUT_DIR))
    existing = [OUTPUT_DIR / name for name in ('protocol.json', 'raw-results.json', 'answer-map.json')]
    if any(path.exists() for path in existing):
        raise FileExistsError('输出目录已有模型运行证据，拒绝覆盖：' + str(OUTPUT_DIR))

    case_bytes, data, freeze_manifest = load_frozen_cases()
    profile = get_profile(PROFILE)
    source_manifest = ROOT / profile['local_path'] / 'source.json'
    if not source_manifest.is_file():
        raise FileNotFoundError('固定版本 Qwen3-1.7B 本地模型未找到：' + str(source_manifest))
    source_info = json.loads(source_manifest.read_text(encoding='utf-8'))
    if source_info.get('model_id') != profile['model_id'] or source_info.get('revision') != profile['revision']:
        raise ValueError('本地模型来源与冻结 profile 不匹配。')

    role_prompt = conversation_messages(dict(character_id='companion', messages=[]))[0]['content']
    protocol = {
        'created_at': datetime.now().astimezone().isoformat(),
        'dataset_id': data['dataset_id'],
        'cases_sha256': sha256_bytes(case_bytes),
        'freeze_manifest_sha256': sha256_bytes((OUTPUT_DIR / 'freeze-manifest.json').read_bytes()),
        'model_id': profile['model_id'], 'model_revision': profile['revision'],
        'local_model_source_sha256': sha256_bytes(source_manifest.read_bytes()),
        'variants': list(VARIANTS),
        'xiaoman_system_prompt_sha256': sha256_bytes(role_prompt.encode('utf-8')),
        'generation_settings': {**GENERATION_SETTINGS, 'max_new_tokens': MAX_NEW_TOKENS},
        'seed': SEED, 'enable_thinking': False, 'input_token_budget': 2048,
        'runtime': {'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                    'cuda_available': torch.cuda.is_available(), 'dtype': 'bfloat16'},
        'changed_factor': 'Only append deterministically retrieved relevant character facts to the unchanged A-stage full role-card prompt.',
        'retrieval_version': 'lexical-v2-natural-query-aliases',
        'retrieval_coverage': {
            case['id']: [row['id'] for text in case['user_turns'] for row in retrieve_role_facts(text)]
            for case in data['cases']
        },
        'rating_policy': 'Author preference is reported only after the author submits actual ratings; no automatic judgement is represented as human rating.',
        'prohibitions': ['not training data', 'do not tune on this holdout', 'do not merge with historical LCCC/shared-role results'],
    }
    (OUTPUT_DIR / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUTPUT_DIR / 'cases.snapshot.json').write_bytes(case_bytes)

    engine = QwenChat(model_profile=PROFILE, trim_history=False)
    results = []
    for case in data['cases']:
        turn_rows = []
        for variant in VARIANTS:
            set_seed(SEED)
            history = []
            turns = []
            for user_text in case['user_turns']:
                a_messages = conversation_messages(dict(character_id='companion', messages=history), user_text)
                if variant == 'role_card':
                    messages, facts = a_messages, []
                else:
                    messages, facts = build_additive_retrieval_messages(history, user_text)
                    if messages[0]['content'][:len(a_messages[0]['content'])] != a_messages[0]['content']:
                        raise AssertionError('追加检索改动了 A 阶段完整角色卡。')
                    if messages[1:] != a_messages[1:]:
                        raise AssertionError('追加检索改动了 A 阶段对话历史或当前问题。')
                answer, metrics = engine.reply(messages, model_mode='base', max_new_tokens=MAX_NEW_TOKENS)
                turns.append(dict(user=user_text, answer=answer, metrics=metrics,
                                  retrieved_fact_ids=[row['id'] for row in facts]))
                history.extend([dict(role='user', content=user_text), dict(role='assistant', content=answer)])
            turn_rows.append((variant, turns))
        a_turns = turn_rows[0][1]
        b_turns = turn_rows[1][1]
        paired_turns = []
        for index, user_text in enumerate(case['user_turns']):
            paired_turns.append(dict(
                user=user_text,
                conditions={
                    'role_card': {key: a_turns[index][key] for key in ('answer', 'metrics', 'retrieved_fact_ids')},
                    'role_card_plus_retrieval': {key: b_turns[index][key] for key in ('answer', 'metrics', 'retrieved_fact_ids')},
                }
            ))
        results.append(dict(case_id=case['id'], focus=case['focus'], turns=paired_turns))
        (OUTPUT_DIR / 'raw-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"已记录 {case['id']}（{len(results)}/{len(data['cases'])}）", flush=True)

    answer_map = {}
    for case in data['cases']:
        bit = hashlib.sha256((case['id'] + data['dataset_id'] + '-C1').encode('utf-8')).digest()[0] & 1
        answer_map[case['id']] = list(VARIANTS[bit:] + VARIANTS[:bit])
    (OUTPUT_DIR / 'answer-map.json').write_text(json.dumps(answer_map, ensure_ascii=False, indent=2), encoding='utf-8')
    make_score_sheet(data['cases'], results, answer_map, OUTPUT_DIR / 'author-rating-sheet.md')

    metrics = {variant: [turn['conditions'][variant]['metrics']
                         for case in results for turn in case['turns']]
               for variant in VARIANTS}
    summary = {
        variant: {
            'reply_count': len(rows),
            'median_output_tokens': statistics.median(row['output_tokens'] for row in rows),
            'max_token_limit_hits': sum(row['output_tokens'] >= MAX_NEW_TOKENS and not row['ended_with_eos'] for row in rows),
            'median_generation_seconds': statistics.median(row['seconds'] for row in rows),
            'median_peak_allocated_gib': statistics.median(row['peak_allocated_gib'] for row in rows),
        } for variant, rows in metrics.items()
    }
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；本阶段不自动退回 CPU。')
    main()
