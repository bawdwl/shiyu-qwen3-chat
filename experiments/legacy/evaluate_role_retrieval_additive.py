"""B2 single-factor comparison: append retrieved facts to the unchanged A prompt."""
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

from transformers import set_seed

from characters import conversation_messages
from evaluate_role_prompt_baseline import PROFILE, SEED, sha256_bytes, validate_cases
from inference import GENERATION_SETTINGS, QwenChat
from role_knowledge import build_additive_retrieval_messages
from runtime_models import get_profile

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / 'data' / 'role-chat-a1-development.json'
A_DIR = ROOT / 'outputs' / 'role-chat-a1-prompt-baseline'
OUTPUT_DIR = ROOT / 'outputs' / 'role-chat-a1-retrieval-additive'
MAX_NEW_TOKENS = 512
TARGET_CASE_IDS = (
    'confirmed-coffee-interest',
    'unknown-bookstore-name',
    'unknown-role-experience',
)
CASE_TITLES = {
    'confirmed-coffee-interest': '问已确认的喜好',
    'unknown-bookstore-name': '问没有设定过的细节',
    'unknown-role-experience': '问角色未确认的日常经历',
}


def load_inputs():
    case_bytes = CASES_PATH.read_bytes()
    data = json.loads(case_bytes)
    validate_cases(data)
    a_protocol = json.loads((A_DIR / 'protocol.json').read_text(encoding='utf-8'))
    a_results = json.loads((A_DIR / 'raw-results.json').read_text(encoding='utf-8'))
    if a_protocol['dataset_sha256'] != sha256_bytes(case_bytes):
        raise ValueError('A1 问题集与已运行的 A 基线不一致')
    if a_protocol['model_id'] != 'Qwen/Qwen3-1.7B' or a_protocol['model_revision'] != get_profile(PROFILE)['revision']:
        raise ValueError('A1 模型版本与 B2 配置不一致')
    if a_protocol['generation_settings'].get('max_new_tokens') != MAX_NEW_TOKENS:
        raise ValueError('A1 与 B2 的回答长度上限不一致')
    a_by_id = {
        row['case_id']: row for row in a_results
        if row['variant'] == 'xiaoman_role_card'
    }
    cases_by_id = {row['id']: row for row in data['cases']}
    for case_id in TARGET_CASE_IDS:
        if case_id not in a_by_id or case_id not in cases_by_id:
            raise ValueError('A1 基线缺少 B2 目标案例：' + case_id)
        if len(cases_by_id[case_id]['user_turns']) != 1:
            raise ValueError('B2 目标问题必须是单轮案例：' + case_id)
    return case_bytes, data, a_protocol, a_by_id, cases_by_id


def make_score_sheet(target_cases, a_by_id, b2_rows, answer_map, output_path):
    b2_by_id = {row['case_id']: row for row in b2_rows}
    lines = [
        '# B2 角色资料追加检索：作者体验评分', '',
        '每题只选择一项：□ 左边更自然　□ 右边更自然　□ 差不多　□ 两边都不满意。',
        '只判断你更愿意继续聊哪一个；也可以把明显编造、没接住话或其他原因写在原因栏。',
        '两边使用完全相同的小满完整角色卡。右侧条件只在问题命中角色资料时追加对应资料；左右顺序逐题打乱。',
        '这三题来自已看过的 A1 开发集，不是新的独立测试，也不用于训练。', ''
    ]
    for index, case in enumerate(target_cases, 1):
        left_variant, right_variant = answer_map[case['id']]
        paired = {
            'role_card': a_by_id[case['id']]['turns'][0]['answer'],
            'role_card_plus_retrieval': b2_by_id[case['id']]['conditions']['role_card_plus_retrieval']['answer'],
        }
        user_text = case['user_turns'][0]
        lines.extend([
            f'## {index}. {CASE_TITLES[case["id"]]}', '',
            '你：' + user_text,
            '左边：' + paired[left_variant],
            '右边：' + paired[right_variant], '',
            '选择：□ 左边更自然　□ 右边更自然　□ 差不多　□ 两边都不满意',
            '原因（可留空）：', ''
        ])
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    case_bytes, data, a_protocol, a_by_id, cases_by_id = load_inputs()
    existing_rows = None
    if OUTPUT_DIR.exists():
        paired_path = OUTPUT_DIR / 'paired-results.json'
        finalized = [OUTPUT_DIR / name for name in ('score-sheet.md', 'summary.json', 'protocol.json')]
        if not paired_path.is_file() or any(path.exists() for path in finalized):
            raise FileExistsError('Refusing to overwrite existing B2 evidence: ' + str(OUTPUT_DIR))
        existing_rows = json.loads(paired_path.read_text(encoding='utf-8'))
        if [row['case_id'] for row in existing_rows] != list(TARGET_CASE_IDS):
            raise ValueError('已有 B2 原始结果不完整或顺序不符，不继续覆盖')
    profile = get_profile(PROFILE)
    source_manifest = ROOT / profile['local_path'] / 'source.json'
    if not source_manifest.is_file():
        raise FileNotFoundError('固定版本 Qwen3-1.7B 本地模型未找到：' + str(source_manifest))
    source_info = json.loads(source_manifest.read_text(encoding='utf-8'))
    if source_info.get('model_id') != profile['model_id'] or source_info.get('revision') != profile['revision']:
        raise ValueError('本地模型来源与冻结 profile 不匹配')

    target_cases = [cases_by_id[case_id] for case_id in TARGET_CASE_IDS]
    if existing_rows is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
        (OUTPUT_DIR / 'cases.snapshot.json').write_bytes(case_bytes)
        engine = QwenChat(model_profile=PROFILE, trim_history=False)
        paired_rows = []
        for case in target_cases:
            user_text = case['user_turns'][0]
            a_messages = conversation_messages(dict(character_id='companion', messages=[]), user_text)
            messages, facts = build_additive_retrieval_messages([], user_text)
            if messages[0]['content'][:len(a_messages[0]['content'])] != a_messages[0]['content']:
                raise AssertionError('B2 改动了 A 的完整系统提示：' + case['id'])
            if messages[1:] != a_messages[1:]:
                raise AssertionError('B2 改动了 A 的对话消息：' + case['id'])
            if not facts:
                raise AssertionError('B2 目标案例未命中任何角色资料：' + case['id'])

            set_seed(SEED)
            answer, metrics = engine.reply(messages, model_mode='base', max_new_tokens=MAX_NEW_TOKENS)
            a_answer = a_by_id[case['id']]['turns'][0]['answer']
            paired_rows.append(dict(
                case_id=case['id'], focus=case['focus'],
                conditions={
                    'role_card': dict(answer=a_answer, messages=a_messages),
                    'role_card_plus_retrieval': dict(answer=answer, messages=messages,
                                                     metrics=metrics, retrieved_facts=facts),
                },
                unchanged_a_prompt_sha256=sha256_bytes(a_messages[0]['content'].encode('utf-8')),
            ))
            (OUTPUT_DIR / 'paired-results.json').write_text(
                json.dumps(paired_rows, ensure_ascii=False, indent=2), encoding='utf-8')
            print('Recorded ' + case['id'] + ' / A vs A+retrieval', flush=True)
    else:
        paired_rows = existing_rows

    answer_map_path = OUTPUT_DIR / 'answer-map.json'
    if answer_map_path.exists():
        answer_map = json.loads(answer_map_path.read_text(encoding='utf-8'))
        if set(answer_map) != set(TARGET_CASE_IDS):
            raise ValueError('已有 B2 左右映射与目标问题不符')
    else:
        answer_map = {}
        for case in target_cases:
            bit = hashlib.sha256((case['id'] + data['dataset_id'] + '-B2').encode('utf-8')).digest()[0] & 1
            variants = ['role_card', 'role_card_plus_retrieval']
            answer_map[case['id']] = variants[bit:] + variants[:bit]
        answer_map_path.write_text(json.dumps(answer_map, ensure_ascii=False, indent=2), encoding='utf-8')
    make_score_sheet(target_cases, a_by_id, paired_rows, answer_map, OUTPUT_DIR / 'score-sheet.md')

    metrics = [row['conditions']['role_card_plus_retrieval']['metrics'] for row in paired_rows]
    summary = dict(
        scenarios=len(paired_rows), replies=len(paired_rows),
        retrieval_hits=sum(bool(row['conditions']['role_card_plus_retrieval']['retrieved_facts']) for row in paired_rows),
        retrieved_record_ids=[fact['id'] for row in paired_rows
                              for fact in row['conditions']['role_card_plus_retrieval']['retrieved_facts']],
        eos_ended=sum(bool(metric['ended_with_eos']) for metric in metrics),
        output_limit_hits=sum(metric['output_tokens'] >= MAX_NEW_TOKENS and not metric['ended_with_eos'] for metric in metrics),
        median_input_tokens=statistics.median(metric['input_tokens'] for metric in metrics),
        median_output_tokens=statistics.median(metric['output_tokens'] for metric in metrics),
    )
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    protocol = dict(
        created_at=datetime.now().astimezone().isoformat(), dataset_id=data['dataset_id'],
        dataset_sha256=sha256_bytes(case_bytes), target_case_ids=list(TARGET_CASE_IDS),
        model_id=profile['model_id'], model_revision=profile['revision'],
        local_model_source_sha256=sha256_bytes(source_manifest.read_bytes()),
        generation_settings={**GENERATION_SETTINGS, 'max_new_tokens': MAX_NEW_TOKENS},
        seed=SEED, enable_thinking=False, input_token_budget=2048,
        changed_factor='Append retrieved facts to the unchanged A-stage full role-card system prompt; no other messages or settings changed.',
        retrieval_implementation='Deterministic lexical matching over canonical confirmed/unconfirmed role records; no embedding/vector database.',
        baseline_protocol_sha256=sha256_bytes((A_DIR / 'protocol.json').read_bytes()),
        baseline_raw_results_sha256=sha256_bytes((A_DIR / 'raw-results.json').read_bytes()),
        note='B2 is a targeted development comparison on three already-reviewed A1 questions, not an independent final test and not training data.'
    )
    (OUTPUT_DIR / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
