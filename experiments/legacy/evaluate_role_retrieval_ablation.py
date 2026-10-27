"""Compare A's full Xiao Man card with B's topical fact retrieval on A1."""
import json
import statistics
import hashlib
from datetime import datetime
from pathlib import Path

from transformers import set_seed

from evaluate_role_prompt_baseline import PROFILE, SEED, sha256_bytes, validate_cases
from inference import GENERATION_SETTINGS, QwenChat
from role_knowledge import build_retrieval_messages
from runtime_models import get_profile

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / 'data' / 'role-chat-a1-development.json'
A_DIR = ROOT / 'outputs' / 'role-chat-a1-prompt-baseline'
OUTPUT_DIR = ROOT / 'outputs' / 'role-chat-a1-retrieval-ablation'
MAX_NEW_TOKENS = 512
CASE_TITLES = {
    'tired-after-work': '下班后很累',
    'listen-without-advice': '只想先说出来',
    'share-good-news': '分享一个好消息',
    'confirmed-coffee-interest': '问已确认的喜好',
    'unknown-bookstore-name': '问没有设定过的细节',
    'corrected-weekend-plan': '更正周末安排',
    'remember-drink-preference': '记住饮料偏好',
    'user-changes-request': '用户改变主意',
    'unknown-role-experience': '问角色未确认的日常经历',
    'basic-calculation': '普通计算',
}


def load_inputs():
    case_bytes = CASES_PATH.read_bytes()
    cases = json.loads(case_bytes)
    validate_cases(cases)
    a_protocol = json.loads((A_DIR / 'protocol.json').read_text(encoding='utf-8'))
    a_results = json.loads((A_DIR / 'raw-results.json').read_text(encoding='utf-8'))
    if a_protocol['dataset_sha256'] != sha256_bytes(case_bytes):
        raise ValueError('A1 问题集与已经运行的 A 基线不一致')
    if a_protocol['model_id'] != 'Qwen/Qwen3-1.7B' or a_protocol['model_revision'] != get_profile(PROFILE)['revision']:
        raise ValueError('A1 模型版本与 B 阶段配置不一致')
    if a_protocol['generation_settings'].get('max_new_tokens') != MAX_NEW_TOKENS:
        raise ValueError('A1 与 B 的回答长度上限不一致')
    return case_bytes, cases, a_protocol, a_results


def make_score_sheet(cases, a_results, b_results, answer_map, output_path):
    a_by_id = {row['case_id']: row for row in a_results if row['variant'] == 'xiaoman_role_card'}
    b_by_id = {row['case_id']: row for row in b_results}
    lines = [
        '# B 阶段角色资料检索对照：作者体验评分', '',
        '每段完整对话只选一项：□ 左边更自然　□ 右边更自然　□ 差不多　□ 两边都不满意。',
        '只判断你更愿意继续聊哪一个；可以把明显编造、没接住话或其他原因写在原因栏。',
        '左右条件按每题分别打乱。题目和 A 阶段相同，所以这是开发对照，不是新的独立测试集。', ''
    ]
    for index, case in enumerate(cases, 1):
        left_variant, right_variant = answer_map[case['id']]
        rows = {'role_card': a_by_id[case['id']], 'retrieval': b_by_id[case['id']]}
        left, right = rows[left_variant], rows[right_variant]
        lines.extend([f'## {index}. {CASE_TITLES[case["id"]]}', ''])
        for turn_index, user_text in enumerate(case['user_turns']):
            lines.append('你：' + user_text)
            lines.append('左边：' + left['turns'][turn_index]['answer'])
            lines.append('右边：' + right['turns'][turn_index]['answer'])
            lines.append('')
        lines.extend(['选择：□ 左边更自然　□ 右边更自然　□ 差不多　□ 两边都不满意', '原因（可留空）：', ''])
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    case_bytes, data, a_protocol, a_results = load_inputs()
    if OUTPUT_DIR.exists():
        raise FileExistsError('Refusing to overwrite B-stage evidence: ' + str(OUTPUT_DIR))
    profile = get_profile(PROFILE)
    source_manifest = ROOT / profile['local_path'] / 'source.json'
    if not source_manifest.is_file():
        raise FileNotFoundError('固定版本 Qwen3-1.7B 本地模型未找到：' + str(source_manifest))
    source_info = json.loads(source_manifest.read_text(encoding='utf-8'))
    if source_info.get('model_id') != profile['model_id'] or source_info.get('revision') != profile['revision']:
        raise ValueError('本地模型来源与冻结 profile 不匹配')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    records = []
    engine = QwenChat(model_profile=PROFILE, trim_history=False)
    for case in data['cases']:
        set_seed(SEED)
        history = []
        turns = []
        for user_text in case['user_turns']:
            messages, retrieved = build_retrieval_messages(history, user_text)
            answer, metrics = engine.reply(messages, model_mode='base', max_new_tokens=MAX_NEW_TOKENS)
            turns.append(dict(user=user_text, answer=answer, metrics=metrics,
                              retrieved_facts=retrieved, messages=messages))
            history.extend([dict(role='user', content=user_text), dict(role='assistant', content=answer)])
        records.append(dict(case_id=case['id'], focus=case['focus'], variant='retrieval', turns=turns))
        (OUTPUT_DIR / 'raw-results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
        print('Recorded ' + case['id'] + ' / retrieval', flush=True)

    answer_map = {}
    for case in data['cases']:
        bit = hashlib.sha256((case['id'] + data['dataset_id'] + '-retrieval').encode('utf-8')).digest()[0] & 1
        variants = ['role_card', 'retrieval']
        answer_map[case['id']] = variants[bit:] + variants[:bit]
    (OUTPUT_DIR / 'answer-map.json').write_text(json.dumps(answer_map, ensure_ascii=False, indent=2), encoding='utf-8')
    make_score_sheet(data['cases'], a_results, records, answer_map, OUTPUT_DIR / 'score-sheet.md')

    all_turns = [turn for row in records for turn in row['turns']]
    retrieved_counts = [len(turn['retrieved_facts']) for turn in all_turns]
    metrics = [turn['metrics'] for turn in all_turns]
    summaries = dict(
        scenarios=len(records), replies=len(all_turns),
        turns_with_retrieval=sum(count > 0 for count in retrieved_counts),
        median_retrieved_facts=statistics.median(retrieved_counts),
        retrieved_record_ids=[row['id'] for turn in all_turns for row in turn['retrieved_facts']],
        median_input_tokens=statistics.median(m['input_tokens'] for m in metrics),
        median_output_tokens=statistics.median(m['output_tokens'] for m in metrics),
        max_token_limit_hits=sum(m['output_tokens'] >= MAX_NEW_TOKENS and not m['ended_with_eos'] for m in metrics),
        eos_ended=sum(bool(m['ended_with_eos']) for m in metrics),
    )
    (OUTPUT_DIR / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    protocol = dict(
        created_at=datetime.now().astimezone().isoformat(),
        dataset_id=data['dataset_id'], dataset_sha256=sha256_bytes(case_bytes),
        model_id=profile['model_id'], model_revision=profile['revision'],
        local_model_source_sha256=sha256_bytes(source_manifest.read_bytes()),
        generation_settings={**GENERATION_SETTINGS, 'max_new_tokens': MAX_NEW_TOKENS},
        seed=SEED, enable_thinking=False, input_token_budget=2048,
        retrieval_implementation='Deterministic lexical matching over confirmed profile facts; no embedding/vector database.',
        role_knowledge_source='characters.py:CHARACTERS.companion.identity',
        baseline_protocol_sha256=sha256_bytes((A_DIR / 'protocol.json').read_bytes()),
        baseline_raw_results_sha256=sha256_bytes((A_DIR / 'raw-results.json').read_bytes()),
        note='Development A-vs-B comparison only; B retrieves static role facts for the current user turn. No training or final independent evaluation.'
    )
    (OUTPUT_DIR / 'protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
