"""Freeze-safe A-stage prompt comparison on Qwen3-1.7B; no model weights are updated."""
import argparse
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

from transformers import set_seed

from characters import conversation_messages
from inference import GENERATION_SETTINGS, QwenChat
from runtime_models import get_profile

ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / 'data' / 'role-chat-a1-development.json'
DEFAULT_OUTPUT = ROOT / 'outputs' / 'role-chat-a1-prompt-baseline'
GENERIC_PROMPT = (
    '你是一个友好、准确的中文聊天助手。请根据用户当前说的话和对话历史自然回应。'
    '表达清楚、简洁，不要编造事实；用户没有要求建议时，不必强行给出方案。'
)
SEED = 20261024
PROFILE = 'qwen3-1.7b'
MAX_NEW_TOKENS = 512


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def validate_cases(data):
    cases = data.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('问题文件必须包含非空 cases 列表')
    ids = [case.get('id') for case in cases]
    if len(set(ids)) != len(ids) or any(not isinstance(item, str) or not item for item in ids):
        raise ValueError('问题场景 id 必须唯一且非空')
    for case in cases:
        turns = case.get('user_turns')
        if not isinstance(turns, list) or not turns or any(not isinstance(text, str) or not text.strip() for text in turns):
            raise ValueError('每个场景必须包含非空 user_turns：' + str(case.get('id')))
    if data.get('dataset_id') != 'xiaoman-a1-development-v1':
        raise ValueError('问题集不是已批准冻结的 A1 开发集')


def render_messages(variant, history, user_text):
    full_history = history + [dict(role='user', content=user_text)]
    if variant == 'general_prompt':
        return [dict(role='system', content=GENERIC_PROMPT), *full_history]
    if variant == 'xiaoman_role_card':
        current = dict(character_id='companion', messages=history)
        return conversation_messages(current, user_text)
    raise ValueError('Unknown prompt variant: ' + variant)


def render_score_sheet(cases, results, answer_map, output_path):
    by_key = {(row['case_id'], row['variant']): row for row in results}
    lines = [
        '# A 阶段提示词对照：作者体验评分', '',
        '请按整段对话选择一项：左边更自然 / 右边更自然 / 差不多 / 两边都不满意。',
        '只评你更愿意继续聊哪一个；不用判断回复长短谁更好。若看到明显编造、没接住话或违背要求，可在原因栏简单写一下。',
        '这里的 A/B 是匿名条件编号，映射单独保存在本地原始记录中。', ''
    ]
    for case in cases:
        left_variant, right_variant = answer_map[case['id']]
        left = by_key[(case['id'], left_variant)]
        right = by_key[(case['id'], right_variant)]
        lines += [f"## {case['id']}", '']
        for index, user_text in enumerate(case['user_turns']):
            lines.append(f'你：{user_text}')
            lines.append(f"左边：{left['turns'][index]['answer']}")
            lines.append(f"右边：{right['turns'][index]['answer']}")
            lines.append('')
        lines += ['选择：左边更自然 / 右边更自然 / 差不多 / 两边都不满意', '原因（可留空）：', '']
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', type=Path, default=DEFAULT_CASES)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--max-new-tokens', type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()
    if not 128 <= args.max_new_tokens <= 1024:
        parser.error('--max-new-tokens must be between 128 and 1024')
    if args.output_dir.exists():
        raise FileExistsError('Refusing to overwrite existing A1 evidence: ' + str(args.output_dir))

    source_bytes = args.cases.read_bytes()
    data = json.loads(source_bytes)
    validate_cases(data)
    profile = get_profile(PROFILE)
    source_manifest = ROOT / profile['local_path'] / 'source.json'
    if not source_manifest.is_file():
        raise FileNotFoundError('固定版本 Qwen3-1.7B 本地模型未找到：' + str(source_manifest))
    source_info = json.loads(source_manifest.read_text(encoding='utf-8'))
    if source_info.get('model_id') != profile['model_id'] or source_info.get('revision') != profile['revision']:
        raise ValueError('本地模型来源与冻结 profile 不匹配')

    role_prompt = conversation_messages(dict(character_id='companion', messages=[]))[0]['content']
    role_prompt_sha = sha256_bytes(role_prompt.encode('utf-8'))
    actual_settings = {**GENERATION_SETTINGS, 'max_new_tokens': args.max_new_tokens}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config = dict(
        created_at=datetime.now().astimezone().isoformat(),
        dataset_id=data['dataset_id'], dataset_sha256=sha256_bytes(source_bytes),
        model_id=profile['model_id'], model_revision=profile['revision'],
        local_model_source_sha256=sha256_bytes(source_manifest.read_bytes()),
        variants=['general_prompt', 'xiaoman_role_card'],
        generic_system_prompt=GENERIC_PROMPT,
        xiaoman_system_prompt=role_prompt, xiaoman_system_prompt_sha256=role_prompt_sha,
        generation_settings=actual_settings, seed=SEED,
        enable_thinking=False, input_token_budget=2048,
        raw_model_path=True,
        note='Development prompt comparison only. Calls QwenChat directly and bypasses deterministic app identity policies. No training or final independent evaluation.'
    )
    (args.output_dir / 'protocol.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    (args.output_dir / 'cases.snapshot.json').write_bytes(source_bytes)

    engine = QwenChat(model_profile=PROFILE, trim_history=False)
    results = []
    variants = ['general_prompt', 'xiaoman_role_card']
    for case in data['cases']:
        for variant in variants:
            set_seed(SEED)
            history = []
            turns = []
            for user_text in case['user_turns']:
                messages = render_messages(variant, history, user_text)
                answer, metrics = engine.reply(messages, model_mode='base', max_new_tokens=args.max_new_tokens)
                turns.append(dict(user=user_text, answer=answer, metrics=metrics))
                history.extend([dict(role='user', content=user_text), dict(role='assistant', content=answer)])
            results.append(dict(case_id=case['id'], focus=case['focus'], variant=variant, turns=turns))
            (args.output_dir / 'raw-results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Recorded {case['id']} / {variant}", flush=True)

    answer_map = {}
    for case in data['cases']:
        bit = hashlib.sha256((case['id'] + data['dataset_id']).encode('utf-8')).digest()[0] & 1
        answer_map[case['id']] = variants[bit:] + variants[:bit]
    (args.output_dir / 'answer-map.json').write_text(json.dumps(answer_map, ensure_ascii=False, indent=2), encoding='utf-8')
    render_score_sheet(data['cases'], results, answer_map, args.output_dir / 'score-sheet.md')

    summaries = {}
    for variant in variants:
        rows = [row for row in results if row['variant'] == variant]
        metrics = [turn['metrics'] for row in rows for turn in row['turns']]
        summaries[variant] = dict(
            scenarios=len(rows), replies=len(metrics),
            median_output_tokens=statistics.median(m['output_tokens'] for m in metrics),
            max_token_limit_hits=sum(m['output_tokens'] >= args.max_new_tokens and not m['ended_with_eos'] for m in metrics),
            eos_ended=sum(bool(m['ended_with_eos']) for m in metrics),
            median_generation_seconds=statistics.median(m['seconds'] for m in metrics),
        )
    (args.output_dir / 'summary.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
