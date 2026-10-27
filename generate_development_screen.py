"""Generate paired base/LoRA outputs for the non-final development screen."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from transformers import set_seed

from inference import GENERATION_SETTINGS, QwenChat
from runtime_models import get_profile
from shared_role_data import load_registry, render_role_prompt
from training_data import file_hash


def generate(run_dir, cases_path, registry_path, output_path):
    run = Path(run_dir)
    cases_file = Path(cases_path)
    registry_file = Path(registry_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError('拒绝覆盖已有开发筛查输出：' + str(output))
    packet = json.loads(cases_file.read_text(encoding='utf-8'))
    registry = load_registry(registry_file)
    cases = packet.get('cases')
    if packet.get('version') != 'shared-role-development-screen-v1' or len(cases) != 12:
        raise ValueError('开发筛查题目版本或数量不匹配')
    if any(case['role_id'] not in {role['id'] for role in registry['roles']}
           for case in cases):
        raise ValueError('开发筛查引用了未知角色卡')
    profile = get_profile('qwen3-1.7b')
    chat = QwenChat(adapter_run=run, model_profile='qwen3-1.7b')
    results = []
    for index, case in enumerate(cases):
        prompt = render_role_prompt(registry, case['role_id'])
        pair = dict(case_id=case['case_id'], role_id=case['role_id'],
                    category='development_only', review_points=case['review_points'],
                    outputs={})
        for mode in ('base', 'lora'):
            set_seed(42)
            history = [dict(role='system', content=prompt)]
            turns = []
            for user_text in case['turns']:
                history.append(dict(role='user', content=user_text))
                answer, metrics = chat.reply(history, model_mode=mode, max_new_tokens=256)
                history.append(dict(role='assistant', content=answer))
                turns.append(dict(user=user_text, answer=answer, metrics=metrics))
            pair['outputs'][mode] = turns
        for mode in ('base', 'lora'):
            answers = [turn['answer'] for turn in pair['outputs'][mode]]
            pair.setdefault('screen_signals', {})[mode] = dict(
                empty_answers=sum(not answer.strip() for answer in answers),
                identical_repeated_answers=len(answers) - len(set(answers)),
                output_cap_hits=sum(
                    turn['metrics']['output_tokens'] >= 254
                    for turn in pair['outputs'][mode]))
        results.append(pair)
    result = dict(
        version='shared-role-development-screen-output-v1',
        created_at=datetime.now().astimezone().isoformat(),
        run_dir=str(run),
        base_model=profile['model_id'],
        revision=profile['revision'],
        settings=dict(seed=42, thinking=False, **GENERATION_SETTINGS),
        cases_file_sha256=file_hash(cases_file),
        role_cards_sha256=file_hash(registry_file),
        adapter_sha256=file_hash(run / 'adapter' / 'adapter_model.safetensors'),
        source='Development-only AI-authored probes; not the final frozen evaluation.',
        signals_note='Counts only empty answers, exact repeated answers, and near-limit outputs. They are diagnostic flags, not quality scores.',
        cases=results)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return dict(output=str(output), cases=len(results), adapter_sha256=result['adapter_sha256'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--cases', required=True)
    parser.add_argument('--registry', default='data/shared-role-chat-v1/role-cards.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.run_dir, args.cases, args.registry, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
