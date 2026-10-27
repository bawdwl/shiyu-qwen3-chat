"""Development-only ablation of decorative profile injection; no weight updates."""
import hashlib
import copy
import json
from datetime import datetime
from pathlib import Path

from transformers import set_seed

from characters import CHARACTERS, conversation_messages
from inference import GENERATION_SETTINGS, QwenChat


def main():
    baseline_root = Path('outputs/lccc-role-v2-development')
    baseline_config = json.loads((baseline_root / 'run-config.json').read_text(encoding='utf-8'))
    baseline_records = json.loads((baseline_root / 'results.json').read_text(encoding='utf-8'))
    source = (baseline_root / 'cases.snapshot.json').read_bytes()
    if hashlib.sha256(source).hexdigest() != baseline_config['development_cases_sha256']:
        raise ValueError('Baseline case snapshot does not match its recorded fingerprint')
    if baseline_config['generation_settings'] != GENERATION_SETTINGS:
        raise ValueError('Generation settings changed')
    historical = copy.deepcopy(CHARACTERS)
    historical['companion']['profile_policy'] = 'always'
    current = conversation_messages(dict(character_id='companion', messages=[]), characters=historical)[0]['content']
    if current != baseline_config['system_prompt']:
        raise ValueError('Baseline system prompt changed')
    cases = json.loads(source)['cases']
    candidate = CHARACTERS['companion']['prompt']
    root = Path('outputs/character-profile-ablation-v2')
    root.mkdir(parents=True, exist_ok=False)
    (root / 'cases.snapshot.json').write_bytes(source)
    config = dict(timestamp=datetime.now().astimezone().isoformat(), model=baseline_config['model_id'],
                  revision=baseline_config['revision'], seeds=baseline_config['seeds'],
                  generation_settings=GENERATION_SETTINGS, baseline_system=current, candidate_system=candidate,
                  changed_factor='Remove decorative scene/profile paragraph only; retain original identity and behavioral instructions',
                  baseline_config_sha256=hashlib.sha256((baseline_root / 'run-config.json').read_bytes()).hexdigest(),
                  development_cases_sha256=hashlib.sha256(source).hexdigest(),
                  note='Exposed original developer scenarios; not independent or blind quality evidence. No model weights trained.')
    (root / 'run-config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    engine = QwenChat(adapter_run='outputs/lora-lccc-base-v1-pilot', model_profile='qwen3-1.7b', trim_history=True)
    records = []
    for case in cases:
        for seed in baseline_config['seeds']:
            set_seed(seed)
            answer, metrics = engine.reply([dict(role='system', content=candidate)] + case['messages'], model_mode='base')
            records.append(dict(case_id=case['id'], seed=seed, answer=answer, metrics=metrics))
            (root / 'results.json').write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Recorded no-profile {case['id']} seed={seed}", flush=True)
    original = [row for row in baseline_records if row['variant'] == 'base']
    terms = ('咖啡', '拿铁', '窗边', '阳光')
    summary = dict(baseline_replies=len(original), candidate_replies=len(records),
                   baseline_setting_term_replies=sum(any(term in row['answer'] for term in terms) for row in original),
                   candidate_setting_term_replies=sum(any(term in row['answer'] for term in terms) for row in records),
                   note='Presence of setting-related terms is a development diagnostic only, not a quality score.')
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 去除场景注入：开发对照', '', '同一基座、固定上下文和随机种子；仅移除背景资料段。没有人工评分。', '']
    for case in cases:
        lines += ['## ' + case['id'], '']
        for message in case['messages']:
            lines += [f"- {message['role']}：{message['content']}"]
        for seed in baseline_config['seeds']:
            old = next(row for row in original if row['case_id'] == case['id'] and row['seed'] == seed)
            new = next(row for row in records if row['case_id'] == case['id'] and row['seed'] == seed)
            lines += ['', f"**原版 / seed={seed}：** {old['answer']}", '', f"**无背景段 / seed={seed}：** {new['answer']}"]
    (root / 'answers.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
