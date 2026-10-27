"""Apply a single controlled shared-role dialogue revision to an immutable data build."""
import argparse
import json
import shutil
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer, __version__ as transformers_version

from prepare_shared_role_data import write_json
from runtime_models import get_profile
from shared_role_data import (DATA_VERSION, audit_shared_dataset, encode_role_sample,
                              content_hash, expand_dialogue, load_registry, render_role_prompt,
                              validate_dialogue_splits)
from training_data import file_hash


def _target_profile(rows):
    values = [row['target_tokens_including_eos'] for row in rows]
    if not values:
        raise ValueError('token 审计为空')
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values), at_most_12=sum(value <= 12 for value in values),
                over_32=sum(value > 32 for value in values),
                at_least_64=sum(value >= 64 for value in values))


def revise(base_dir, revision_path, output_dir):
    base = Path(base_dir).resolve()
    revision_file = Path(revision_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('拒绝覆盖已有修订包：' + str(output))
    if not base.is_dir() or not revision_file.is_file() or not output.parent.is_dir():
        raise FileNotFoundError('基础数据包、修订清单或输出父目录不存在')

    base_manifest = json.loads((base / 'freeze-manifest.json').read_text(encoding='utf-8'))
    revision = json.loads(revision_file.read_text(encoding='utf-8'))
    if base_manifest.get('version') != DATA_VERSION:
        raise ValueError('基础数据版本不匹配')
    if file_hash(base / 'dialogues.json') != revision.get('base_dialogues_sha256'):
        raise ValueError('基础对话与修订清单锁定指纹不匹配')
    if revision.get('version') != 'shared-role-context-revision-v1' or len(revision.get('entries', [])) != 20:
        raise ValueError('修订清单版本或修订数量不符合本轮方案')

    registry = load_registry(base / 'role-cards.json')
    dialogues = json.loads((base / 'dialogues.json').read_text(encoding='utf-8'))
    index = {item['id']: item for item in dialogues}
    if len(index) != len(dialogues):
        raise ValueError('基础数据包含重复对话 ID')
    entries = revision['entries']
    entry_ids = [entry.get('dialogue_id') for entry in entries]
    if len(entry_ids) != len(set(entry_ids)):
        raise ValueError('修订清单含重复对话 ID')
    if not set(entry_ids) <= set(index):
        raise ValueError('修订清单引用了基础包中不存在的对话')

    change_log = []
    for entry in entries:
        row = index[entry['dialogue_id']]
        if row['split'] != 'train' or row['category'] != 'context':
            raise ValueError('本轮只允许改训练集上下文类别')
        if row['role_id'] not in {role['id'] for role in registry['roles']
                                  if role['partition'] == 'train'}:
            raise ValueError('修订角色不属于训练分区')
        if len(row['messages']) != 10 or len(entry.get('messages', [])) != 10:
            raise ValueError('每条修订必须保持原有五轮对话长度')
        if entry.get('scenario_group') == row['scenario_group']:
            raise ValueError('修订对话必须记录新的具体场景组')
        before_hash = content_hash(row['messages'])
        for position, message in enumerate(entry['messages']):
            if message.get('role') != ('user' if position % 2 == 0 else 'assistant'):
                raise ValueError('修订后的用户/助手轮次顺序错误：' + entry['dialogue_id'])
            if set(message) != {'role', 'content'}:
                raise ValueError('修订消息仅允许 role/content 字段')
        old_group = row['scenario_group']
        row['messages'] = entry['messages']
        row['scenario_group'] = entry['scenario_group']
        row['source_origin'] = dict(
            **row.get('source_origin', {}),
            revision_id=revision['revision_id'],
            revision_file=str(revision_file),
            original_content_sha256=before_hash,
            authoring=revision['source_note'])
        change_log.append(dict(
            dialogue_id=row['id'], role_id=row['role_id'], old_scenario_group=old_group,
            new_scenario_group=row['scenario_group'], rounds=len(row['messages']) // 2,
            old_content_sha256=before_hash,
            new_content_sha256=content_hash(row['messages'])))

    dialogues.sort(key=lambda item: item['id'])
    expanded, counts = validate_dialogue_splits(dialogues, registry)
    if ({key: sum(values.values()) for key, values in counts.items()}
            != {'train': 400, 'validation': 50}):
        raise ValueError('修订改变了400/50对话拆分数量')
    profile = get_profile('qwen3-1.7b')
    if (base_manifest.get('model', {}).get('model_id') != profile['model_id']
            or base_manifest.get('model', {}).get('revision') != profile['revision']):
        raise ValueError('基础模型 revision 与本项目锁定版本不符')
    tokenizer = AutoTokenizer.from_pretrained(
        profile['local_path'], revision=profile['revision'], local_files_only=True)
    token_rows = []
    for split, rows in expanded.items():
        for row in rows:
            encoded = encode_role_sample(row, tokenizer, registry, max_length=1024)
            target = tokenizer.encode(row['messages'][-1]['content'], add_special_tokens=False)
            target.append(tokenizer.eos_token_id)
            if [token for token in encoded['labels'] if token != -100] != target:
                raise ValueError('assistant target / EOS mismatch: ' + row['id'])
            if encoded['labels'][:-len(target)] != [-100] * (len(encoded['labels']) - len(target)):
                raise ValueError('user/history labels are not masked: ' + row['id'])
            full = tokenizer.apply_chat_template(
                [dict(role='system', content=render_role_prompt(registry, row['role_id']))]
                + row['messages'], tokenize=True, return_dict=False,
                add_generation_prompt=False, enable_thinking=False)
            end = max(i for i, value in enumerate(full)
                      if value == tokenizer.eos_token_id)
            if full[:end + 1] != encoded['input_ids']:
                raise ValueError('chat-template mismatch: ' + row['id'])
            token_rows.append(dict(id=row['id'], role_id=row['role_id'], split=split,
                                   input_tokens=len(encoded['input_ids']),
                                   target_tokens_including_eos=len(target)))
    if len(token_rows) != 1058 or any(row['input_tokens'] > 1024 for row in token_rows):
        raise ValueError('目标数量或1024 token预算检查失败')

    output.mkdir(parents=True, exist_ok=False)
    for name in ('role-cards.json', 'rendered-prompts.json'):
        shutil.copyfile(base / name, output / name)
    write_json(output / 'dialogues.json', dialogues)
    for split, rows in expanded.items():
        (output / f'{split}.jsonl').write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
            encoding='utf-8', newline='\n')
    write_json(output / 'token-audit.json', token_rows)

    old_token_rows = json.loads((base / 'token-audit.json').read_text(encoding='utf-8'))
    by_category = {}
    for label, rows in (('before', old_token_rows), ('after', token_rows)):
        records = {record['id']: record for split in ('train', 'validation')
                   for record in expanded[split]}
        grouped = {}
        for row in rows:
            source_id = row['id'].rsplit('-t', 1)[0]
            category = index[source_id]['category']
            if row['split'] == 'train':
                grouped.setdefault(category, []).append(row['target_tokens_including_eos'])
        by_category[label] = {name: _target_profile([dict(target_tokens_including_eos=value)
                                                     for value in values])
                              for name, values in sorted(grouped.items())}
    revision_report = dict(
        version=revision['revision_id'], created_at=datetime.now().astimezone().isoformat(),
        base_dialogues_sha256=file_hash(base / 'dialogues.json'),
        revision_file_sha256=file_hash(revision_file), changed_dialogues=change_log,
        changed_training_dialogues=len(change_log), changed_assistant_targets=sum(
            item['rounds'] for item in change_log),
        source_dialogues={key: sum(value.values()) for key, value in counts.items()},
        assistant_targets={key: len(value) for key, value in expanded.items()},
        target_length_by_category=by_category,
        validation_unchanged=(file_hash(base / 'validation.jsonl')
                              == file_hash(output / 'validation.jsonl')),
        validation_jsonl_sha256_before=file_hash(base / 'validation.jsonl'),
        validation_jsonl_sha256_after=file_hash(output / 'validation.jsonl'),
        role_cards_sha256_before=file_hash(base / 'role-cards.json'),
        role_cards_sha256_after=file_hash(output / 'role-cards.json'),
        method='Revised 20 long training-only context conversations (one per training role); kept 400/50 splits, role cards, validation bytes, template, base revision, and model settings fixed.',
        limitation='All revised examples are AI-authored and have not received independent human review. This is a data-quality pilot, not proof of representative natural conversation.')
    write_json(output / 'revision-report.json', revision_report)

    manifest = json.loads((base / 'freeze-manifest.json').read_text(encoding='utf-8'))
    manifest.update(
        created_at=datetime.now().astimezone().isoformat(),
        stage='v14_data_revision_pending_final_freeze_and_gpu_preflight',
        training_ready=False, final_evaluation_frozen=False,
        reserved_final_role_ids=[], evaluation_files={}, final_evaluation={}, training_preflight={},
        training_config={},
        provenance=dict(**manifest.get('provenance', {}), revision=dict(
            id=revision['revision_id'], file=str(revision_file),
            sha256=file_hash(revision_file), changed_dialogues=len(change_log),
            changed_assistant_targets=revision_report['changed_assistant_targets'],
            authoring=revision['source_note'])),
        files={name: dict(sha256=file_hash(output / name)) for name in (
            'role-cards.json', 'dialogues.json', 'train.jsonl', 'validation.jsonl')},
        code_sha256={name: file_hash(name) for name in (
            'shared_role_data.py', 'training_data.py', 'prepare_shared_role_data.py',
            'prepare_shared_role_revision.py')},
        remaining=['Freeze the unchanged final evaluation against this revised training set',
                   'Run full tokenizer/label and longest-sample GPU preflight',
                   'Only then train the single data-revision candidate'])
    manifest['target_counts'] = {key: len(value) for key, value in expanded.items()}
    manifest['source_dialogues'] = {key: sum(value.values()) for key, value in counts.items()}
    manifest['dialogues_by_role'] = {key: dict(value) for key, value in counts.items()}
    manifest['files']['token-audit.json'] = dict(sha256=file_hash(output / 'token-audit.json'))
    manifest['files']['revision-report.json'] = dict(sha256=file_hash(output / 'revision-report.json'))
    manifest['near_duplicates'] = dict(
        method='Cross-split surface trigram audit is rerun during final evaluation freeze.',
        status='pending_final_evaluation_freeze')
    write_json(output / 'freeze-manifest.json', manifest)

    qc = json.loads((base / 'preflight.json').read_text(encoding='utf-8'))
    qc.update(
        status='passed_structural_v14_revision_pending_freeze', training_ready=False,
        final_evaluation_frozen=False,
        validation_unchanged=(revision_report['validation_jsonl_sha256_before']
                             == revision_report['validation_jsonl_sha256_after']),
        tokenizer_cases_checked=len(token_rows), full_template_matches=len(token_rows),
        all_context_labels_masked=True, truncated_samples=0,
        input_tokens=dict(minimum=min(row['input_tokens'] for row in token_rows),
                          median=statistics.median(row['input_tokens'] for row in token_rows),
                          maximum=max(row['input_tokens'] for row in token_rows)),
        target_tokens_including_eos=_target_profile(token_rows),
        targets_at_least_64_tokens=sum(row['target_tokens_including_eos'] >= 64
                                       for row in token_rows),
        train_targets_at_least_64_tokens=sum(
            row['split'] == 'train' and row['target_tokens_including_eos'] >= 64
            for row in token_rows),
        revision_report='revision-report.json',
        limitation='Structural/tokenizer checks do not establish human naturalness or response-quality improvement.')
    write_json(output / 'preflight.json', qc)
    data_audit = audit_shared_dataset(output, require_training_ready=False)
    if not revision_report['validation_unchanged'] or data_audit['counts'] != {
            'train': 932, 'validation': 126}:
        raise ValueError('修订后拆分、目标数量或验证集指纹未保持')
    return dict(output_dir=str(output), training_ready=False,
                changed_dialogues=len(change_log), changed_assistant_targets=revision_report['changed_assistant_targets'],
                tokenizer_targets=len(token_rows), maximum_input_tokens=max(row['input_tokens'] for row in token_rows),
                validation_unchanged=revision_report['validation_unchanged'],
                role_cards_unchanged=file_hash(output / 'role-cards.json') == file_hash(base / 'role-cards.json'),
                target_length_by_category=by_category)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='outputs/shared-role-build-v13')
    parser.add_argument('--revision', default='data/shared-role-chat-v1/revisions/v14-context-naturalization.json')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(revise(args.base_dir, args.revision, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
