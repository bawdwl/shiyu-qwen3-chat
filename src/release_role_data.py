"""Role-conditioned release experiment; independent of historical datasets.

Manifests are created only after content review and split freezing. The pilot
is deliberately not accepted by this loader. No model or torch imports.
"""
import json
from pathlib import Path

from training_data import encode_sample, file_hash, read_records

DATA_VERSION = 'rolechat-release-v1'
MARKERS = ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>')


def validate_row(row):
    for key in ('id', 'source_dialogue_id', 'scene_group', 'role_id',
                'system_prompt', 'source_kind', 'review_status'):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Missing nonempty field: ' + key)
    if row.get('training_ready') is not True:
        raise ValueError('Unreviewed pilot cannot enter formal training')
    messages = row.get('messages')
    if not isinstance(messages, list) or len(messages) < 2 or len(messages) % 2:
        raise ValueError('Expected complete user/assistant pairs')
    texts = [row['system_prompt']]
    for i, message in enumerate(messages):
        expected = 'user' if i % 2 == 0 else 'assistant'
        if message.get('role') != expected or not isinstance(message.get('content'), str) or not message['content'].strip():
            raise ValueError('Invalid alternating dialogue')
        texts.append(message['content'])
    if any(marker in text for text in texts for marker in MARKERS):
        raise ValueError('Control token in data')
    if row['source_kind'] not in ('original_ai_synthetic', 'public_crowdsourced'):
        raise ValueError('Unrecognized provenance')
    if row['source_kind'] == 'public_crowdsourced':
        for key in ('source', 'source_revision', 'source_license', 'source_message_ids', 'source_tree_id'):
            if not row.get(key):
                raise ValueError('Missing public source attribution: ' + key)


def audit_release_dataset(root):
    root = Path(root)
    manifest = json.loads((root / 'freeze-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != DATA_VERSION or manifest.get('status') != 'frozen_for_training':
        raise ValueError('Dataset is not frozen for this experiment')
    if not manifest.get('evaluation_sha256') or not manifest.get('content_review_record'):
        raise ValueError('Missing evaluation freeze or review record')
    for filename, hash_key in (('final-evaluation.json', 'evaluation_sha256'),
                               ('development.json', 'development_sha256'),
                               ('evaluation-protocol.json', 'protocol_sha256'),
                               ('content-review.json', 'content_review_sha256')):
        if file_hash(root / filename) != manifest.get(hash_key):
            raise ValueError('Supporting freeze hash mismatch: ' + filename)
    final = json.loads((root / 'final-evaluation.json').read_text(encoding='utf-8'))
    if set(final['roles']) != set(manifest.get('evaluation_role_ids', [])):
        raise ValueError('Final role registry mismatch')
    sets, counts, hashes = {}, {}, {}
    seen_ids = set()
    for split in ('train', 'validation'):
        path = root / (split + '.jsonl')
        hashes[split] = file_hash(path)
        rows = read_records(path)
        expected = manifest['outputs'][split]
        if hashes[split] != expected['sha256'] or len(rows) != expected['count'] or not rows:
            raise ValueError('Split hash/count mismatch: ' + split)
        groups = {key: set() for key in ('source_dialogue_id', 'scene_group', 'role_id', 'source_tree_id', 'source_message_ids')}
        for row in rows:
            validate_row(row)
            if row['id'] in seen_ids:
                raise ValueError('Duplicate sample ID')
            seen_ids.add(row['id'])
            for key in groups:
                value = row.get(key)
                if value:
                    groups[key].update(value if isinstance(value, list) else [value])
            if row['role_id'] in manifest.get('evaluation_role_ids', []):
                raise ValueError('Final evaluation role leaked into data')
        sets[split] = groups
        counts[split] = len(rows)
    for key in sets['train']:
        if sets['train'][key] & sets['validation'][key]:
            raise ValueError('Cross-split overlap: ' + key)
    return dict(version=DATA_VERSION, counts=counts, sha256=hashes,
                manifest_sha256=file_hash(root / 'freeze-manifest.json'),
                evaluation_sha256=manifest['evaluation_sha256'],
                conditioning='per_row_system_prompt',
                limitation='Exact group isolation does not prove semantic independence or answer correctness.')


def encode_release_sample(row, tokenizer, max_length):
    validate_row(row)
    return encode_sample(row, tokenizer, max_length=max_length,
                         trim_history=False, system_prompt=row['system_prompt'])


def training_common_prefix(rows):
    """Recover actual frozen conditioning, not a newly edited source prompt."""
    marker = '\n【当前角色】\n'
    originals = [row for row in rows if row.get('source_kind') == 'original_ai_synthetic']
    if not originals:
        raise ValueError('No original role-conditioned rows')
    prefixes = set()
    for row in originals:
        validate_row(row)
        prompt = row['system_prompt']
        if prompt.count(marker) != 1:
            raise ValueError('Expected exactly one role-card delimiter')
        prefixes.add(prompt.split(marker)[0])
    if len(prefixes) != 1:
        raise ValueError('Inconsistent common conditioning in frozen data')
    return prefixes.pop() + marker
