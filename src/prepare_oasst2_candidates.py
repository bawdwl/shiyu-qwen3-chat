"""公开中文对话候选：按对话树划分，不自动当作人工审核训练集。"""
import argparse
import collections
import gzip
import hashlib
import json
import re
from pathlib import Path

from fetch_oasst2_source import FILENAME, FILE_SHA256, REPO, REVISION


def normalize(text):
    return re.sub(r'\W+', '', text).lower()


def split_for_root(text):
    bucket = int(hashlib.sha256(normalize(text).encode('utf-8')).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 80 else ('validation' if bucket < 90 else 'test')


def label_value(row, key):
    value = row.get('labels', {}).get(key, {})
    return value.get('value') if isinstance(value, dict) else None


def rejection(row):
    if row.get('lang') != 'zh':
        return 'not_zh'
    if row.get('review_result') is not True or row.get('deleted') is not False:
        return 'not_accepted_or_deleted'
    if row.get('synthetic') is not False:
        return 'synthetic_or_unknown'
    text = row.get('text', '').strip()
    if not text or len(text) > 3000:
        return 'empty_or_too_many_chars'
    if re.search(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)', text):
        return 'contact_pattern'
    for key in ('spam', 'pii', 'not_appropriate', 'hate_speech', 'sexual_content', 'lang_mismatch'):
        score = label_value(row, key)
        if score is not None and score >= 0.25:
            return 'upstream_flag_' + key
    if row.get('role') == 'assistant':
        if row.get('rank') not in (None, 0):
            return 'nonpreferred_answer'
        quality = label_value(row, 'quality')
        if quality is None or quality < 0.5:
            return 'assistant_quality_below_0.5_or_missing'
        if (label_value(row, 'fails_task') or 0) >= 0.25:
            return 'upstream_fails_task'
    return None


def collect_candidates(rows):
    index = {row['message_id']: row for row in rows}
    if len(index) != len(rows):
        raise ValueError('Duplicate source message IDs')
    stats = collections.Counter()
    candidates = []
    seen = set()
    for leaf in rows:
        if leaf['role'] != 'assistant':
            continue
        chain, current, visited = [], leaf, set()
        reason = None
        while current is not None:
            mid = current['message_id']
            if mid in visited:
                reason = 'cycle'
                break
            visited.add(mid)
            reason = rejection(current)
            if reason:
                break
            chain.append(current)
            parent = current.get('parent_id')
            if parent is None:
                break
            current = index.get(parent)
            if current is None:
                reason = 'missing_or_non_zh_parent'
                break
        if reason:
            stats[reason] += 1
            continue
        chain.reverse()
        if len(chain) < 2 or len(chain) > 8 or len(chain) % 2 or any(
            row['role'] != ('prompter' if i % 2 == 0 else 'assistant')
            for i, row in enumerate(chain)
        ):
            stats['invalid_roles_or_turn_count'] += 1
            continue
        messages = [dict(role='user' if row['role'] == 'prompter' else 'assistant',
                         content=row['text'].strip()) for row in chain]
        signature = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
        if signature in seen:
            stats['duplicate_conversation'] += 1
            continue
        seen.add(signature)
        candidates.append(dict(id='oasst2_' + leaf['message_id'], source=REPO,
            source_revision=REVISION, source_license='apache-2.0',
            source_message_ids=[r['message_id'] for r in chain],
            source_tree_id=chain[0].get('message_tree_id') or chain[0]['message_id'],
            split_group_sha256=hashlib.sha256(normalize(messages[0]['content']).encode('utf-8')).hexdigest(),
            split=split_for_root(messages[0]['content']), messages=messages,
            final_upstream_quality=label_value(leaf, 'quality'),
            review_status='candidate-not-locally-reviewed'))
    return candidates, dict(stats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', default='outputs/dataset-sources/oasst2')
    parser.add_argument('--output-dir', default='outputs/data-candidates/oasst2-v1')
    args = parser.parse_args()
    source = Path(args.source_dir) / FILENAME
    if hashlib.sha256(source.read_bytes()).hexdigest() != FILE_SHA256:
        raise ValueError('Upstream data hash mismatch')
    protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in Path('data').rglob('*') if p.is_file()}
    rows, total = [], 0
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            total += 1
            if row['lang'] == 'zh':
                rows.append(row)
    candidates, rejected = collect_candidates(rows)
    # CPU-only length audit against the existing, cached baseline tokenizer.
    from transformers import AutoTokenizer
    from inference import MODEL_ID, MODEL_REVISION
    from training_data import encode_sample
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    selected = []
    for record in candidates:
        try:
            encoded = encode_sample(record, tokenizer, max_length=512)
        except ValueError:
            rejected['token_limit_512'] = rejected.get('token_limit_512', 0) + 1
            continue
        record['baseline_token_count'] = len(encoded['input_ids'])
        selected.append(record)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    counts, hashes, groups = {}, {}, {}
    for split in ('train', 'validation', 'test'):
        subset = [r for r in selected if r['split'] == split]
        payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in subset).encode('utf-8')
        (root / (split + '-candidates.jsonl')).write_bytes(payload)
        counts[split] = len(subset)
        hashes[split] = hashlib.sha256(payload).hexdigest()
        groups[split] = {r['split_group_sha256'] for r in subset}
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not groups[left] & groups[right]
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in protected.items())
    manifest = dict(repo=REPO, revision=REVISION, source_sha256=FILE_SHA256,
        source_rows=total, chinese_source_messages=len(rows),
        candidate_count_before_token_filter=len(candidates), split_counts=counts,
        multi_turn_candidates=sum(len(r['messages']) > 2 for r in selected),
        max_baseline_tokens=max((r['baseline_token_count'] for r in selected), default=0),
        tokenizer_model=MODEL_ID, tokenizer_revision=MODEL_REVISION,
        rejection_counts=rejected, split_sha256=hashes,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protected_existing_data_sha256=protected,
        policy='Chinese, accepted, nondeleted, declared nonsynthetic, preferred rank 0 or unranked; assistant quality>=0.5; flagged/contact patterns excluded; max8 messages; <=512 baseline tokens. Group by normalized root question, not individual turns.',
        limitations='Candidates only, not expert-verified or privacy-certified. No semantic dedup guarantee. Original ready-message export has been regrouped; these are not official source splits. Retokenize if base model changes. Existing data/ files unchanged.')
    (root / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'protected_existing_data_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
