"""Bounded official-source sampling; candidate inspection only, never starts training."""
import argparse
import hashlib
import json
import re
import random
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


def fetch_json(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'shiyu-data-audit/1.0'})
    with urllib.request.urlopen(request, timeout=25) as response:
        raw = response.read(20 * 1024 * 1024 + 1)
    if len(raw) > 20 * 1024 * 1024:
        raise ValueError('Response exceeds bounded audit size')
    return json.loads(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--windows', type=int, default=0,
                        help='0 repeats the original three windows; 1–60 samples seeded windows across the population')
    parser.add_argument('--population-size', type=int, default=837989,
                        help='Declared source row count, recorded for reproducibility; verify against the source card')
    parser.add_argument('--seed', type=int, default=20261026)
    args = parser.parse_args()
    if not 0 <= args.windows <= 60 or args.population_size < 100:
        parser.error('windows must be 0–60 and population-size at least 100')
    offsets = ([0, 1000, 5000] if args.windows == 0 else
               sorted(random.Random(args.seed).sample(range(0, args.population_size - 99, 100), args.windows)))
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    dataset = 'allenai/WildChat-1M'
    metadata = fetch_json('https://huggingface.co/api/datasets/' + dataset)
    revision = metadata['sha']
    report = dict(dataset=dataset, revision=revision,
                  recorded_at=datetime.now(timezone.utc).isoformat(),
                  license=metadata.get('cardData', {}).get('license'),
                  sample_method=('Three fixed viewer windows; not representative.' if args.windows == 0 else
                                 'Seeded windows across declared population; clustered window sampling, not independent row sampling.'),
                  population_size_declared=args.population_size, seed=args.seed, offsets=offsets,
                  purpose='Candidate inspection only. No training authorization or quality approval implied.',
                  pages=[], sampled_rows=0, chinese_rows=0, retained_rows=0, rejected={},
                  pii_metadata_policy='Do not save IP hashes, headers, timestamps, country/state, or demographics.')
    candidates = []
    seen = set()
    def load_window(offset):
        params = dict(dataset=dataset, config='default', split='train', offset=offset, length=100)
        url = 'https://datasets-server.huggingface.co/rows?' + urllib.parse.urlencode(params)
        try:
            page = fetch_json(url)
        except (OSError, ValueError, urllib.error.HTTPError) as error:
            return offset, None, str(error)[:180]
        return offset, page, None

    with ThreadPoolExecutor(max_workers=4) as pool:
        pages = list(pool.map(load_window, offsets))
    for offset, page, error in pages:
        if page is None:
            report['pages'].append(dict(offset=offset, status='failed', error=error))
            continue
        rows = page.get('rows', [])
        report['pages'].append(dict(offset=offset, status='ok', rows=len(rows)))
        report['sampled_rows'] += len(rows)
        for wrapper in rows:
            row = wrapper['row']
            conversation = row.get('conversation', [])
            if row.get('language', '').lower() not in ('chinese', 'zh'):
                continue
            report['chinese_rows'] += 1
            messages = [dict(role=m.get('role'), content=m.get('content', '').strip()) for m in conversation]
            joined = '\n'.join(m['content'] for m in messages)
            reason = None
            if row.get('toxic') or row.get('redacted') or any(m.get('toxic') or m.get('redacted') for m in conversation):
                reason = 'source_flagged_or_redacted'
            elif not messages or len(messages) < 2 or messages[0]['role'] != 'user' or messages[-1]['role'] != 'assistant':
                reason = 'incomplete_conversation'
            elif any(m['role'] != ('user' if i % 2 == 0 else 'assistant') or not m['content'] for i, m in enumerate(messages)):
                reason = 'empty_or_invalid_alternation'
            elif re.search(r'(?i)https?://|[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?<!\d)1[3-9]\d{9}(?!\d)', joined):
                reason = 'contact_or_link_requires_review'
            elif len(joined) > 12000:
                reason = 'too_long_for_initial_audit'
            digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if digest in seen:
                reason = 'duplicate'
            seen.add(digest)
            if reason:
                report['rejected'][reason] = report['rejected'].get(reason, 0) + 1
                continue
            candidates.append(dict(source_dataset=dataset, source_revision=revision,
                                   source_row_index=wrapper.get('row_idx'),
                                   source_conversation_hash=row.get('conversation_hash'),
                                   content_sha256=digest, messages=messages,
                                   provenance='Real user inputs; ChatGPT assistant outputs. Not human-human dialogue.',
                                   review_status='unreviewed_not_training_ready'))
    # Detect upstream revision changes during the audit rather than claiming pinned row content.
    after = fetch_json('https://huggingface.co/api/datasets/' + dataset)['sha']
    report['revision_after_sampling'] = after
    report['revision_stable_during_sampling'] = after == revision
    report['retained_rows'] = len(candidates)
    report['assistant_character_lengths'] = [len(m['content']) for c in candidates for m in c['messages'] if m['role'] == 'assistant']
    text = ''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in candidates)
    (root / 'candidates.jsonl').write_text(text, encoding='utf-8')
    report['candidates_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    (root / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('dataset', 'revision', 'sampled_rows', 'chinese_rows', 'retained_rows', 'pages')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
