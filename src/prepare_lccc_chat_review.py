"""Freeze manually chosen, self-contained real user utterances for App review.

Selection is by prompt content only, before generating any candidate answers.
Local research material only; not part of the release data allowlist.
"""
import argparse
import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path

from inspect_lccc import parse_dialog
from prepare_lccc import acceptable, dialog_hash
from training_data import audit_lccc, read_records


def choose_utterance(dialog, turn, known_hashes, known_text):
    if type(turn) is not int or turn < 0 or turn % 2 or turn + 1 >= len(dialog):
        raise ValueError('Expected a user-side turn with an actual following human reply')
    digest = dialog_hash(dialog)
    text = dialog[turn]
    if digest in known_hashes or text in known_text:
        raise ValueError('Exact source dialogue or chosen utterance overlaps train/validation')
    return text, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', required=True, help='JSON list: source_dialog_id, turn_index')
    parser.add_argument('--data-dir', default='data/lccc-v1-pilot')
    parser.add_argument('--output-dir', default='outputs/lccc-real-chat-review-v2-frozen')
    args = parser.parse_args()
    audit = audit_lccc(args.data_dir)
    source_manifest = json.loads((Path(args.data_dir) / 'freeze-manifest.json').read_text(encoding='utf-8'))
    selected = json.loads(Path(args.selection).read_text(encoding='utf-8'))
    if not selected or len({str(row['source_dialog_id']) for row in selected}) != len(selected):
        raise ValueError('Selection must have unique, nonempty source dialogues')
    wanted = {str(row['source_dialog_id']) for row in selected}
    old_used, old_runs = set(), []
    for path in sorted(Path('outputs').glob('lccc-base-blind-review-*/run-config.json')):
        config = json.loads(path.read_text(encoding='utf-8'))
        old_used.update(config['source_dialogues'])
        old_runs.append(str(path))
    if wanted & old_used:
        raise ValueError('Selected source dialogue was used by an earlier blind review')
    known_hashes, known_text = set(), set()
    for split in ('train', 'validation'):
        for row in read_records(Path(args.data_dir) / f'{split}.jsonl'):
            known_hashes.add(row['source_dialog_hash'])
            known_text.update(message['content'] for message in row['messages'])
    source = Path(source_manifest['source_dir']) / 'lccc_base_test.jsonl.gz'
    found = {}
    with gzip.open(source, 'rt', encoding='utf-8') as stream:
        for index, line in enumerate(stream):
            if str(index) in wanted:
                normalized, reason = acceptable(parse_dialog(line))
                if reason:
                    raise ValueError(f'Source {index} rejected: {reason}')
                found[str(index)] = normalized
    cases = []
    for row in selected:
        source_id = str(row['source_dialog_id'])
        dialog = found[source_id]
        turn = row['turn_index']
        text, digest = choose_utterance(dialog, turn, known_hashes, known_text)
        cases.append(dict(id=f'lccc_prompt_{int(source_id):07d}_t{turn:02d}',
                          source_dialog_id=source_id, source_dialog_hash=digest, source_turn_index=turn,
                          messages=[dict(role='user', content=text)]))
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    content = json.dumps(cases, ensure_ascii=False, indent=2).encode('utf-8')
    (root / 'cases.json').write_bytes(content)
    manifest = dict(timestamp=datetime.now().astimezone().isoformat(), source='LCCC-base author test split',
                    source_test_sha256=source_manifest['source_files']['test'],
                    training_data_sha256=audit['sha256'], case_count=len(cases),
                    cases_sha256=hashlib.sha256(content).hexdigest(),
                    selection_sha256=hashlib.sha256(Path(args.selection).read_bytes()).hexdigest(),
                    excluded_review_runs=old_runs, excluded_source_dialogue_count=len(old_used),
                    exact_dialogue_and_utterance_overlap_train_validation=0,
                    selection_method='Manually curated self-contained user-side utterances, prompt-only selection before answers',
                    task='Single-turn character chat; preceding human dialogue and reference answers omitted. Not a multi-turn memory test.',
                    limitations='One curated source; base-model pretraining contamination unknown. Local research only, do not distribute.')
    (root / 'freeze-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
