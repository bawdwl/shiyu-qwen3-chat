"""Freeze local-only, multi-turn NaturalConv prompts for a blind user review."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from training_data import audit_naturalconv, read_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selection', required=True, help='JSON list of NaturalConv evaluation record IDs')
    parser.add_argument('--data-dir', default='data/naturalconv-v1')
    parser.add_argument('--output-dir', default='outputs/naturalconv-real-chat-review-v1-frozen')
    args = parser.parse_args()

    root = Path(args.data_dir)
    audit = audit_naturalconv(root)
    manifest = json.loads((root / 'freeze-manifest.json').read_text(encoding='utf-8'))
    selection_bytes = Path(args.selection).read_bytes()
    selected = json.loads(selection_bytes)
    if not isinstance(selected, list) or len(selected) < 4:
        raise ValueError('Selection must contain at least four record IDs')
    ids = [row['id'] if isinstance(row, dict) else row for row in selected]
    if any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError('Selection IDs must be unique nonempty strings')

    records = read_records(root / 'eval.jsonl')
    by_id = {row['id']: row for row in records}
    if len(by_id) != len(records):
        raise ValueError('Duplicate NaturalConv evaluation IDs')
    cases, used_documents = [], set()
    for record_id in ids:
        if record_id not in by_id:
            raise ValueError('Selected ID is absent from the frozen evaluation split: ' + record_id)
        row = by_id[record_id]
        document_id = str(row['source_document_id'])
        if document_id in used_documents:
            raise ValueError('Choose at most one conversation per source document')
        used_documents.add(document_id)
        messages = row['messages']
        if not messages or messages[-1]['role'] != 'user' or messages[-1]['content'] != row['prompts'][-1]:
            raise ValueError('Selected conversation must end at its current user turn')
        # Store only the conversation prefix and the current user turn. The other
        # human speaker's reference response is deliberately excluded.
        cases.append(dict(id=record_id, topic=row['topic'],
                          source_document_id=document_id, messages=messages))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    payload = json.dumps(cases, ensure_ascii=False, indent=2).encode('utf-8')
    (output / 'cases.json').write_bytes(payload)
    frozen = dict(
        version='naturalconv-blind-review-v1',
        created_at=datetime.now().astimezone().isoformat(),
        source='NaturalConv local test split; scenario-elicited two-person dialogue',
        source_license=manifest['license'], use_scope=manifest['use_scope'],
        source_eval_sha256=audit['sha256']['eval'],
        case_count=len(cases), cases_sha256=hashlib.sha256(payload).hexdigest(),
        selection_sha256=hashlib.sha256(selection_bytes).hexdigest(),
        selected_ids=ids, selected_source_documents=sorted(used_documents),
        selection_method='Curated from user-visible multi-turn context before generating either model answer; one conversation per source document',
        excluded_reference_responses=True,
        limitations=[
            'Scenario-elicited dialogue is not a random sample of real app usage.',
            'This is a small personal preference review, not a general benchmark or factuality score.',
            'Source data and all derived review files are for local non-commercial research/testing only; do not distribute.',
            'The source assistant turns are human dialogue context, not role/persona instructions for the evaluated AI.',
        ],
    )
    (output / 'freeze-manifest.json').write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='frozen', case_count=len(cases), cases_sha256=frozen['cases_sha256'],
                          license=frozen['source_license'], use_scope=frozen['use_scope']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
