"""Audit the new original pilot without loading model weights or starting training."""
import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_data import encode_sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/rolechat-release-v1/original-pilot.json')
    parser.add_argument('--evaluation', default='data/rolechat-release-v1/evaluation-draft.json')
    parser.add_argument('--model-path', default='outputs/models/qwen3-1.7b')
    parser.add_argument('--max-length', type=int, default=1024)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    source = Path(args.source)
    data = json.loads(source.read_text(encoding='utf-8'))
    evaluation = json.loads(Path(args.evaluation).read_text(encoding='utf-8'))
    if data['status'] != 'content_review_pilot_not_training_ready':
        raise ValueError('This auditor is only for an unapproved pilot')
    if set(data['roles']) & set(evaluation['roles']):
        raise ValueError('Evaluation role IDs leaked into pilot')
    ids = [d['id'] for d in data['dialogues']]
    scenes = [d['scene_group'] for d in data['dialogues']]
    if len(set(ids)) != len(ids) or len(set(scenes)) != len(scenes):
        raise ValueError('Duplicate pilot ID or scene group')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    lengths, targets, characters, rows = [], [], [], []
    answers, fingerprints = [], set()
    for dialogue in data['dialogues']:
        prompt = data['common_rules'] + '\n【当前角色】\n' + data['roles'][dialogue['role_id']]
        messages = []
        for turn, pair in enumerate(dialogue['turns'], 1):
            if len(pair) != 2 or not all(isinstance(s, str) and s.strip() == s and s for s in pair):
                raise ValueError('Invalid user/assistant pair')
            messages.extend([{'role': 'user', 'content': pair[0]}, {'role': 'assistant', 'content': pair[1]}])
            row = dict(id=f"{dialogue['id']}-turn-{turn}", source_dialogue_id=dialogue['id'],
                       scene_group=dialogue['scene_group'], role_id=dialogue['role_id'],
                       system_prompt=prompt, messages=list(messages),
                       source_kind='original_ai_synthetic', review_status='AI_audited_not_human_reviewed',
                       training_ready=False)
            encoded = encode_sample(row, tokenizer, max_length=args.max_length, system_prompt=prompt)
            labels = encoded['labels']
            if labels[-1] != tokenizer.convert_tokens_to_ids('<|im_end|>'):
                raise ValueError('Missing expected Qwen chat EOS')
            if all(t == -100 for t in labels):
                raise ValueError('Empty supervised target')
            lengths.append(len(encoded['input_ids']))
            targets.append(sum(t != -100 for t in labels[1:]))
            characters.append(len(pair[1]))
            answers.append(pair[1])
            digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            if digest in fingerprints:
                raise ValueError('Duplicate dialogue prefix')
            fingerprints.add(digest)
            rows.append(row)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    payload = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
    (root / 'pilot-targets.jsonl').write_text(payload, encoding='utf-8')
    report = dict(status='structural_and_token_audit_passed_not_training_ready',
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  evaluation_draft_sha256=hashlib.sha256(Path(args.evaluation).read_bytes()).hexdigest(),
                  source_dialogues=len(data['dialogues']), roles=len(data['roles']), targets=len(rows),
                  dialogues_per_role=dict(Counter(d['role_id'] for d in data['dialogues'])),
                  categories=dict(Counter(d['category'] for d in data['dialogues'])),
                  turns_distribution=dict(Counter(len(d['turns']) for d in data['dialogues'])),
                  max_length=args.max_length, input_tokens=dict(min=min(lengths), median=statistics.median(lengths), max=max(lengths)),
                  target_tokens=dict(min=min(targets), median=statistics.median(targets), max=max(targets), total=sum(targets)),
                  target_characters=dict(min=min(characters), median=statistics.median(characters), max=max(characters)),
                  answers_with_question_marks=sum('？' in a or '?' in a for a in answers),
                  common_three_character_openings=Counter(a[:3] for a in answers).most_common(8),
                  truncation='None: oversize samples raise errors',
                  limitation='AI-authored and AI-inspected, no human or independent quality rating. No semantic leakage guarantee. Not frozen training data.')
    (root / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
