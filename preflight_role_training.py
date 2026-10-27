"""Audit a prompt-only training ablation without loading GPU model weights."""
import argparse
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer

from characters import conversation_messages
from runtime_models import get_profile
from training_data import audit_lccc, encode_sample, read_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/lccc-v1-pilot')
    parser.add_argument('--output', default='outputs/lccc-role-v2-preflight.json')
    args = parser.parse_args()
    audit = audit_lccc(args.data_dir)
    profile = get_profile('qwen3-1.7b')
    tokenizer = AutoTokenizer.from_pretrained(profile['local_path'], local_files_only=True,
                                             revision=profile['revision'])
    prompt = conversation_messages(dict(character_id='companion', messages=[]))[0]['content']
    report = dict(timestamp=datetime.now().astimezone().isoformat(), data=audit,
                  model_id=profile['model_id'], revision=profile['revision'],
                  character_id='companion', system_prompt=prompt,
                  system_prompt_sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                  max_length=512, splits={})
    for split in ('train', 'validation'):
        records = read_records(Path(args.data_dir) / f'{split}.jsonl')
        lengths, original_lengths, chars, trimmed = [], [], [], 0
        for row in records:
            original = encode_sample(row, tokenizer, max_length=512, trim_history=True)
            aligned = encode_sample(row, tokenizer, max_length=512, trim_history=True, system_prompt=prompt)
            old_labels = [value for value in original['labels'] if value != -100]
            new_labels = [value for value in aligned['labels'] if value != -100]
            expected = tokenizer.encode(row['messages'][-1]['content'], add_special_tokens=False)
            expected.append(tokenizer.convert_tokens_to_ids('<|im_end|>'))
            if old_labels != new_labels or new_labels != expected:
                raise ValueError('Adding system changed supervised target tokens: ' + row['id'])
            lengths.append(len(aligned['input_ids']))
            original_lengths.append(len(original['input_ids']))
            chars.append(len(row['messages'][-1]['content']))
            trimmed += aligned['history_pairs_trimmed']
        report['splits'][split] = dict(samples=len(records), original_max_tokens=max(original_lengths),
                                      conditioned_max_tokens=max(lengths),
                                      conditioned_median_tokens=statistics.median(lengths),
                                      target_median_characters=statistics.median(chars),
                                      target_le10_characters=sum(n <= 10 for n in chars),
                                      history_pairs_trimmed=trimmed,
                                      unchanged_supervised_tokens=True)
    report['status'] = 'passed'
    report['note'] = 'Only system conditioning changes. Test responses were not inspected or scored. No GPU training.'
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError('Refusing to overwrite existing preflight: ' + str(target))
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
