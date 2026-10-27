"""Validate and render the original AI-authored pilot; does not start training."""
import argparse
import json
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from characters import conversation_messages
from runtime_models import get_profile
from training_data import encode_sample, file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='data/xiaoman-synthetic-v1/pilot-dialogues.json')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    source = Path(args.source)
    rows = json.loads(source.read_text(encoding='utf-8'))
    if len(rows) != 20 or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Pilot must contain 20 uniquely identified conversations')
    if len({row['scenario_group'] for row in rows}) != len(rows):
        raise ValueError('Pilot scenario groups must be unique')
    from transformers import AutoTokenizer
    profile = get_profile('qwen3-1.7b')
    tokenizer = AutoTokenizer.from_pretrained(profile['local_path'], local_files_only=True,
                                              revision=profile['revision'])
    system = conversation_messages(dict(character_id='companion', messages=[]))[0]['content']
    encoded_lengths, supervised_lengths, records, questions, issues = [], [], [], [], []
    digest_texts = set()
    categories = Counter()
    for row in rows:
        categories[row['category']] += 1
        messages = row['messages']
        if not 4 <= len(messages) <= 12 or len(messages) % 2:
            raise ValueError('Expected 2–6 complete turns: ' + row['id'])
        signature = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if signature in digest_texts:
            raise ValueError('Duplicate conversation: ' + row['id'])
        digest_texts.add(signature)
        for index, message in enumerate(messages):
            expected = 'user' if index % 2 == 0 else 'assistant'
            if message.get('role') != expected or not isinstance(message.get('content'), str):
                raise ValueError('Invalid role or content: ' + row['id'])
            if not message['content'].strip() or message['content'] != message['content'].strip():
                raise ValueError('Source text must have explicit clean boundaries: ' + row['id'])
            if expected == 'assistant':
                if message['content'].count('？') + message['content'].count('?') > 1:
                    issues.append(dict(id=row['id'], turn=index // 2 + 1, reason='more than one question mark'))
                record = dict(id=f"{row['id']}-t{index // 2 + 1:02}", source='AI-synthetic-original',
                              source_dialog_id=row['id'], scenario_group=row['scenario_group'],
                              category=row['category'], split='train_candidate',
                              messages=messages[:index + 1])
                encoded = encode_sample(record, tokenizer, max_length=1024,
                                        trim_history=False, system_prompt=system)
                target = [value for value in encoded['labels'] if value != -100]
                full = tokenizer.apply_chat_template([dict(role='system', content=system)] + record['messages'],
                                                     tokenize=True, return_dict=False,
                                                     add_generation_prompt=False, enable_thinking=False)
                last_eos = max(i for i, token in enumerate(full) if token == tokenizer.eos_token_id)
                if full[:last_eos + 1] != encoded['input_ids']:
                    raise ValueError('Full-template boundary mismatch: ' + record['id'])
                encoded_lengths.append(len(encoded['input_ids']))
                supervised_lengths.append(len(target))
                records.append(record)
            else:
                questions.append(message['content'])
    if issues:
        raise ValueError('Pilot requires content review: ' + json.dumps(issues, ensure_ascii=False))
    if dict(categories) != dict(daily=6, emotional=5, context=4, persona=3, general=2):
        raise ValueError('Pilot category counts differ from the recorded construction plan')
    candidate_file = output / 'training-candidates.jsonl'
    candidate_file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records), encoding='utf-8', newline='\n')
    (output / 'system-prompt.txt').write_text(system, encoding='utf-8', newline='\n')
    report = dict(timestamp=datetime.now().astimezone().isoformat(),
                  source='Original conversations authored by Codex in this session; AI-generated, not human dialogue logs',
                  source_file=str(source), source_sha256=file_hash(source),
                  dialogues=len(rows), categories=dict(categories), assistant_targets=len(records),
                  model_id=profile['model_id'], revision=profile['revision'], max_length=1024,
                  total_tokens=dict(minimum=min(encoded_lengths), median=statistics.median(encoded_lengths), maximum=max(encoded_lengths)),
                  target_tokens_including_eos=dict(minimum=min(supervised_lengths), median=statistics.median(supervised_lengths), maximum=max(supervised_lengths)),
                  full_template_matches=len(records), silently_truncated_targets=0,
                  system_and_history_labels_masked=True,
                  system_prompt_sha256=file_hash(output / 'system-prompt.txt'),
                  candidates_sha256=file_hash(candidate_file),
                  human_review='not performed; AI self-review and programmatic checks only',
                  split='All 20 pilot source conversations remain train candidates; none assigned to validation or final test',
                  limitations=['No semantic independence guarantee from exact-duplicate checks',
                               'A 20-dialogue AI-authored pilot does not represent real users',
                               'No model training, generation, held-out scoring or GPU capacity test performed'],
                  status='pilot_structure_passed_not_training_ready')
    (output / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    review = ['# 小满角色合成对话：首批 20 段', '',
              '这是 Codex 在本次会话编写的 AI 合成训练候选，不是真人聊天记录。尚未训练、没有人工验收分数。', '',
              '先检查是否像你愿意继续聊的对话：接住原话、身份稳定、不过度安慰、不机械追问，也不为凑长度堆话。', '',
              '训练与验证尚未形成完整拆分，这 20 段仅作为训练候选；最终测试不会由这些情节改写。', '']
    for row in rows:
        review.extend([f"## {row['id']} · {row['category']} · {row['scenario_group']}", ''])
        for message in row['messages']:
            review.extend([f"**{'你' if message['role'] == 'user' else '小满'}：** {message['content']}", ''])
    (output / 'pilot-review.md').write_text('\n'.join(review), encoding='utf-8', newline='\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
