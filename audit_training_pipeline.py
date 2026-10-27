"""Read-only stage 1 audit with the pinned real tokenizer; no GPU model load.

Writes train/validation diagnostics to a new local output directory. Does not
read final evaluation examples, modify old runs, or award language-quality scores.
"""
import argparse
import copy
import hashlib
import json
import statistics
from datetime import datetime
from pathlib import Path

from characters import character_profile_reply, conversation_messages
from runtime_models import get_profile
from training_data import encode_sample, equal_bucket_sample_weights, file_hash, read_records


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def distribution(values):
    return dict(count=len(values), minimum=min(values), median=statistics.median(values),
                maximum=max(values), sum=sum(values))


def template_target_ids(tokenizer, messages, eos):
    """Serialize the entire dialogue independently; ignore only post-EOS newline."""
    result = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                           add_generation_prompt=False, enable_thinking=False)
    end = max(i for i, token in enumerate(result) if token == eos)
    return result[:end + 1]


def inspect_row(row, tokenizer, system, max_length, eos):
    before = copy.deepcopy(row)
    encoded = encode_sample(row, tokenizer, max_length=max_length, trim_history=True,
                            system_prompt=system)
    assert row == before, 'Encoding mutated source record'
    target = tokenizer.encode(row['messages'][-1]['content'], add_special_tokens=False) + [eos]
    supervised = [value for value in encoded['labels'] if value != -100]
    assert supervised == target, 'Supervised target differs from assistant + EOS'
    assert eos not in target[:-1], 'An answer contains premature EOS'
    boundary = len(encoded['input_ids']) - len(target)
    assert encoded['labels'][:boundary] == [-100] * boundary
    assert encoded['input_ids'][boundary:] == target
    assert len(encoded['attention_mask']) == len(encoded['input_ids']) == len(encoded['labels'])
    assert all(value == 1 for value in encoded['attention_mask'])
    trimmed = encoded['history_pairs_trimmed']
    retained = row['messages'][2 * trimmed:]
    context = ([dict(role='system', content=system)] if system else []) + retained[:-1]
    prefix = tokenizer.apply_chat_template(context, tokenize=True, return_dict=False,
                                           add_generation_prompt=True, enable_thinking=False)
    assert prefix == encoded['input_ids'][:boundary]
    assert tokenizer.decode(prefix).endswith('<think>\n\n</think>\n\n')
    if system:
        assert context[0]['content'] == system
    assert context[-1] == row['messages'][-2]
    full_messages = ([dict(role='system', content=system)] if system else []) + retained
    reference = template_target_ids(tokenizer, full_messages, eos)
    source_texts = [message['content'] for message in row['messages']]
    special_in_content = [token for token in tokenizer.all_special_tokens
                          if any(token in content for content in source_texts)]
    return dict(total_tokens=len(encoded['input_ids']), target_tokens=len(target),
                target_characters=len(row['messages'][-1]['content']),
                history_pairs_trimmed=trimmed,
                template_reference_equal=reference == encoded['input_ids'],
                leading_whitespace=bool(row['messages'][-1]['content'][:1].isspace()),
                think_in_content=any('<think>' in text or '</think>' in text for text in source_texts),
                special_in_content=special_in_content,
                prefix_boundary=boundary, encoded=encoded)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/lora-lccc-base-v4-balanced')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    run = Path(args.run_dir)
    config = json.loads((run / 'run-config.json').read_text(encoding='utf-8'))
    profile = get_profile(config['model_profile'])
    assert config['base_model'] == profile['model_id'] and config['revision'] == profile['revision']
    from transformers import AutoTokenizer, __version__ as transformers_version
    tokenizer = AutoTokenizer.from_pretrained(profile['local_path'], local_files_only=True,
                                              revision=profile['revision'])
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    assert eos == tokenizer.eos_token_id
    generation = json.loads((Path(profile['local_path']) / 'generation_config.json').read_text(encoding='utf-8'))
    assert eos in generation['eos_token_id']
    current_prompt = conversation_messages(dict(character_id='companion', messages=[]))[0]['content']
    training_prompt = config.get('conditioning', {}).get('system_prompt')
    data_root = Path(config['data']['data_dir'])
    report = dict(timestamp=datetime.now().astimezone().isoformat(), stage='1.1',
                  model_id=profile['model_id'], revision=profile['revision'],
                  transformers=transformers_version, run_dir=str(run),
                  scope='train and validation only; no model load, generation or test-case reading',
                  run_config_sha256=file_hash(run / 'run-config.json'),
                  tokenizer_config_sha256=file_hash(Path(profile['local_path']) / 'tokenizer_config.json'),
                  eos_token_id=eos, generation_eos_ids=generation['eos_token_id'],
                  training_system_present=bool(training_prompt),
                  current_system_tokens=len(tokenizer.encode(current_prompt, add_special_tokens=False)),
                  current_system_sha256=sha_text(current_prompt),
                  max_training_length=config['max_length'], app_input_budget=2048, splits={})
    (output / 'current-system-prompt.txt').write_text(current_prompt, encoding='utf-8')
    visuals, mismatches = [], []
    for split in ('train', 'validation'):
        path = data_root / f'{split}.jsonl'
        assert file_hash(path) == config['data']['sha256'][split]
        rows = read_records(path)
        details, aligned_lengths, aligned_failures, prompt_variants = [], [], [], set()
        aligned_trimmed = 0
        at_1024_lengths, at_1024_trimmed, at_1024_rejected = [], 0, 0
        for row in rows:
            detail = inspect_row(row, tokenizer, training_prompt, config['max_length'], eos)
            encoded = detail.pop('encoded')
            if not detail['template_reference_equal']:
                mismatches.append(dict(split=split, id=row['id'], leading_whitespace=detail['leading_whitespace']))
            if len(visuals) < 8 and split == 'train':
                boundary = detail['prefix_boundary']
                visuals.append(dict(id=row['id'], context_tail=tokenizer.decode(encoded['input_ids'][max(0, boundary - 25):boundary]),
                                    supervised_text=tokenizer.decode(encoded['input_ids'][boundary:]),
                                    target_tokens=[dict(position=i, token_id=token,
                                                        token=tokenizer.convert_ids_to_tokens(token), label=label)
                                                   for i, (token, label) in enumerate(zip(encoded['input_ids'], encoded['labels']))
                                                   if i >= max(0, boundary - 4)]))
            details.append(detail)
            app_messages = conversation_messages(dict(character_id='companion', messages=row['messages'][:-1]))
            prompt_variants.add(sha_text(app_messages[0]['content']))
            try:
                aligned = inspect_row(row, tokenizer, current_prompt, config['max_length'], eos)
                aligned_lengths.append(aligned['total_tokens'])
                aligned_trimmed += aligned['history_pairs_trimmed']
            except ValueError as exc:
                aligned_failures.append(dict(id=row['id'], reason=str(exc)))
            try:
                expanded = encode_sample(row, tokenizer, max_length=1024, trim_history=True,
                                         system_prompt=current_prompt)
                at_1024_lengths.append(len(expanded['input_ids']))
                at_1024_trimmed += expanded['history_pairs_trimmed']
            except ValueError:
                at_1024_rejected += 1
        if split == 'train':
            weights, weight_audit = equal_bucket_sample_weights(rows)
            eos_mass = sum(weight / item['target_tokens'] for weight, item in zip(weights, details)) / sum(weights)
            report['training_sample_weighting'] = weight_audit
            report['eos_loss_mass_proxy'] = dict(
                unweighted_mean_per_sample=sum(1 / item['target_tokens'] for item in details) / len(details),
                v4_length_weighted_mean_per_sample=eos_mass,
                token_aggregate=len(details) / sum(item['target_tokens'] for item in details),
                note='Weight mass assigned to EOS if per-token losses were equal; not measured loss, gradient or causal attribution.')
        report['splits'][split] = dict(
            sha256=file_hash(path), samples=len(rows), total_tokens=distribution([d['total_tokens'] for d in details]),
            target_tokens_including_eos=distribution([d['target_tokens'] for d in details]),
            target_characters=distribution([d['target_characters'] for d in details]),
            targets_le10_characters=sum(d['target_characters'] <= 10 for d in details),
            targets_le5_tokens_including_eos=sum(d['target_tokens'] <= 5 for d in details),
            history_pairs_trimmed=sum(d['history_pairs_trimmed'] for d in details),
            template_reference_mismatches=sum(not d['template_reference_equal'] for d in details),
            special_token_content_rows=sum(bool(d['special_in_content']) for d in details),
            think_content_rows=sum(d['think_in_content'] for d in details),
            current_prompt_at_512=dict(fit_samples=len(aligned_lengths), rejected_samples=len(aligned_failures),
                                       history_pairs_trimmed=aligned_trimmed,
                                       first_rejections=aligned_failures[:5]),
            current_prompt_at_1024=dict(fit_samples=len(at_1024_lengths), rejected_samples=at_1024_rejected,
                                        history_pairs_trimmed=at_1024_trimmed,
                                        total_tokens=distribution(at_1024_lengths)),
            app_system_prompt_variants=len(prompt_variants))
    # Edge probes are original diagnostics, not final evaluation examples.
    probes = []
    for answer in ('自然的短句。', ' Hello', '\n你好', '<think>推理</think>回答', '回答<|im_end|>'):
        row = dict(id='edge', messages=[dict(role='user', content='一起聊聊吧。'), dict(role='assistant', content=answer)])
        try:
            item = inspect_row(row, tokenizer, None, 512, eos)
            item.pop('encoded')
            probes.append(dict(answer=answer, result=item))
        except (ValueError, AssertionError) as exc:
            probes.append(dict(answer=answer, rejection=str(exc)))
    questions = ('你今年多大？', '我刚把一个报错自己排查好了。', '你平时喜欢什么？')
    report['app_policy_probes'] = [dict(question=q, rule_reply=character_profile_reply('companion', q),
                                      system_sha256=sha_text(conversation_messages(dict(character_id='companion', messages=[]), q)[0]['content']))
                                 for q in questions]
    report['template_reference_mismatches'] = mismatches[:20]
    report['edge_probes'] = probes
    report['source_sha256'] = {name: file_hash(name) for name in (
        'training_data.py', 'train_lora.py', 'inference.py', 'context_window.py', 'characters.py', 'generation_options.py')}
    report['status'] = 'completed_with_findings'
    report['quality_improvement_claimed'] = False
    (output / 'supervision-examples.json').write_text(json.dumps(visuals, ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], report=str(output / 'audit.json'),
                          current_system_tokens=report['current_system_tokens'],
                          splits={name: {key: item[key] for key in (
                              'samples', 'template_reference_mismatches', 'current_prompt_at_512',
                              'current_prompt_at_1024')} for name, item in report['splits'].items()}),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
