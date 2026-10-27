"""Build balanced partial shared-role data with real-tokenizer QC; never mark ready."""
import argparse
import copy
import hashlib
import json
import re
import shutil
import statistics
from collections import Counter
from datetime import datetime
from pathlib import Path

from runtime_models import get_profile
from shared_role_data import (DATA_VERSION, RENDERER_VERSION, audit_shared_dataset,
                              encode_role_sample, load_registry, render_role_prompt,
                              validate_dialogue_splits)
from training_data import file_hash


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')


def load_pilot_dialogues(path, xiaoman_path):
    packet = json.loads(Path(path).read_text(encoding='utf-8'))
    if packet.get('version') != 'shared-role-pilot-v1':
        raise ValueError('不支持的试写数据版本')
    dialogues = copy.deepcopy(packet['dialogues'])
    imported = packet.get('import_xiaoman_ids', [])
    if len(imported) != len(set(imported)):
        raise ValueError('重复导入小满源对话')
    originals = {row['id']: row for row in json.loads(Path(xiaoman_path).read_text(encoding='utf-8'))}
    for source_id in imported:
        if source_id not in originals:
            raise ValueError('小满源对话缺失：' + source_id)
        original = originals[source_id]
        dialogues.append(dict(id='sr-tr-xiaoman-' + source_id, role_id='xiaoman', split='train',
                              category=original['category'], scenario_group=original['scenario_group'],
                              source_origin=dict(file=str(xiaoman_path), dialogue_id=source_id),
                              messages=copy.deepcopy(original['messages'])))
    return sorted(dialogues, key=lambda row: row['id'])


def load_additional_dialogues(path):
    packet = json.loads(Path(path).read_text(encoding='utf-8'))
    if packet.get('version') != 'shared-role-expansion-v1':
        raise ValueError('不支持的扩充数据版本')
    batch_id = packet.get('batch_id')
    if not isinstance(batch_id, str) or not re.fullmatch(r'[a-z0-9-]+', batch_id):
        raise ValueError('扩充批次 ID 无效')
    if not packet.get('source') or not isinstance(packet.get('rows'), list) or not packet['rows']:
        raise ValueError('扩充数据缺少来源或内容')
    rows = []
    for index, entry in enumerate(packet['rows'], 1):
        if not isinstance(entry, list) or len(entry) != 5:
            raise ValueError('扩充行需要角色、拆分、类别、场景和对话')
        role_id, split, category, scene, texts = entry
        if not isinstance(texts, list) or any(not isinstance(text, str) for text in texts):
            raise ValueError('扩充消息必须为文本列表')
        rows.append(dict(id=f'sr-{batch_id}-{index:03}', role_id=role_id, split=split,
                         category=category, scenario_group=scene,
                         source_origin=dict(file=str(path), batch_id=batch_id, row=index, generation=packet['source']),
                         messages=[dict(role='user' if pos % 2 == 0 else 'assistant', content=text)
                                   for pos, text in enumerate(texts)]))
    return rows


def validate_build_counts(counts, registry, expanded_build=False):
    train_ids = {card['id'] for card in registry['roles'] if card['partition'] == 'train'}
    val_ids = {card['id'] for card in registry['roles'] if card['partition'] == 'validation_only'}
    if len(train_ids) != 20 or len(val_ids) != 5 or set(counts['train']) != train_ids:
        raise ValueError('数据包需要20个训练角色和5个验证留出角色')
    train_counts = set(counts['train'].values())
    validation_counts = set(counts['validation'].values())
    if (len(train_counts) != 1 or not 2 <= next(iter(train_counts)) <= 20
            or (not expanded_build and train_counts != {2})):
        raise ValueError('训练角色段数必须均衡，试写为每角色2段，扩充不超过20段')
    if (len(counts['validation']) != 10 or not val_ids <= set(counts['validation'])
            or not set(counts['validation']) <= train_ids | val_ids
            or len(validation_counts) != 1 or not 1 <= next(iter(validation_counts)) <= 5
            or (not expanded_build and validation_counts != {1})):
        raise ValueError('验证需为5个已见角色与5个留出角色，段数均衡且不超过各5段')
    return train_ids, val_ids


def text_grams(messages):
    text = ''.join(message['content'] for message in messages if message['role'] == 'user')
    text = ''.join(re.findall(r'[\w\u4e00-\u9fff]', text.lower()))
    return {text[index:index + 3] for index in range(max(0, len(text) - 2))}


def surface_profile(dialogues):
    """Describe source utterances once, not their repeated training prefixes.

    Keyword indicators are review cues, never semantic correctness scores.
    """
    profile = {}
    for split in ('train', 'validation'):
        source = [row for row in dialogues if row['split'] == split]
        users = [msg['content'] for row in source for msg in row['messages'] if msg['role'] == 'user']
        answers = [msg['content'] for row in source for msg in row['messages'] if msg['role'] == 'assistant']
        constrained = lambda text: bool(re.search(r'别|不要|不用|不想|先不|不要替|不许', text))
        counts = Counter(answers)
        profile[split] = dict(source_dialogues=len(source), user_turns=len(users), assistant_turns=len(answers),
            dialogues_with_constraint_cue=sum(any(constrained(msg['content']) for msg in row['messages']
                                                  if msg['role'] == 'user') for row in source),
            user_turns_with_constraint_cue=sum(constrained(text) for text in users),
            assistant_turns_with_question_mark=sum('?' in text or '？' in text for text in answers),
            assistant_turns_with_multiple_question_marks=sum(text.count('?') + text.count('？') > 1 for text in answers),
            assistant_turns_with_explicit_unset_identity_cue=sum(bool(re.search(r'没.{0,2}设定|未设定|没有给定', text))
                                                                for text in answers),
            exact_duplicate_answers=[dict(text=text, count=count) for text, count in counts.most_common()
                                     if count > 1],
            most_common_two_character_openings=[dict(prefix=prefix, count=count)
                                                for prefix, count in Counter(text[:2] for text in answers).most_common(10)])
    return dict(method='Surface keyword/exact-string indicators on original source turns, not expanded prefixes',
                limitation='No semantic grading, naturalness score, or claim of representative human conversation distribution.',
                splits=profile)


def near_duplicate_report(dialogues):
    train = [(row, text_grams(row['messages'])) for row in dialogues if row['split'] == 'train']
    validation = [(row, text_grams(row['messages'])) for row in dialogues if row['split'] == 'validation']
    pairs = []
    for left, left_grams in train:
        for right, right_grams in validation:
            union = left_grams | right_grams
            score = len(left_grams & right_grams) / len(union) if union else 0.0
            pairs.append(dict(train_id=left['id'], validation_id=right['id'], similarity=round(score, 6)))
    pairs.sort(key=lambda row: row['similarity'], reverse=True)
    return dict(method='Whole user-script Unicode character trigram Jaccard; punctuation/spacing normalized',
                compared_pairs=len(pairs), review_threshold=0.75,
                flagged=[row for row in pairs if row['similarity'] >= 0.75], largest_pairs=pairs[:8],
                limitation='Surface screening only; not proof of semantic independence.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', default='data/shared-role-chat-v1')
    parser.add_argument('--xiaoman-source', default='data/xiaoman-synthetic-v1/pilot-dialogues.json')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--additional-source', action='append', default=[],
                        help='追加原创扩充批次，可重复指定；仍不能直接标为训练就绪')
    args = parser.parse_args()
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError('拒绝覆盖已有数据包：' + str(output))
    registry = load_registry(source / 'role-cards.json')
    dialogues = load_pilot_dialogues(source / 'pilot-dialogues.json', args.xiaoman_source)
    additional_paths = [Path(path) for path in args.additional_source]
    for path in additional_paths:
        dialogues.extend(load_additional_dialogues(path))
    dialogues.sort(key=lambda row: row['id'])
    expanded, counts = validate_dialogue_splits(dialogues, registry)
    train_ids, val_ids = validate_build_counts(counts, registry, expanded_build=bool(additional_paths))
    from transformers import AutoTokenizer, __version__ as transformers_version
    profile = get_profile('qwen3-1.7b')
    tokenizer = AutoTokenizer.from_pretrained(profile['local_path'], revision=profile['revision'], local_files_only=True)
    token_rows, prompts = [], {}
    for card in registry['roles']:
        prompt = render_role_prompt(registry, card['id'])
        prompts[card['id']] = dict(system_prompt=prompt,
                                  sha256=hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                                  tokens=len(tokenizer.encode(prompt, add_special_tokens=False)))
    for split, rows in expanded.items():
        for row in rows:
            encoded = encode_role_sample(row, tokenizer, registry, max_length=1024)
            prompt = prompts[row['role_id']]['system_prompt']
            full = tokenizer.apply_chat_template([dict(role='system', content=prompt)] + row['messages'],
                                                  tokenize=True, return_dict=False,
                                                  add_generation_prompt=False, enable_thinking=False)
            end = max(index for index, token in enumerate(full) if token == tokenizer.eos_token_id)
            if full[:end + 1] != encoded['input_ids']:
                raise ValueError('完整聊天模板与目标拼接不匹配：' + row['id'])
            target = tokenizer.encode(row['messages'][-1]['content'], add_special_tokens=False) + [tokenizer.eos_token_id]
            if [value for value in encoded['labels'] if value != -100] != target:
                raise ValueError('监督目标异常：' + row['id'])
            if encoded['labels'][:-len(target)] != [-100] * (len(encoded['labels']) - len(target)):
                raise ValueError('角色或历史标签泄漏：' + row['id'])
            token_rows.append(dict(id=row['id'], role_id=row['role_id'], split=split,
                                   input_tokens=len(encoded['input_ids']), target_tokens_including_eos=len(target)))
    duplicates = near_duplicate_report(dialogues)
    if duplicates['flagged']:
        raise ValueError('出现跨拆分近重复候选，需要人工审查：' + json.dumps(duplicates['flagged'], ensure_ascii=False))
    output.mkdir(parents=True, exist_ok=False)
    # Preserve the original registry bytes, including its source declaration.
    shutil.copyfile(source / 'role-cards.json', output / 'role-cards.json')
    write_json(output / 'dialogues.json', dialogues)
    for split, rows in expanded.items():
        (output / f'{split}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows),
                                             encoding='utf-8', newline='\n')
    write_json(output / 'rendered-prompts.json', prompts)
    write_json(output / 'token-audit.json', token_rows)
    manifest = dict(version=DATA_VERSION, created_at=datetime.now().astimezone().isoformat(),
                    stage='balanced_expansion_only' if additional_paths else 'balanced_pilot_only',
                    training_ready=False, final_evaluation_frozen=False,
                    reserved_final_role_ids=[], renderer_version=RENDERER_VERSION,
                    provenance=dict(generation='Original AI-authored conversations, not human chat logs',
                                    human_review='Not performed; AI self-review and programmatic QC only',
                                    source_files={str(path): file_hash(path) for path in (
                                        source / 'role-cards.json', source / 'pilot-dialogues.json', Path(args.xiaoman_source),
                                        *additional_paths)}),
                    model=dict(model_id=profile['model_id'], revision=profile['revision'], transformers=transformers_version),
                    max_length=1024, source_dialogues={key: sum(value.values()) for key, value in counts.items()},
                    dialogues_by_role={key: dict(value) for key, value in counts.items()},
                    target_counts={key: len(value) for key, value in expanded.items()},
                    categories={split: dict(Counter(row['category'] for row in dialogues if row['split'] == split))
                                for split in expanded},
                    files={name: dict(sha256=file_hash(output / name)) for name in (
                        'role-cards.json', 'dialogues.json', 'train.jsonl', 'validation.jsonl')},
                    code_sha256={name: file_hash(name) for name in ('shared_role_data.py', 'training_data.py',
                                                                  'prepare_shared_role_data.py')},
                    intended_full_dialogues=dict(train=400, validation=50), near_duplicates=duplicates,
                    remaining=['Expand and review full 400/50 dialogue splits', 'Freeze six new final roles and 30+20 cases',
                               'Full-data preflight and longest-sample GPU memory check'],
                    note='A partial dataset is structurally validated, not training-ready or proof of response quality.')
    write_json(output / 'freeze-manifest.json', manifest)
    audit = audit_shared_dataset(output, require_training_ready=False)
    lengths = [row['input_tokens'] for row in token_rows]
    target_lengths = [row['target_tokens_including_eos'] for row in token_rows]
    qc = dict(status='passed_structural_partial_only' if additional_paths else 'passed_structural_pilot_only', training_ready=False,
              train_roles=len(train_ids), validation_only_roles=len(val_ids),
              dialogues=audit['source_dialogues'], assistant_targets=audit['counts'],
              tokenizer_cases_checked=len(token_rows), full_template_matches=len(token_rows),
              all_context_labels_masked=True, truncated_samples=0,
              prompt_token_range=[min(row['tokens'] for row in prompts.values()), max(row['tokens'] for row in prompts.values())],
              input_tokens=dict(minimum=min(lengths), median=statistics.median(lengths), maximum=max(lengths)),
              target_tokens_including_eos=dict(minimum=min(target_lengths), median=statistics.median(target_lengths), maximum=max(target_lengths)),
              targets_at_least_64_tokens=sum(value >= 64 for value in target_lengths),
              train_targets_at_least_64_tokens=sum(row['split'] == 'train' and row['target_tokens_including_eos'] >= 64
                                                  for row in token_rows),
              dialogue_turn_histogram={split: dict(sorted(Counter(len(row['messages']) // 2 for row in dialogues
                                                                  if row['split'] == split).items()))
                                       for split in expanded},
              target_counts_by_role={split: dict(Counter(row['role_id'] for row in token_rows if row['split'] == split))
                                     for split in expanded},
              surface_diagnostics=surface_profile(dialogues),
              near_duplicates=duplicates, data_audit=audit)
    write_json(output / 'preflight.json', qc)
    review = ['# 共享角色聊天：均衡试写批次', '',
              f"20 个训练角色各 {next(iter(counts['train'].values()))} 段，共 {sum(counts['train'].values())} 段；另有 {sum(counts['validation'].values())} 段验证候选。所有内容是 AI 合成，未人工验收、未训练。", '',
              '验证中的5个角色没有训练样本；最终6个测试角色和题目尚未构造/冻结。本文件不是人工盲评包。', '']
    for card in registry['roles']:
        review.extend([f"## {card['name']} · {card['partition']}", '',
                       f"{card['gender']}，{card['age']}岁，{card['location']}，{card['occupation']}。", '',
                       '**表达习惯：** ' + card['speaking_style'], ''])
        for dialogue in [row for row in dialogues if row['role_id'] == card['id']]:
            review.extend([f"### {dialogue['id']} · {dialogue['split']} · {dialogue['category']}", ''])
            for message in dialogue['messages']:
                review.extend([f"**{'用户' if message['role'] == 'user' else card['name']}：** {message['content']}", ''])
    (output / 'pilot-review.md').write_text('\n'.join(review), encoding='utf-8', newline='\n')
    print(json.dumps({key: value for key, value in qc.items() if key not in ('near_duplicates', 'data_audit', 'surface_diagnostics')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
