"""Create a frozen, small NaturalConv SFT pilot from the upstream splits.

The source corpus is restricted to personal non-commercial research and local
development/testing. Keep both source and derived files local; do not package
them with the public application.
"""
import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SOURCE_FILES = (
    'dialog_release.json', 'document_url_release.json',
    'train.txt', 'dev.txt', 'test.txt', 'LICENSE', 'README.md',
)
TRAIN_TARGET_INDICES = (1, 5, 9, 13, 17)
HOLDOUT_TARGET_INDEX = 9


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_ids(path):
    values = [line.strip() for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f'官方划分文件有重复 ID：{path}')
    return values


def stratified_dialog_ids(train_ids, dialog_by_id, topic_by_document, limit, seed):
    by_topic = defaultdict(list)
    for dialog_id in train_ids:
        dialog = dialog_by_id[dialog_id]
        topic = topic_by_document[str(dialog['document_id'])]
        by_topic[topic].append(dialog_id)
    if limit is None or limit >= len(train_ids):
        selected = set(train_ids)
    else:
        if limit <= 0:
            raise ValueError('--train-dialogues 必须大于0')
        quotas = {topic: limit * len(ids) / len(train_ids) for topic, ids in by_topic.items()}
        counts = {topic: math.floor(value) for topic, value in quotas.items()}
        remaining = limit - sum(counts.values())
        for topic in sorted(quotas, key=lambda key: (-(quotas[key] - counts[key]), key))[:remaining]:
            counts[topic] += 1
        rng = random.Random(seed)
        selected = set()
        for topic, ids in sorted(by_topic.items()):
            selected.update(rng.sample(ids, counts[topic]))
    return [dialog_id for dialog_id in train_ids if dialog_id in selected]


def role_messages(dialogue, final_index):
    """Map the first human participant to user and the second to assistant."""
    content = dialogue['content'][:final_index + 1]
    if final_index % 2 != 1 or len(content) != final_index + 1:
        raise ValueError('目标索引必须是存在的助手轮次')
    if any(not isinstance(text, str) or not text.strip() for text in content):
        raise ValueError(f"{dialogue['dialog_id']} 含空或非文本话语")
    messages = [dict(role='user' if index % 2 == 0 else 'assistant', content=text)
                for index, text in enumerate(content)]
    if messages[-1]['role'] != 'assistant':
        raise AssertionError('目标映射没有以 assistant 结束')
    return messages


def sft_record(dialogue, topic, target_index):
    return dict(
        id=f"nc_{dialogue['dialog_id']}_t{target_index:02d}",
        source='NaturalConv',
        source_dialog_id=str(dialogue['dialog_id']),
        source_document_id=str(dialogue['document_id']),
        topic=topic,
        target_turn_index=target_index,
        messages=role_messages(dialogue, target_index),
    )


def eval_record(dialogue, topic, target_index):
    full = role_messages(dialogue, target_index)
    context = full[:-1]
    if context[-1]['role'] != 'user':
        raise AssertionError('评估问题上下文应以用户发言结束')
    return dict(
        id=f"nc_eval_{dialogue['dialog_id']}_t{target_index:02d}",
        source='NaturalConv',
        source_dialog_id=str(dialogue['dialog_id']),
        source_document_id=str(dialogue['document_id']),
        topic=topic,
        target_turn_index=target_index,
        messages=context,
        prompts=[context[-1]['content']],
        reference=full[-1]['content'],
    )


def write_jsonl(path, rows):
    raw = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                  for row in rows).encode('utf-8')
    Path(path).write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def prepare(source_dir, output_dir, train_dialogues=600, seed=20261023, source_origin='unverified'):
    source = Path(source_dir)
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f'拒绝覆盖冻结数据目录：{destination}')
    for name in SOURCE_FILES:
        if not (source / name).is_file():
            raise FileNotFoundError(f'缺少 NaturalConv 官方文件：{source / name}')

    dialogues = json.loads((source / 'dialog_release.json').read_text(encoding='utf-8'))
    documents = json.loads((source / 'document_url_release.json').read_text(encoding='utf-8'))
    if not isinstance(dialogues, list) or not isinstance(documents, list):
        raise ValueError('NaturalConv JSON 根节点必须为列表')
    dialog_by_id = {str(row['dialog_id']): row for row in dialogues}
    topic_by_document = {str(row['document_id']): row['topic'] for row in documents}
    if len(dialog_by_id) != len(dialogues) or len(topic_by_document) != len(documents):
        raise ValueError('NaturalConv 存在重复对话或文档 ID')

    official = {
        'train': read_ids(source / 'train.txt'),
        'validation': read_ids(source / 'dev.txt'),
        'eval': read_ids(source / 'test.txt'),
    }
    all_ids = [dialog_id for ids in official.values() for dialog_id in ids]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(dialog_by_id):
        raise ValueError('官方 train/dev/test 划分未能互斥且完整覆盖对话')
    for split, ids in official.items():
        missing = set(ids) - set(dialog_by_id)
        if missing:
            raise ValueError(f'{split} 引用了不存在的对话：{next(iter(missing))}')

    document_splits = {}
    for split, ids in official.items():
        current = set()
        for dialog_id in ids:
            document_id = str(dialog_by_id[dialog_id]['document_id'])
            if document_id not in topic_by_document:
                raise ValueError(f'{dialog_id} 的 document_id 不存在于文档索引')
            current.add(document_id)
        document_splits[split] = current
    for left, right in (('train', 'validation'), ('train', 'eval'), ('validation', 'eval')):
        if document_splits[left] & document_splits[right]:
            raise ValueError(f'官方 {left}/{right} 存在源文档交叉，拒绝沿用此评估划分')

    selected = stratified_dialog_ids(
        official['train'], dialog_by_id, topic_by_document, train_dialogues, seed,
    )
    train_rows = []
    for dialog_id in selected:
        dialogue = dialog_by_id[dialog_id]
        content = dialogue['content']
        if len(content) <= max(TRAIN_TARGET_INDICES):
            raise ValueError(f'{dialog_id} 的对话过短，缺少固定训练目标轮次')
        if any('\ufffd' in text or '\x00' in text for text in content):
            raise ValueError(f'{dialog_id} 存在替换字符或 NUL，疑似编码损坏')
        topic = topic_by_document[str(dialogue['document_id'])]
        train_rows.extend(sft_record(dialogue, topic, index) for index in TRAIN_TARGET_INDICES)

    def holdout_rows(split):
        rows = []
        for dialog_id in official[split]:
            dialogue = dialog_by_id[dialog_id]
            if len(dialogue['content']) <= HOLDOUT_TARGET_INDEX:
                raise ValueError(f'{dialog_id} 缺少验证/测试目标轮次')
            if any('\ufffd' in text or '\x00' in text for text in dialogue['content']):
                raise ValueError(f'{dialog_id} 存在替换字符或 NUL，疑似编码损坏')
            topic = topic_by_document[str(dialogue['document_id'])]
            if split == 'validation':
                rows.append(sft_record(dialogue, topic, HOLDOUT_TARGET_INDEX))
            else:
                rows.append(eval_record(dialogue, topic, HOLDOUT_TARGET_INDEX))
        return rows

    validation_rows = holdout_rows('validation')
    eval_rows = holdout_rows('eval')
    if not train_rows or not validation_rows or not eval_rows:
        raise ValueError('训练、验证、测试样本都必须非空')

    destination.mkdir(parents=True, exist_ok=False)
    output_hashes = {
        'train': write_jsonl(destination / 'train.jsonl', train_rows),
        'validation': write_jsonl(destination / 'validation.jsonl', validation_rows),
        'eval': write_jsonl(destination / 'eval.jsonl', eval_rows),
    }
    source_hashes = {name: sha256(source / name) for name in SOURCE_FILES}
    source_path = str(source.resolve().relative_to(Path.cwd().resolve()))
    manifest = dict(
        version='naturalconv-v1-pilot',
        frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        source='NaturalConv; source attribution is not authenticated by this script',
        source_origin=source_origin,
        source_dir=source_path,
        source_files=source_hashes,
        license='Tencent AI Lab NaturalConv Dataset Terms and Conditions',
        use_scope='personal non-commercial research, development, and testing only; keep source and derivatives local; do not distribute',
        upstream_dialogues=len(dialogues),
        upstream_documents=len(documents),
        official_dialogue_counts={key: len(value) for key, value in official.items()},
        official_document_counts={key: len(value) for key, value in document_splits.items()},
        official_document_overlaps=0,
        selected_train_dialogues=len(selected),
        selected_train_topics=dict(Counter(topic_by_document[str(dialog_by_id[i]['document_id'])] for i in selected)),
        train_target_turn_indices=list(TRAIN_TARGET_INDICES),
        validation_and_eval_target_turn_index=HOLDOUT_TARGET_INDEX,
        mapping='Each source dialogue alternates two human speakers; source speaker one maps to user, speaker two to assistant.',
        seed=seed,
        outputs={
            'train': dict(count=len(train_rows), sha256=output_hashes['train']),
            'validation': dict(count=len(validation_rows), sha256=output_hashes['validation']),
            'eval': dict(count=len(eval_rows), sha256=output_hashes['eval']),
        },
        limitations=[
            'This is an initial 600-dialogue pilot, not the full training split.',
            'NaturalConv is scenario-elicited human-human dialogue, not real IM app logs.',
            'The source and derivative files are restricted; this run is not for public model/data distribution or promotion.',
        ],
    )
    (destination / 'freeze-manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8',
    )
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dir', default='data/NaturalConv')
    parser.add_argument('--output-dir', default='data/naturalconv-v1')
    parser.add_argument('--train-dialogues', type=int, default=600)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--source-origin', default='unverified',
                        help='record the actual download channel; this does not verify licensing')
    args = parser.parse_args()
    manifest = prepare(args.source_dir, args.output_dir, args.train_dialogues, args.seed, args.source_origin)
    print(json.dumps({
        'status': 'prepared',
        'output_dir': args.output_dir,
        'train_dialogues': manifest['selected_train_dialogues'],
        'train_examples': manifest['outputs']['train']['count'],
        'validation_examples': manifest['outputs']['validation']['count'],
        'eval_examples': manifest['outputs']['eval']['count'],
        'train_topics': manifest['selected_train_topics'],
        'official_documents': manifest['official_document_counts'],
        'official_document_overlaps': manifest['official_document_overlaps'],
        'license_scope': manifest['use_scope'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
