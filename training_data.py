"""验证训练数据隔离，并构造只监督助手回答的标签。"""
import hashlib
import json
import re
from pathlib import Path


def read_records(path):
    # JSON strings may contain literal Unicode separators such as U+2028.
    # str.splitlines() splits those inside a valid JSON record as well.
    with Path(path).open(encoding='utf-8') as source:
        return [json.loads(line) for line in source if line.strip()]


def equal_bucket_sample_weights(records, threshold_chars=10):
    """Weight short and longer assistant targets to equal total sample mass.

    This changes only the training objective; it does not remove or duplicate
    examples. Length is measured in Unicode characters, and a short reply is
    not assumed to be low quality. Callers must record this as an experiment.
    """
    if not records:
        raise ValueError('records must not be empty')
    if type(threshold_chars) is not int or threshold_chars < 1:
        raise ValueError('threshold_chars must be a positive integer')
    counts = {'short_le_threshold': 0, 'long_gt_threshold': 0}
    buckets = []
    for row in records:
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages:
            raise ValueError('each record must contain messages')
        target = messages[-1].get('content')
        if not isinstance(target, str) or not target:
            raise ValueError('each record must end with a nonempty text target')
        bucket = 'short_le_threshold' if len(target) <= threshold_chars else 'long_gt_threshold'
        buckets.append(bucket)
        counts[bucket] += 1
    if not counts['short_le_threshold'] or not counts['long_gt_threshold']:
        raise ValueError('equal-bucket weighting requires examples in both length buckets')
    total = len(records)
    weights_by_bucket = {
        bucket: total / (2 * count)
        for bucket, count in counts.items()
    }
    weights = [weights_by_bucket[bucket] for bucket in buckets]
    return weights, {
        'method': 'equal_total_sample_weight_by_target_character_length',
        'threshold_chars': threshold_chars,
        'counts': counts,
        'weights': weights_by_bucket,
        'total_weight_by_bucket': {
            bucket: counts[bucket] * weights_by_bucket[bucket]
            for bucket in counts
        },
        'note': 'Short replies remain in training; length is not a quality label.',
    }


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audit_splits(data_dir='data'):
    root=Path(data_dir)
    version2=(root/'freeze-manifest.json').exists()
    files={name:root/('test.jsonl' if name=='eval' and version2 else f'{name}.jsonl')
           for name in ['train','validation','eval']}
    splits = {name: read_records(path) for name,path in files.items()}
    if version2:
        freeze=json.loads((root/'freeze-manifest.json').read_text(encoding='utf-8'))
        for key,split in [('test','eval'),('validation','validation')]:
            if file_hash(files[split])!=freeze[key]['sha256']:
                raise ValueError(f'冻结集合指纹不匹配：{split}')
        training=json.loads((root/'training-manifest.json').read_text(encoding='utf-8'))
        if file_hash(files['train'])!=training['sha256']:
            raise ValueError('训练集指纹不匹配')
    def normalize(s):
        return re.sub(r'\W+', '', s).lower()
    def grams(s):
        s = normalize(s)
        return {s[i:i+3] for i in range(max(1, len(s)-2))}
    texts = {}
    for split, records in splits.items():
        assert len({r['id'] for r in records}) == len(records)
        texts[split] = [(r['id'], p) for r in records for p in
                       (r['prompts'] if split == 'eval' else [m['content'] for m in r['messages'] if m['role']=='user'])]
        if split != 'eval':
            for r in records:
                assert len(r['messages'])>=2 and len(r['messages'])%2==0
                assert all(m['role']==('user' if i%2==0 else 'assistant') for i,m in enumerate(r['messages']))
                assert all(m['content'].strip() for m in r['messages'])
    maxima, flagged = [], []
    for left, right in [('train', 'validation'), ('train', 'eval'), ('validation', 'eval')]:
        best = (0, '', '')
        for lid, a in texts[left]:
            for rid, b in texts[right]:
                ga, gb = grams(a), grams(b)
                score = len(ga & gb) / len(ga | gb)
                if score > best[0]:
                    best = (score, lid, rid)
                if normalize(a) == normalize(b) or score >= 0.35:
                    flagged.append(dict(left=lid, right=rid, similarity=score))
        maxima.append(dict(splits=[left, right], max_similarity=best[0], pair=best[1:]))
    if flagged:
        raise ValueError(f'发现重复或需审查的近重复问题：{flagged}')
    return dict(counts={k:len(v) for k,v in splits.items()},
                sha256={k:file_hash(files[k]) for k in splits},
                data_dir=str(root),
                prompt_trigram_jaccard_maxima=maxima, flagged=flagged,
                limitation='三字符 Jaccard 只筛查表面近重复，不保证语义无泄漏；另人工检查主题及内容。')


def audit_naturalconv(data_dir):
    """Verify the frozen local NaturalConv pilot using upstream document splits.

    Short greetings are common in conversational corpora, so surface n-gram
    overlap is not used as a rejection rule here. Leakage is checked at the
    source-document level instead.
    """
    root = Path(data_dir)
    manifest = json.loads((root / 'freeze-manifest.json').read_text(encoding='utf-8'))
    files = {name: root / f'{name}.jsonl' for name in ('train', 'validation', 'eval')}
    hashes = {name: file_hash(path) for name, path in files.items()}
    for name, digest in hashes.items():
        if digest != manifest['outputs'][name]['sha256']:
            raise ValueError(f'NaturalConv {name} 文件指纹不匹配')
    source_root = Path(manifest['source_dir'])
    for name, digest in manifest['source_files'].items():
        if file_hash(source_root / name) != digest:
            raise ValueError(f'NaturalConv 源文件指纹不匹配：{name}')

    splits = {name: read_records(path) for name, path in files.items()}
    document_ids = {}
    for split, records in splits.items():
        if len({r['id'] for r in records}) != len(records):
            raise ValueError(f'{split} 存在重复样本 ID')
        document_ids[split] = set()
        for record in records:
            if not record.get('source_dialog_id') or record.get('source_document_id') is None:
                raise ValueError(f'{split} 缺少源对话或文档标识')
            document_ids[split].add(str(record['source_document_id']))
            messages = record.get('messages')
            if not isinstance(messages, list) or not messages:
                raise ValueError(f'{split} 消息为空')
            if split == 'eval':
                if messages[-1]['role'] != 'user' or len(record.get('prompts', [])) != 1:
                    raise ValueError('eval 样本必须以用户消息结束并恰有一条末轮问题')
                if record['prompts'][0] != messages[-1]['content']:
                    raise ValueError('eval 末轮问题与消息上下文不一致')
                if not record.get('reference', '').strip():
                    raise ValueError('eval 样本缺少人工对话目标')
            else:
                if len(messages) < 2 or len(messages) % 2:
                    raise ValueError(f'{split} 样本必须以助手目标结束')
                for index, message in enumerate(messages):
                    expected = 'user' if index % 2 == 0 else 'assistant'
                    if message.get('role') != expected or not message.get('content', '').strip():
                        raise ValueError(f'{split} 样本消息顺序或内容异常')
    for left, right in (('train', 'validation'), ('train', 'eval'), ('validation', 'eval')):
        overlap = document_ids[left] & document_ids[right]
        if overlap:
            raise ValueError(f'{left}/{right} 按源文档泄漏：{len(overlap)} 个文档')
    counts = {name: len(rows) for name, rows in splits.items()}
    for name, count in counts.items():
        if count != manifest['outputs'][name]['count']:
            raise ValueError(f'NaturalConv {name} 样本数不匹配')
    return dict(counts=counts, sha256=hashes, data_dir=str(root),
                source_split_unit='document_id', source_documents={k: len(v) for k, v in document_ids.items()},
                cross_split_document_overlap=0,
                limitation='NaturalConv 两位真人标注者对话映射为 user/assistant；常见寒暄短句不作为泄漏证据。')


def audit_lccc(data_dir):
    """Verify a frozen LCCC pilot by source-dialogue IDs and content hashes."""
    from inspect_lccc import CONTACT_PATTERNS
    contact_request = re.compile(
        r'(?:加我(?:微信|微.?信|QQ)|私信我|发我(?:微信|QQ)|(?:微信|QQ)号|手机号|身份证号|联系方式)'
    )

    root = Path(data_dir)
    manifest = json.loads((root / 'freeze-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != 'lccc-base-pilot-v1':
        raise ValueError('LCCC 冻结清单版本不匹配')
    files = {name: root / f'{name}.jsonl' for name in ('train', 'validation', 'eval')}
    hashes = {name: file_hash(path) for name, path in files.items()}
    for name, digest in hashes.items():
        if digest != manifest['outputs'][name]['sha256']:
            raise ValueError(f'LCCC {name} 文件指纹不匹配')

    source_root = Path(manifest['source_dir'])
    source_names = {'train': 'lccc_base_train.jsonl.gz',
                    'validation': 'lccc_base_valid.jsonl.gz',
                    'test': 'lccc_base_test.jsonl.gz'}
    for split, filename in source_names.items():
        if file_hash(source_root / filename) != manifest['source_files'][split]:
            raise ValueError(f'LCCC 源文件指纹不匹配：{filename}')

    expected_source_split = {'train': 'train', 'validation': 'validation', 'eval': 'test'}
    splits = {name: read_records(path) for name, path in files.items()}
    source_hashes = {}
    source_ids = {}
    for split, records in splits.items():
        if len({row['id'] for row in records}) != len(records):
            raise ValueError(f'LCCC {split} 存在重复样本 ID')
        source_hashes[split], source_ids[split] = set(), set()
        for row in records:
            if row.get('source') != 'LCCC-base' or row.get('source_split') != expected_source_split[split]:
                raise ValueError(f'LCCC {split} 的来源拆分标记异常')
            if not row.get('source_dialog_id') or not row.get('source_dialog_hash'):
                raise ValueError(f'LCCC {split} 缺少源对话标识')
            source_ids[split].add(row['source_dialog_id'])
            source_hashes[split].add(row['source_dialog_hash'])
            messages = row.get('messages')
            if not isinstance(messages, list) or not messages:
                raise ValueError(f'LCCC {split} 消息为空')
            if split == 'eval':
                if messages[-1].get('role') != 'user' or row.get('prompts') != [messages[-1].get('content')]:
                    raise ValueError('LCCC eval 必须以当前用户消息结束，且 prompts 一致')
                reference = row.get('reference', '')
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError('LCCC eval 缺少参考回复')
                texts = [message.get('content', '') for message in messages] + [reference]
            else:
                if len(messages) < 2 or len(messages) % 2:
                    raise ValueError(f'LCCC {split} 样本必须以助手目标结束')
                for index, message in enumerate(messages):
                    expected_role = 'user' if index % 2 == 0 else 'assistant'
                    if message.get('role') != expected_role or not message.get('content', '').strip():
                        raise ValueError(f'LCCC {split} 消息顺序或内容异常')
                texts = [message['content'] for message in messages]
            if any(pattern.search(text) for pattern in CONTACT_PATTERNS.values() for text in texts) or any(
                contact_request.search(text) for text in texts
            ):
                raise ValueError(f'LCCC {split} 仍含常见联系方式模式')
    for left, right in (('train', 'validation'), ('train', 'eval'), ('validation', 'eval')):
        if source_hashes[left] & source_hashes[right]:
            raise ValueError(f'LCCC {left}/{right} 存在相同来源对话')

    counts = {name: len(rows) for name, rows in splits.items()}
    source_dialog_counts = {name: len(ids) for name, ids in source_ids.items()}
    for name, count in counts.items():
        if count != manifest['outputs'][name]['count']:
            raise ValueError(f'LCCC {name} 样本数不匹配')
    if source_dialog_counts != manifest['selected_dialogue_counts_by_output']:
        raise ValueError('LCCC 输出源对话数量与冻结清单不匹配')
    return dict(counts=counts, sha256=hashes, data_dir=str(root),
                source_dialogues=source_dialog_counts, cross_split_exact_dialogue_overlap=0,
                filtered_contact_patterns=0,
                limitation='基于作者提供的 train/validation/test 划分；只检查常见联系方式格式，无法证明内容已完全去标识。')


def encode_sample(record, tokenizer, max_length=512, trim_history=False, system_prompt=None):
    if system_prompt is not None and (not isinstance(system_prompt, str) or not system_prompt.strip()):
        raise ValueError('system_prompt 必须是非空文本或 None')
    messages = record['messages']
    if len(messages)<2 or len(messages)%2 or any(
        m['role']!=('user' if i%2==0 else 'assistant') or not m['content'].strip()
        for i,m in enumerate(messages)
    ):
        raise ValueError('训练消息必须是非空、交替的 user/assistant，且以助手目标结尾')
    # Literal framing tokens would be interpreted as dialogue boundaries rather
    # than ordinary text and can add an early EOS inside the supervised target.
    framing = ('<|im_start|>', '<|im_end|>', '<|endoftext|>')
    if any(marker in message['content'] for message in messages for marker in framing):
        raise ValueError('训练消息不能包含 Qwen 对话控制标记，请先在数据准备阶段处理')
    response = messages[-1]
    # Qwen's full-message template strips leading newlines and parses thinking
    # blocks; concatenating either as a plain target would serialize differently.
    if response['content'].startswith('\n') or response['content'].startswith('\r'):
        raise ValueError('助手目标不能以换行开头，请在数据准备阶段显式规范化并记录')
    if '<think>' in response['content'] or '</think>' in response['content']:
        raise ValueError('当前训练入口只接受不含 thinking 标记的可见助手回答')
    answer = tokenizer.encode(response['content'], add_special_tokens=False)
    context = messages[:-1]
    trimmed_pairs = 0
    # 使用推理时的同一模板前缀（含空 thinking 区），监督答案及结束标记。
    eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if eos is None or eos == tokenizer.unk_token_id:
        raise ValueError('缺少 Qwen 对话结束标记')
    while True:
        conditioning = [dict(role='system', content=system_prompt)] if system_prompt is not None else []
        prefix = tokenizer.apply_chat_template(conditioning + context, tokenize=True,
                                               add_generation_prompt=True, enable_thinking=False,
                                               return_dict=False)
        ids = prefix + answer + [eos]
        if len(ids) <= max_length:
            break
        if not trim_history or len(context) < 3:
            raise ValueError(f"{record['id']} 超过长度上限，拒绝静默截断")
        if context[0]['role'] != 'user' or context[1]['role'] != 'assistant':
            raise ValueError(f"{record['id']} 无法按完整对话轮次裁剪历史")
        context = context[2:]
        trimmed_pairs += 1
    labels = [-100] * len(prefix) + answer + [eos]
    assert any(x != -100 for x in labels) and labels[-1] == eos
    result = dict(input_ids=ids, attention_mask=[1]*len(ids), labels=labels)
    if trim_history:
        result['history_pairs_trimmed'] = trimmed_pairs
    return result
