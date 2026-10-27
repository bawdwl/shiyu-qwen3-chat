"""Frozen per-sample character conditioning and leakage checks for shared SFT.

No torch/model imports. The renderer is also usable by raw-model evaluation so
training and inference can receive the same role card without application rules.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from training_data import encode_sample, file_hash, read_records

DATA_VERSION = 'shared-role-chat-v1'
CARD_VERSION = 'shared-role-cards-v1'
RENDERER_VERSION = 'shared-role-prompt-v1'
CONTROL_MARKERS = ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>')
CATEGORIES = {'daily', 'emotional', 'context', 'persona', 'general'}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()


def clean_text(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f'{name} 必须是无首尾空白的非空文本')
    if any(marker in value for marker in CONTROL_MARKERS):
        raise ValueError(f'{name} 不能包含对话或思考控制标记')
    return value


def validate_registry(registry):
    if not isinstance(registry, dict) or registry.get('version') != CARD_VERSION:
        raise ValueError('角色卡清单版本不匹配')
    clean_text(registry.get('common_rules'), 'common_rules')
    roles = registry.get('roles')
    if not isinstance(roles, list) or not roles:
        raise ValueError('角色卡清单为空')
    index = {}
    for card in roles:
        for name in ('id', 'version', 'name', 'gender', 'location', 'occupation',
                     'speaking_style', 'world', 'creation_boundary'):
            clean_text(card.get(name), name)
        if type(card.get('age')) is not int or card['age'] < 1:
            raise ValueError('角色年龄必须是正整数；虚构世界年龄也需明确')
        if card.get('partition') not in ('train', 'validation_only', 'final_only'):
            raise ValueError('角色必须声明 train、validation_only 或 final_only 分区')
        for name in ('personality', 'interests', 'unknown_fields'):
            if not isinstance(card.get(name), list) or not card[name]:
                raise ValueError(f'{name} 必须是非空文本列表')
            for value in card[name]:
                clean_text(value, name)
        if card['id'] in index:
            raise ValueError('重复 role_id：' + card['id'])
        index[card['id']] = card
    return index


def load_registry(path):
    registry = json.loads(Path(path).read_text(encoding='utf-8'))
    validate_registry(registry)
    return registry


def role_card(registry, role_id):
    index = validate_registry(registry)
    if role_id not in index:
        raise ValueError('未知角色，禁止回退为默认人物：' + str(role_id))
    return index[role_id]


def render_role_prompt(registry, role_id):
    card = role_card(registry, role_id)
    # Training-only partition IDs and provenance metadata are not model input.
    return '\n'.join((registry['common_rules'], '【当前角色卡】',
                      f"姓名：{card['name']}；性别：{card['gender']}；年龄：{card['age']}岁。",
                      f"所在地：{card['location']}；职业：{card['occupation']}。",
                      '性格：' + '、'.join(card['personality']) + '。',
                      '兴趣：' + '、'.join(card['interests']) + '。',
                      '表达习惯：' + card['speaking_style'],
                      '世界背景：' + card['world'],
                      '创作边界：' + card['creation_boundary'],
                      '尚未设定：' + '、'.join(card['unknown_fields']) + '。'))


def validate_messages(messages):
    if not isinstance(messages, list) or not 4 <= len(messages) <= 12 or len(messages) % 2:
        raise ValueError('共享训练源对话必须包含2–6轮完整问答')
    for position, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {'role', 'content'}:
            raise ValueError('消息仅允许 role/content 字段')
        if message['role'] != ('user' if position % 2 == 0 else 'assistant'):
            raise ValueError('消息角色顺序不匹配')
        clean_text(message['content'], 'message.content')


def expand_dialogue(dialogue, registry):
    for key in ('id', 'role_id', 'scenario_group'):
        clean_text(dialogue.get(key), key)
    if dialogue.get('split') not in ('train', 'validation') or dialogue.get('category') not in CATEGORIES:
        raise ValueError('对话拆分或场景类型无效')
    if any(key in dialogue for key in ('system_prompt', 'system', 'role_card')):
        raise ValueError('对话不能覆盖已冻结的角色卡或 system')
    card = role_card(registry, dialogue['role_id'])
    if dialogue['split'] == 'train' and card['partition'] != 'train':
        raise ValueError('验证留出角色不能进入训练集')
    validate_messages(dialogue.get('messages'))
    prompt_hash = hashlib.sha256(render_role_prompt(registry, card['id']).encode('utf-8')).hexdigest()
    rows = []
    for position in range(1, len(dialogue['messages']), 2):
        rows.append(dict(id=f"{dialogue['id']}-t{position // 2 + 1:02}",
                         source='AI-synthetic-original', source_dialog_id=dialogue['id'],
                         source_dialog_hash=content_hash(dialogue['messages']),
                         scenario_group=dialogue['scenario_group'], category=dialogue['category'],
                         split=dialogue['split'], role_id=card['id'],
                         validation_scope=('seen_role' if card['partition'] == 'train' else 'unseen_role')
                         if dialogue['split'] == 'validation' else None,
                         role_card_version=card['version'], role_card_sha256=content_hash(card),
                         renderer_version=RENDERER_VERSION, system_prompt_sha256=prompt_hash,
                         messages=dialogue['messages'][:position + 1]))
    return rows


def encode_role_sample(record, tokenizer, registry, max_length=1024):
    if any(key in record for key in ('system_prompt', 'system', 'role_card')):
        raise ValueError('样本不能覆盖已冻结的角色卡或 system')
    card = role_card(registry, record.get('role_id'))
    if record.get('role_card_version') != card['version'] or record.get('role_card_sha256') != content_hash(card):
        raise ValueError('样本角色卡版本或指纹不匹配')
    if record.get('split') not in ('train', 'validation'):
        raise ValueError('样本拆分无效')
    if record['split'] == 'train' and card['partition'] != 'train':
        raise ValueError('验证留出角色不能进入训练集')
    prompt = render_role_prompt(registry, card['id'])
    if record.get('renderer_version') != RENDERER_VERSION or record.get('system_prompt_sha256') != hashlib.sha256(prompt.encode('utf-8')).hexdigest():
        raise ValueError('样本提示词指纹或渲染版本不匹配')
    return encode_sample(record, tokenizer, max_length=max_length, trim_history=False, system_prompt=prompt)


def validate_dialogue_splits(dialogues, registry):
    ids, scenes, hashes = set(), {}, {}
    expanded = {'train': [], 'validation': []}
    dialogue_counts = {'train': Counter(), 'validation': Counter()}
    for dialogue in dialogues:
        rows = expand_dialogue(dialogue, registry)
        if dialogue['id'] in ids:
            raise ValueError('重复源对话 ID')
        ids.add(dialogue['id'])
        split = dialogue['split']
        scene = dialogue['scenario_group']
        if scene in scenes and scenes[scene] != split:
            raise ValueError('同一场景不能跨训练和验证')
        scenes[scene] = split
        signature = content_hash(dialogue['messages'])
        if signature in hashes:
            raise ValueError('重复完整对话；不允许换 role_id 或 ID 后扩增')
        hashes[signature] = split
        expanded[split].extend(rows)
        dialogue_counts[split][dialogue['role_id']] += 1
    if not all(expanded.values()):
        raise ValueError('训练与验证拆分均需有样本')
    return expanded, dialogue_counts


def resolve_run_system_prompt(config, run_dir):
    """Validate the saved conditioning before loading any GPU model."""
    conditioning = config.get('conditioning', {})
    if conditioning.get('mode') != 'per_sample_role_card':
        return conditioning.get('system_prompt')
    root = Path(run_dir).resolve()
    cards = (root / conditioning['role_cards_file']).resolve()
    if not cards.is_relative_to(root):
        raise ValueError('角色卡快照路径超出训练目录')
    if (conditioning.get('renderer_version') != RENDERER_VERSION
            or file_hash(cards) != conditioning.get('role_cards_sha256')):
        raise ValueError('Adapter 的角色卡快照或渲染版本不匹配')
    return render_role_prompt(load_registry(cards), conditioning['verification_role_id'])


def _validate_checkpoints(checkpoints, turn_count, label):
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f'{label} 必须包含预先写定的检查点')
    seen_turns = set()
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, dict) or set(checkpoint) != {
                'after_turn', 'acceptance', 'failure_signals'}:
            raise ValueError(f'{label} 检查点字段无效')
        turn = checkpoint['after_turn']
        if type(turn) is not int or not 1 <= turn <= turn_count or turn in seen_turns:
            raise ValueError(f'{label} 检查点轮次无效或重复')
        seen_turns.add(turn)
        for field in ('acceptance', 'failure_signals'):
            values = checkpoint[field]
            if not isinstance(values, list) or not values:
                raise ValueError(f'{label} 检查点 {field} 必须是非空列表')
            for value in values:
                clean_text(value, field)
    if max(seen_turns) != turn_count:
        raise ValueError(f'{label} 必须覆盖最后一轮')


def _user_trigrams(turns):
    text = ''.join(char.lower() for turn in turns for char in turn
                   if char.isalnum() or '\u4e00' <= char <= '\u9fff')
    return {text[index:index + 3] for index in range(max(0, len(text) - 2))}


def audit_final_evaluation(evaluation_dir, main_registry, training_dialogues,
                           reserved_role_ids=None, protocol_path=None):
    """Validate the frozen holdout cases without loading a model or seeing outputs."""
    root = Path(evaluation_dir)
    final_registry = load_registry(root / 'role-cards.json')
    if final_registry['common_rules'] != main_registry['common_rules']:
        raise ValueError('最终角色评测必须使用与训练相同的共同规则')
    role_index = validate_registry(final_registry)
    role_ids = set(role_index)
    if len(role_ids) != 6 or any(card['partition'] != 'final_only'
                                 for card in role_index.values()):
        raise ValueError('最终评测必须是6个 final_only 角色')
    main_role_ids = {card['id'] for card in validate_registry(main_registry).values()}
    if role_ids & main_role_ids:
        raise ValueError('最终角色与训练/验证角色重复')
    if reserved_role_ids is not None and role_ids != set(reserved_role_ids):
        raise ValueError('最终角色 ID 与冻结清单不一致')

    chat_packet = json.loads((root / 'chat-cases.json').read_text(encoding='utf-8'))
    if (chat_packet.get('version') != 'shared-role-final-chat-v1'
            or not isinstance(chat_packet.get('source'), str) or not chat_packet['source']):
        raise ValueError('最终聊天评测版本或来源说明无效')
    cases = chat_packet.get('cases')
    if not isinstance(cases, list) or len(cases) != 30:
        raise ValueError('最终聊天评测必须包含30个场景')
    categories = Counter()
    role_counts = Counter()
    case_ids = set()
    all_evaluation_turns = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
                'case_id', 'role_id', 'category', 'turns', 'checkpoints'}:
            raise ValueError('最终聊天场景字段无效')
        clean_text(case.get('case_id'), 'case_id')
        if case['case_id'] in case_ids:
            raise ValueError('最终评测出现重复 case_id')
        case_ids.add(case['case_id'])
        if case.get('role_id') not in role_index:
            raise ValueError('最终聊天场景引用了未知角色')
        category = case.get('category')
        if category not in ('daily', 'context', 'persona'):
            raise ValueError('最终聊天场景类别无效')
        turns = case.get('turns')
        if not isinstance(turns, list) or not turns:
            raise ValueError('最终聊天场景缺少用户轮次')
        if ((category in ('daily', 'persona') and len(turns) != 1)
                or (category == 'context' and not 3 <= len(turns) <= 5)):
            raise ValueError('单轮场景或上下文场景轮数不符合冻结方案')
        for turn in turns:
            clean_text(turn, 'user_turn')
        _validate_checkpoints(case.get('checkpoints'), len(turns), case['case_id'])
        categories[category] += 1
        role_counts[case['role_id']] += 1
        all_evaluation_turns.append((case['case_id'], turns))
    if categories != Counter({'daily': 10, 'context': 10, 'persona': 10}):
        raise ValueError('最终聊天类别必须各10题')
    if role_counts != Counter({role_id: 5 for role_id in role_ids}):
        raise ValueError('最终6个角色必须各有5个聊天场景')

    regression_packet = json.loads((root / 'capability-regression.json').read_text(encoding='utf-8'))
    if (regression_packet.get('version') != 'shared-role-capability-regression-v1'
            or not isinstance(regression_packet.get('source'), str)
            or not regression_packet['source']):
        raise ValueError('能力回归版本或来源说明无效')
    regression = regression_packet.get('cases')
    if not isinstance(regression, list) or len(regression) != 20:
        raise ValueError('能力回归必须包含20个案例')
    regression_categories = Counter()
    for case in regression:
        if not isinstance(case, dict) or set(case) != {
                'case_id', 'category', 'turns', 'checkpoints'}:
            raise ValueError('能力回归案例字段无效或混入角色条件')
        clean_text(case.get('case_id'), 'case_id')
        if case['case_id'] in case_ids:
            raise ValueError('聊天和能力回归 case_id 重复')
        case_ids.add(case['case_id'])
        category = case.get('category')
        if category not in ('memory_correction', 'basic_math', 'python',
                            'general_knowledge', 'uncertainty_and_limits'):
            raise ValueError('能力回归类别无效')
        turns = case.get('turns')
        if not isinstance(turns, list) or not 1 <= len(turns) <= 5:
            raise ValueError('能力回归轮次无效')
        for turn in turns:
            clean_text(turn, 'user_turn')
        _validate_checkpoints(case.get('checkpoints'), len(turns), case['case_id'])
        regression_categories[category] += 1
        all_evaluation_turns.append((case['case_id'], turns))
    expected_regression = Counter({'memory_correction': 4, 'basic_math': 4, 'python': 4,
                                   'general_knowledge': 4, 'uncertainty_and_limits': 4})
    if regression_categories != expected_regression:
        raise ValueError('五类能力回归案例必须各4题')

    if protocol_path is not None:
        protocol = json.loads(Path(protocol_path).read_text(encoding='utf-8'))
        expected_generation = dict(thinking=False, seed=42, temperature=0.7,
                                   top_p=0.8, top_k=20, max_new_tokens=256,
                                   input_budget=2048)
        generation = protocol.get('generation', {})
        if (protocol.get('version') != 'shared-role-final-evaluation-protocol-v1'
                or any(generation.get(key) != value for key, value in expected_generation.items())
                or protocol.get('case_counts', {}).get('chat', {}).get('total') != 30
                or protocol.get('case_counts', {}).get('capability_regression', {}).get('total') != 20):
            raise ValueError('最终评测生成设置或数量与预注册方案不符')
        clean_text(protocol.get('neutral_system_prompt'), 'neutral_system_prompt')

    # Compare whole user scripts with SFT dialogues and with each other. This
    # catches surface-level copies; it does not establish semantic independence.
    prior = [(row['id'], [message['content'] for message in row['messages']
                          if message['role'] == 'user']) for row in training_dialogues]
    evaluation = [(case_id, turns) for case_id, turns in all_evaluation_turns]
    flagged = []
    compared = 0
    for index, (left_id, left_turns) in enumerate(evaluation):
        left = _user_trigrams(left_turns)
        for right_id, right_turns in prior:
            right = _user_trigrams(right_turns)
            union = left | right
            score = len(left & right) / len(union) if union else 0.0
            compared += 1
            if score >= 0.75:
                flagged.append(dict(left_id=left_id, right_id=right_id, similarity=round(score, 6)))
        for right_id, right_turns in evaluation[index + 1:]:
            right = _user_trigrams(right_turns)
            union = left | right
            score = len(left & right) / len(union) if union else 0.0
            compared += 1
            if score >= 0.75:
                flagged.append(dict(left_id=left_id, right_id=right_id, similarity=round(score, 6)))
    if flagged:
        raise ValueError('最终评测与训练/验证或彼此存在表面近重复：' + json.dumps(flagged, ensure_ascii=False))
    return dict(final_role_ids=sorted(role_ids), chat_cases=len(cases),
                chat_by_category=dict(sorted(categories.items())),
                chat_by_role=dict(sorted(role_counts.items())),
                capability_cases=len(regression),
                capability_by_category=dict(sorted(regression_categories.items())),
                near_duplicate_comparisons=compared, near_duplicate_flagged=0,
                limitation='Surface trigram screening only; it cannot prove semantic independence or human quality.')


def audit_shared_dataset(data_dir, require_training_ready=True):
    root = Path(data_dir)
    manifest = json.loads((root / 'freeze-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('version') != DATA_VERSION:
        raise ValueError('共享角色数据版本不匹配')
    if require_training_ready and manifest.get('training_ready') is not True:
        raise ValueError('当前仅为 pilot 数据，完整拆分及最终评测尚未冻结，禁止开训')
    required = ('role-cards.json', 'dialogues.json', 'train.jsonl', 'validation.jsonl')
    for name in required:
        if file_hash(root / name) != manifest.get('files', {}).get(name, {}).get('sha256'):
            raise ValueError('共享角色冻结文件指纹不匹配：' + name)
    registry = load_registry(root / 'role-cards.json')
    dialogues = json.loads((root / 'dialogues.json').read_text(encoding='utf-8'))
    expected, counts = validate_dialogue_splits(dialogues, registry)
    for split in expected:
        if read_records(root / f'{split}.jsonl') != expected[split]:
            raise ValueError('源对话、角色卡与展开样本不一致：' + split)
    train_roles = set(counts['train'])
    heldout_validation_roles = {role['id'] for role in registry['roles'] if role['partition'] == 'validation_only'}
    if train_roles & heldout_validation_roles:
        raise ValueError('留出角色泄漏进训练')
    planned_final_roles = set(manifest.get('reserved_final_role_ids', []))
    if planned_final_roles & {role['id'] for role in registry['roles']}:
        raise ValueError('最终角色不能进入训练/验证角色清单')
    if require_training_ready:
        if len(train_roles) != 20 or any(count != 20 for count in counts['train'].values()):
            raise ValueError('完整训练集必须为20个角色各20段对话')
        val_seen = sum(count for role_id, count in counts['validation'].items() if role_id in train_roles)
        val_unseen = sum(count for role_id, count in counts['validation'].items() if role_id in heldout_validation_roles)
        seen_counts = [count for role_id, count in counts['validation'].items() if role_id in train_roles]
        if (val_seen != 25 or val_unseen != 25 or len(heldout_validation_roles) != 5
                or len(seen_counts) != 5 or any(count != 5 for count in seen_counts)
                or any(counts['validation'][role_id] != 5 for role_id in heldout_validation_roles)):
            raise ValueError('完整验证集必须包含25段已见角色与25段留出角色')
        training_preflight = manifest.get('training_preflight', {})
        if (training_preflight.get('file') != 'gpu-preflight.json'
                or not training_preflight.get('sha256')
                or file_hash(root / training_preflight['file']) != training_preflight['sha256']):
            raise ValueError('训练前全量数据/GPU预检记录未登记或已变更')
        preflight_report = json.loads((root / training_preflight['file']).read_text(encoding='utf-8'))
        if (preflight_report.get('status') != 'passed_pretraining_capacity_and_data_checks'
                or preflight_report.get('gpu', {}).get('passed') is not True
                or preflight_report.get('data', {}).get('checked_targets') != 1058
                or manifest.get('training_config') != preflight_report.get('locked_training_config')):
            raise ValueError('训练前数据/GPU预检未通过或参数与冻结记录不一致')
    evaluation_audit = None
    if manifest.get('final_evaluation_frozen'):
        if len(planned_final_roles) != 6:
            raise ValueError('冻结的最终评测必须包含6个新角色')
        # Final evaluation content stays separate; validate both its bytes and
        # its schema before a GPU model can be loaded for training.
        for key in ('final_chat', 'capability_regression'):
            item = manifest.get('evaluation_files', {}).get(key, {})
            if not item.get('path') or not item.get('sha256') or file_hash(root / item['path']) != item['sha256']:
                raise ValueError('最终评测文件指纹未登记或已变更：' + key)
        evaluation_dir = root / 'final-evaluation'
        if file_hash(evaluation_dir / 'role-cards.json') != manifest.get('evaluation_files', {}).get('role_cards', {}).get('sha256'):
            raise ValueError('最终角色卡指纹未登记或已变更')
        if file_hash(evaluation_dir / 'protocol.json') != manifest.get('evaluation_files', {}).get('protocol', {}).get('sha256'):
            raise ValueError('最终评测协议指纹未登记或已变更')
        evaluation_audit = audit_final_evaluation(
            evaluation_dir, registry, dialogues, reserved_role_ids=planned_final_roles,
            protocol_path=evaluation_dir / 'protocol.json')
    elif require_training_ready:
        raise ValueError('最终6个角色及评测仍未冻结')
    return dict(version=DATA_VERSION, training_ready=manifest.get('training_ready') is True,
                counts={name: len(rows) for name, rows in expected.items()},
                source_dialogues={name: sum(per_role.values()) for name, per_role in counts.items()},
                dialogues_by_role={name: dict(per_role) for name, per_role in counts.items()},
                role_cards_sha256=file_hash(root / 'role-cards.json'),
                sha256={name: file_hash(root / f'{name}.jsonl') for name in expected},
                renderer_version=RENDERER_VERSION, data_dir=str(root),
                evaluation=evaluation_audit,
                limitation='Exact dialogue/group/role checks do not prove semantic independence or human quality.')
