"""Small, auditable retrieval for confirmed static character facts.

The Xiao Man identity in ``characters.py`` remains the source of truth. This
module creates a compact lexical index from it rather than copying character
facts into a second hand-maintained profile or introducing a vector database.
"""

from characters import CHARACTERS, conversation_messages


_UNCONFIRMED = {
    '生日': [('生日',), ('出生日期',)],
    '学历': [('学历',), ('毕业',), ('大学',), ('学校',)],
    '家庭成员': [('家人',), ('父母',), ('家庭',), ('兄弟姐妹',)],
    '家乡': [('家乡',), ('老家',), ('哪里人',), ('哪儿人',), ('本地人',), ('从哪里来',), ('从哪儿来',)],
    '书店名称': [('书店', '名字'), ('书店', '名称'), ('书店', '叫什么'), ('书店', '叫啥'), ('书店', '店名')],
    '是否养猫及猫的名字': [('养猫',), ('有猫吗',), ('猫叫什么',), ('猫的名字',), ('猫咪叫什么',)],
    '具体住址': [('住址',), ('具体住哪',), ('家庭住址',), ('详细地址',)],
    '详细成长经历': [('成长经历',), ('哪里长大',), ('哪儿长大',), ('小时候',)],
}

_EXPERIENCE_TERMS = [
    ('客人',), ('顾客',), ('印象深刻',), ('遇到过',), ('工作经历',), ('工作往事',),
    ('书店', '故事'), ('以前', '经历'), ('发生过什么',),
]


def _record(record_id, category, status, text, match_any, source):
    return {
        'id': record_id,
        'category': category,
        'status': status,
        'text': text,
        'match_any': [list(group) for group in match_any],
        'source': source,
    }


def build_role_knowledge(character_id='companion'):
    """Build stable, traceable retrieval records from the canonical profile."""
    role = CHARACTERS.get(character_id)
    if role is None or not role.get('identity'):
        raise ValueError('角色没有可检索的固定身份资料：' + str(character_id))
    identity = role['identity']
    source = 'characters.py:CHARACTERS.' + character_id + '.identity'
    records = [
        _record('identity.name', 'identity', 'confirmed', '姓名：' + identity['name'] + '。',
                [('你叫什么',), ('你叫啥',), ('你的名字',), ('名字是什么',), ('姓名',), ('自我介绍',), ('介绍一下你',), ('你是谁',)], source),
        _record('identity.gender', 'identity', 'confirmed', '性别：' + identity['gender'] + '。',
                [('性别',), ('你是女生',), ('你是女孩子',), ('你是男生',), ('你是男性',), ('你是女性',), ('你是爷们',)], source),
        _record('identity.age', 'identity', 'confirmed', '年龄：' + identity['age'] + '。',
                [('你多大',), ('你几岁',), ('你的年龄',), ('你年龄',), ('你岁数',),
                 ('今年多大',), ('今年几岁',), (identity['age'],)], source),
        _record('identity.city', 'identity', 'confirmed', '所在地：' + identity['city'] + '。',
                [('你在哪个城市',), ('你在哪座城市',), ('你现在在哪',), ('你住在哪',), ('你的所在地',), ('你住哪里',),
                 ('在' + identity['city'],), (identity['city'],)], source),
        _record('identity.occupation', 'identity', 'confirmed', '职业：' + identity['occupation'] + '。',
                [('你做什么工作',), ('你是做什么的',), ('你的职业',), ('你在哪工作',), ('你在哪上班',), ('你的工作',),
                 ('你工作',), ('你在书店工作',), ('店员',), ('书店上班',), ('书店工作',)], source),
        _record('personality.traits', 'personality', 'confirmed',
                '性格：' + '、'.join(identity['traits']) + '。',
                [('性格',), ('温柔',), ('直率',)], source),
        _record('interest.coffee', 'interest', 'confirmed', '兴趣之一：咖啡。',
                [('你喜欢咖啡',), ('你也喜欢咖啡',), ('喜欢喝咖啡',), ('爱喝咖啡',),
                 ('你喜欢喝什么',), ('你的兴趣爱好',), ('你有什么爱好',), ('你有哪些爱好',), ('你有什么兴趣',)], source),
        _record('interest.walking', 'interest', 'confirmed', '兴趣之一：散步。',
                [('你喜欢散步',), ('你爱散步',), ('散步',), ('你的兴趣爱好',), ('你有什么爱好',),
                 ('你有哪些爱好',), ('你有什么兴趣',)], source),
        _record('interest.cats', 'interest', 'confirmed', '兴趣之一：猫。',
                [('你喜欢猫',), ('你喜欢猫咪',), ('你喜欢小猫',), ('你爱猫',),
                 ('你的兴趣爱好',), ('你有什么爱好',), ('你有哪些爱好',), ('你有什么兴趣',)], source),
        _record('relationship.boundary', 'relationship', 'confirmed', identity['relationship'],
                [('我们的关系',), ('我们是什么关系',), ('我们算是朋友',), ('线下见过',), ('现实中认识',)], source),
    ]

    label_to_id = {
        '生日': 'unknown.birthday', '学历': 'unknown.education', '家庭成员': 'unknown.family',
        '家乡': 'unknown.hometown', '书店名称': 'unknown.bookstore_name',
        '是否养猫及猫的名字': 'unknown.cat_ownership_or_name', '具体住址': 'unknown.address',
        '详细成长经历': 'unknown.life_history',
    }
    for label in identity.get('unconfirmed', []):
        match_any = _UNCONFIRMED.get(label)
        if match_any:
            records.append(_record(label_to_id[label], 'unknown', 'unconfirmed', label + '尚未设定。',
                                   match_any, source))
    records.append(_record('unknown.work_experiences', 'unknown', 'unconfirmed',
                           '没有确认过具体的书店顾客故事或工作往事。', _EXPERIENCE_TERMS,
                           'characters.py:companion.prompt.no_unconfirmed_experiences'))
    return records


def _matches(query, match_any):
    normalized = ''.join(str(query).lower().split())
    return any(all(term.lower() in normalized for term in group) for group in match_any)


def retrieve_role_facts(query, character_id='companion', limit=6):
    """Return deterministic topically matched records with explicit provenance."""
    if not isinstance(query, str) or not query.strip():
        return []
    if type(limit) is not int or limit < 1:
        raise ValueError('limit 必须是正整数')
    records = build_role_knowledge(character_id)
    normalized = ''.join(query.lower().split())
    if any(term in normalized for term in ('角色设定', '角色背景', '介绍一下你', '介绍一下自己', '自我介绍')):
        return records[:limit]
    return [record for record in records if _matches(query, record['match_any'])][:limit]


def build_retrieval_messages(history, user_text, character_id='companion'):
    """Create B-condition messages: behavior prompt plus only relevant facts."""
    role = CHARACTERS.get(character_id)
    if role is None or not role.get('identity') or not role.get('prompt'):
        raise ValueError('角色没有可用的提示词与固定身份资料：' + str(character_id))
    prompt = role['prompt']
    prompt = prompt.replace('固定身份档案会在每轮对话中提供', '角色资料会根据当前问题从资料库中检索后提供')
    prompt = prompt.replace('只引用档案里明确列出的事实', '只引用本轮检索到的已确认资料')
    facts = retrieve_role_facts(user_text, character_id=character_id)
    if facts:
        details = '\n'.join('- [' + row['status'] + '] ' + row['text'] for row in facts)
        retrieval = '【本轮检索到的角色资料】\n' + details
    else:
        retrieval = '【本轮检索到的角色资料】\n没有检索到与当前问题直接相关的角色资料。不要因此补造角色身份细节或个人经历。'
    messages = [dict(role='system', content=prompt + '\n\n' + retrieval)]
    messages.extend(dict(role=row['role'], content=row['content']) for row in history)
    messages.append(dict(role='user', content=user_text))
    return messages, facts


def build_additive_retrieval_messages(history, user_text, character_id='companion'):
    """Create B2 messages by preserving A's prompt and appending matched facts.

    For a query with no retrieved facts, the returned message list is exactly
    the same as the A-stage role-card input. This keeps retrieval as the only
    changed factor in the targeted B2 comparison.
    """
    current = dict(character_id=character_id, messages=history)
    messages = conversation_messages(current, user_text)
    facts = retrieve_role_facts(user_text, character_id=character_id)
    if facts:
        details = '\n'.join('- [' + row['status'] + '] ' + row['text'] for row in facts)
        addition = '\n\n【本轮补充的角色资料】\n' + details
        messages[0] = dict(messages[0], content=messages[0]['content'] + addition)
    return messages, facts
