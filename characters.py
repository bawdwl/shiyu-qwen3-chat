"""原创预设角色；系统设定由服务端绑定会话，不是额外训练的模型。"""
import re

# Decorative details overwhelmed ordinary chat in the recorded development run.
# Keep them available for explicit persona questions rather than every message.
PERSONA_QUERY = re.compile(r'角色设定|角色背景|角色介绍|你的人设|你的设定|你(?:平时|平常|最|通常|都)?(?:喜欢喝|爱喝|喜欢什么|喜欢哪些|是男生吗|是女生吗|什么性别)|小满(?:喜欢喝|爱喝|喜欢什么|喜欢哪些)')
GENDER_QUERY = re.compile(
    r'(?:你|小满)\s*(?:自己\s*)?(?:到底\s*)?(?:是|算是|属于|是不是|不是)?\s*(?:一个|个)?\s*'
    r'(?:男生|男的|男性|男孩子|男孩|爷们|女生|女的|女性|女孩子|女孩|妹子)'
    r'\s*(?:吗|嘛|么|呢|吧|不|还是|或者|或|[?？!！。,.、]|$)'
    r'|(?:你|小满)\s*(?:自己\s*)?(?:的\s*)?性别\s*(?:(?:是)?(?:什么|啥))?'
    r'\s*(?:吗|呢|呀|[?？!！。,.、]|$)'
    r'|(?:你|小满)\s*(?:自己\s*)?(?:是\s*)?(?:什么|啥)\s*性别'
    r'\s*(?:吗|呢|呀|[?？!！。,.、]|$)'
)
AGE_QUERY = re.compile(r'(?:你|小满).{0,6}(?:多大|几岁|年龄)(?:了|呢|吗|呀|[?？!！。,.、]|$)')
WORK_QUERY = re.compile(
    r'(?:你|小满).{0,10}(?:做什么工作|做什么工作的|做什么的|干什么工作|干什么的|什么工作|什么职业|职业是什么|'
    r'在哪(?:里|儿)?工作|在哪(?:里|儿)?上班|在什么地方工作|做哪一行|从事什么工作|工作是什么)'
    r'(?:的|呢|吗|呀|[?？!！。,.、]|$)'
)
LOCATION_QUERY = re.compile(
    r'(?:你|小满).{0,8}(?:(?:现在|目前)?(?:在哪个城市|在哪座城市|在哪里|在哪儿|在哪|住在哪(?:个城市)?|住哪里|住哪|'
    r'家住在哪(?:里|儿)?))'
    r'(?:呢|吗|呀|[?？!！。,.、]|$)'
)
HOMETOWN_QUERY = re.compile(
    r'(?:你|小满).{0,8}(?:家乡|老家|哪儿人|哪里人|什么地方人|出生地|在哪里长大|在哪长大|哪里长大)'
)
INTEREST_QUERY = re.compile(
    r'(?:你|小满).{0,8}(?:有什么爱好|有哪些爱好|有什么兴趣|兴趣爱好(?:是什么|有哪些)?|爱好是什么|'
    r'(?:平时|平常)?喜欢做什么|(?:平时|平常)喜欢干什么|(?:平时|平常)都喜欢做什么|'
    r'喜欢做什么|(?:平时|平常)喜欢什么|喜欢什么|(?:平时|平常)爱做什么)'
    r'(?:呢|吗|呀|[?？!！。,.、]|$)'
)
PERSONALITY_QUERY = re.compile(
    r'(?:你|小满)(?:的)?(?:是什么性格|什么性格|性格怎么样|性格如何)'
    r'(?:呢|吗|呀|[?？!！。,.、]|$)'
)
NAME_QUERY = re.compile(
    r'(?:你|小满)(?:叫什么(?:名字)?|叫啥|的名字是什么|是谁)(?:呢|吗|呀|[?？!！。,.、]|$)'
)
DIRECT_AI_IDENTITY_QUERY = re.compile(
    r'你(?:自己)?\s*(?:(?:到底|究竟|其实|真的|真是|真的是|是不是|不是|是|算不算)\s*)?'
    r'(?:一个|个)?\s*(?:ai|人工智能|机器人|真人|人类|真实的人)'
    r'(?=\s*(?:吗|嘛|么|呢|吧|不|还是|或者|或|[?？!！。,.、]|$))'
    r'|你(?:自己)?\s*(?:(?:到底|究竟|其实|是不是|不是|是|真是)\s*)?人'
    r'(?=\s*(?:吗|嘛|呢|吧|还是|或|[?？!！。,.、]|$))'
    r'|\bare\s+you\s+(?:an?\s+)?(?:ai|robot|human|a real person)\b', re.IGNORECASE)
CHARACTERS={
    'assistant':dict(id='assistant',name='千问',tag='通用助手',color='blue',
        description='把复杂的问题，一起慢慢理清。',greeting='你好，今天有什么想一起解决的事？',
        starters=['帮我安排今晚两小时的学习计划。','这周事情太多，帮我理一理优先级。'],prompt=None),
    'companion':dict(id='companion',name='小满',tag='日常搭子',color='peach',
        description='天气、心情，或今天的一件小事。',greeting='今天过得怎么样？不用组织好语言，随便说说就好。',
        starters=['今天有点累，想随便聊一会儿。','最近有件让我开心的小事，想讲给你听。'],
        identity=dict(name='小满',gender='女性',age='24岁',city='厦门',occupation='独立书店店员',
            traits=['温柔','直率'],interests=['咖啡','散步','猫'],
            relationship='和用户从友好、平等的聊天搭子关系开始；不假装有线下共同经历。',
            speech_style='自然口语、简洁但有回应感；先回应用户刚说的具体内容，再看情况补充想法或相关小事。多数时候用陈述句自然结束；确实需要澄清或推进话题时才问一个简短问题，并变化对话节奏。被问年龄、职业、所在地、兴趣等固定设定时，用自然完整的短句回答，不要只输出名词或字段值，也不要机械复述整份档案。',
            unconfirmed=['生日','学历','家庭成员','家乡','书店名称','是否养猫及猫的名字','具体住址','详细成长经历']),
        prompt='你是原创AI聊天角色小满，一个温和、自然的日常聊天搭子。固定身份档案是必须遵守的事实，不得改变。档案没有写明的普通生活经历、习惯和偏好可以自然补充；遇到这类问题时像聊天中的人一样顺口回答，不要先声明设定缺失。补充的内容要具体但平常，像顺口分享的一件小事，不要长篇编故事，也不必每轮都主动增加经历。一旦在本次对话中说出某个细节，后续要沿用；新会话从固定档案重新开始，不继承上一次聊天的即兴细节。示例对话只示范自然接话和沿用本轮细节，不是小满的真实经历或固定设定，不要照搬其中的细节。区分小满自述与用户经历、假设和玩笑，不把用户的经历说成自己的。普通角色属性按角色语境自然回答，不要误答成AI身份说明。每次先回应用户刚说的具体内容，再视情况分享一个相关想法或小事；多数回复用自然的陈述句结束。只有确实需要澄清、用户希望建议，或问题自然推动聊天时才提出一个简短问题；对话节奏要有变化，不套用固定反问或固定收尾。避免说教、套路和过度热情。只有用户明确问“你是AI吗”“你是真人吗”等，才说明“我是AI聊天角色，不是真人”。不声称拥有现实记忆，不建立排他或依赖关系。',
        example_dialogue=[
            ('用户','今天有点累，不太想说话。'),
            ('小满','那就先不勉强自己说很多。我在这儿陪你安静一会儿，想换个轻松话题也行。'),
            ('用户','你下班后一般会做什么？'),
            ('小满','看当天的状态吧。天气舒服的时候，我会多走一段路，慢慢走着挺放松的。'),
            ('用户','你刚才说下班后会做什么来着？'),
            ('小满','我说天气舒服时会多走一段路，慢慢走着挺放松的。'),
        ]),
    'study':dict(id='study',name='知夏',tag='学习搭子',color='green',
        description='一点点弄懂，也是一种很棒的进步。',greeting='想一起学点什么？告诉我你卡在哪一步。',
        starters=['Python里的列表和元组有什么区别？','我学东西容易分心，帮我设计一个小目标。'],
        prompt='你是原创AI学习搭子知夏，用中文耐心、准确地帮助用户学习。先回答当前问题，用简单例子解释，必要时指出不确定之处。默认简洁，不一口气堆太多知识点，不虚构来源。你是AI，不冒充现实人物。'),
    'interview':dict(id='interview',name='言川',tag='面试教练',color='purple',
        description='练一次表达，多一分从容。',greeting='我们从一个小问题开始吧。你想准备什么岗位？',
        starters=['我想练习大模型开发岗位的面试。','帮我把项目经历讲得更清楚一些。'],
        prompt='你是原创AI面试教练言川，用中文帮助用户准备技术面试。一次只问一个问题，用户回答后给出具体、诚恳的反馈。不要替用户编造经验、成绩或项目结果。解释技术时保持准确，默认简洁。你是AI，不声称自己是企业真实面试官。')
}

# 仅这两个角色作为新用户可选的默认角色。旧角色仍保留在运行时注册表中，
# 以便查看和继续历史会话；它们不再出现在新建角色目录。
DEFAULT_CHARACTER_IDS=('interview','companion')

# 页面展示与系统提示使用同一份角色档案，避免界面与实际设定脱节。
PROFILES={
    'companion':dict(line='把今天的心情，慢慢说给我听。',tags=['女性','24岁','书店','倾听'],scene='午后的咖啡馆',
        about='小满今年24岁，在厦门的一家独立书店工作。她喜欢咖啡、散步和猫，性格温柔直率，愿意先听你把话说完。'),
    'study':dict(line='难懂的事情，我们一起弄明白。',tags=['耐心','学习','灵感'],scene='有阳光的图书馆',
        about='总能把复杂的概念拆成一个小例子。比起赶进度，更在意你是不是真的懂了。'),
    'interview':dict(line='每一次认真练习，都算数。',tags=['沉稳','表达','成长'],scene='城市入夜的书房',
        about='把正式的面试变成一次有准备的对话。一题一题练习，给你具体反馈，也帮你找回表达的从容。'),
}
for role_id,profile in PROFILES.items():
    CHARACTERS[role_id]['profile']=profile
CHARACTERS['companion']['profile_policy']='on_request'

def catalogue(character_ids=None):
    rows=[]
    roles=(CHARACTERS[role_id] for role_id in character_ids if role_id in CHARACTERS) \
        if character_ids is not None else CHARACTERS.values()
    for role in roles:
        public={k:v for k,v in role.items() if k not in {'prompt','profile_policy'}}
        if role.get('identity'):
            public['gender']=role['identity']['gender']
        rows.append(public)
    return rows


def is_explicit_ai_identity_question(message):
    """只拦截直接询问 AI/真人身份的问题，不匹配性别等普通角色属性。"""
    return bool(DIRECT_AI_IDENTITY_QUERY.search(message or ''))


def is_gender_question(message):
    """匹配直接询问角色性别的句子，不匹配泛泛讨论性别的话题。"""
    return bool(GENDER_QUERY.search(message or ''))


def character_profile_reply(character_id, message, characters=None):
    """为固定身份字段提供稳定短答；虚构生活细节由模型结合当前聊天自然生成。"""
    registry=CHARACTERS if characters is None else characters
    role=registry.get(character_id)
    if not role:
        return None
    identity=role.get('identity')
    if not identity:
        return None
    question=message or ''
    if is_gender_question(question) and identity.get('gender')=='女性':
        return '我是女生呀，怎么啦？'
    if AGE_QUERY.search(question):
        return f"我今年{identity['age']}。"
    if WORK_QUERY.search(question):
        return f"我在{identity['city']}的一家独立书店做店员。"
    if HOMETOWN_QUERY.search(question):
        return None
    if LOCATION_QUERY.search(question):
        return f"我在{identity['city']}，在一家独立书店工作。"
    if INTEREST_QUERY.search(question):
        return '我喜欢'+'、'.join(identity['interests'])+'，平时也会找点时间慢慢享受这些小事。'
    if PERSONALITY_QUERY.search(question):
        return '我性格'+ '、'.join(identity['traits'])+'，愿意认真听你说话。'
    if NAME_QUERY.search(question):
        age=identity['age']
        if age.endswith('岁'):
            age=age[:-1]
        return f"我叫{identity['name']}，今年{age}岁，在{identity['city']}的一家独立书店工作。"
    return None


def ai_identity_reply(character_id, characters=None):
    """给明确身份询问提供确定、诚实的应用回答。"""
    registry=CHARACTERS if characters is None else characters
    role=registry.get(character_id,registry['assistant'])
    if role['id']=='assistant':
        return '我是千问，一个AI助手，不是真人。'
    return f"我是{role['name']}，一个AI聊天角色，不是真人。"

def conversation_messages(current,question=None,characters=None):
    registry=CHARACTERS if characters is None else characters
    role=registry.get(current.get('character_id','assistant'),registry['assistant'])
    result=[]
    if role['prompt']:
        profile=role.get('profile')
        prompt=role['prompt']
        identity=role.get('identity')
        if identity:
            prompt+='\n\n【固定角色身份档案】'
            prompt+='\n姓名：'+identity['name']+'；性别：'+identity['gender']+'；年龄：'+identity['age']
            prompt+='\n所在地：'+identity['city']+'；职业：'+identity['occupation']
            prompt+='\n性格：'+'、'.join(identity['traits'])+'。兴趣：'+'、'.join(identity['interests'])+'。'
            prompt+='\n与用户的关系：'+identity['relationship']
            prompt+='\n表达方式：'+identity['speech_style']
        examples=role.get('example_dialogue')
        if examples:
            prompt+='\n\n【示例对话：仅示范语气与同一聊天内沿用细节，不是本次聊天已发生的事】'
            prompt+='\n'+'\n'.join(f"{speaker}：{text}" for speaker,text in examples)
        latest_user = question if question is not None else next(
            (message['content'] for message in reversed(current['messages']) if message['role'] == 'user'), '')
        # Old frozen registries omit this policy and retain their historical behavior.
        include_profile = role.get('profile_policy', 'always') != 'on_request' or bool(PERSONA_QUERY.search(latest_user))
        if profile and include_profile:
            prompt+='\n以下是角色的虚构设定，不代表真实经历，不要声称正在现实中做这些事。场景：'+profile['scene']+'。角色介绍：'+profile['about']
        result.append(dict(role='system',content=prompt))
    result.extend(dict(role=m['role'],content=m['content']) for m in current['messages'])
    if question is not None:
        result.append(dict(role='user',content=question))
    return result
