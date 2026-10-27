"""从被上下文窗口裁掉的旧助手回答中，按需找回本会话角色自述。"""
import re

_RECALL_CUES = (
    '你刚才说', '你刚刚说', '你之前说', '你以前说', '你说过', '你提过', '你提到过',
    '刚才提到', '刚刚提到', '之前提到', '前面说', '前面提到', '还记得',
    '是什么来着', '叫什么来着', '什么来着', '哪一个来着', '哪个来着',
    '上次说', '刚说的', '之前说的', '你说的那个', '你说的那件', '之前那件事',
    '前面那个', '刚才那个', '那件事后来',
)
_STOP_BIGRAMS = {
    '你说', '说的', '说过', '刚才', '刚刚', '之前', '以前', '提到', '提过',
    '什么', '怎么', '哪个', '哪家', '哪里', '哪儿', '那个', '这个', '一下',
    '来着', '还有', '然后', '是不是', '可以吗', '好吗', '我想', '你能',
    '喜欢', '工作', '周末', '习惯', '生活',
}
_TOPIC_TERMS = (
    '咖啡豆', '拿铁', '美式', '书店', '散步', '路线', '猫咪', '小猫', '猫', '咖啡',
    '家乡', '老家', '生日', '同事', '朋友', '室友', '家人', '上班',
    '早餐', '晚饭', '饮料', '小时候', '旅行', '电影', '音乐', '宠物',
)
_SENTENCE_SPLIT = re.compile(r'(?<=[。！？!?；;\n])')
_CJK_RUN = re.compile(r'[\u3400-\u9fff]+')
_PUNCTUATION = re.compile(r'[\s，。！？、；：,.!?;:"“”‘’（）()【】\[\]…—_-]+')


def _normalise(text):
    return _PUNCTUATION.sub('', text or '').lower()


def _query_terms(query):
    normalised = _normalise(query)
    terms = {term for term in _TOPIC_TERMS if term in normalised}
    for run in _CJK_RUN.findall(normalised):
        for index in range(len(run) - 1):
            term = run[index:index + 2]
            if term not in _STOP_BIGRAMS:
                terms.add(term)
    return terms


def _self_sentences(answer):
    """只使用助手原话，优先选包含小满第一人称的句子，不重写其内容。"""
    sentences = [part.strip() for part in _SENTENCE_SPLIT.split(answer or '') if part.strip()]
    return [sentence for sentence in sentences
            if re.search(r'我(?!们)', sentence) and len(sentence) >= 5]


def build_same_chat_memory(omitted_pairs, query, max_sentences=2, max_chars=420):
    """检索旧裁剪轮次中的小满原话；结果只含助手回答，不含用户内容。"""
    terms = _query_terms(query)
    recall_requested = any(cue in (query or '') for cue in _RECALL_CUES)
    candidates = []
    for pair in omitted_pairs:
        answer = pair.get('assistant', '')
        source_user = _normalise(pair.get('user', ''))
        for sentence in _self_sentences(answer):
            normalised = _normalise(sentence)
            # 原问题用于判断这句助手回答属于哪个话题；只注入助手原话，
            # 所以用户说过的事实不会被写成角色记忆。
            overlap = sum(1 for term in terms if term in normalised or term in source_user)
            if overlap:
                candidates.append((overlap, pair['turn_index'], sentence))
    # 诸如“你刚才说的那个叫什么来着”没有主题词时，取最近一条第一人称原话。
    has_topic = any(term in (query or '') for term in _TOPIC_TERMS)
    if not candidates and recall_requested and not has_topic:
        for pair in reversed(omitted_pairs):
            sentences = _self_sentences(pair.get('assistant', ''))
            if sentences:
                candidates.append((0, pair['turn_index'], sentences[-1]))
                break
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = []
    used = 0
    seen = set()
    for _, turn_index, sentence in candidates:
        if sentence in seen:
            continue
        remaining = max_chars - used
        if remaining < 8:
            break
        clipped = sentence if len(sentence) <= remaining else sentence[:remaining].rstrip() + '…'
        selected.append(dict(turn_index=turn_index, text=clipped))
        seen.add(sentence)
        used += len(clipped)
        if len(selected) >= max_sentences or used >= max_chars:
            break
    return selected


def inject_same_chat_memory(messages, omitted_pairs, query):
    """只为小满提示词追加本会话记忆，并返回可审计的检索指标。"""
    result = [dict(message) for message in messages]
    if not result or result[0].get('role') != 'system' or not result[0].get('content', '').startswith('你是原创AI聊天角色小满'):
        return result, dict(same_chat_memory_retrieved=0, same_chat_memory_source_turns=[],
                            same_chat_memory_chars=0)
    memories = build_same_chat_memory(omitted_pairs, query)
    if not memories:
        return result, dict(same_chat_memory_retrieved=0, same_chat_memory_source_turns=[],
                            same_chat_memory_chars=0)
    lines = [f"- 第{item['turn_index']}轮小满原话：{item['text']}" for item in memories]
    addition = (
        '\n\n【本次聊天中较早轮次的小满原话】\n'
        '以下内容来自本会话中小满此前的助手回答，供你接续同一段聊天。只把其中明确以第一人称说出的生活经历、习惯或偏好视为小满自述；'
        '不要把用户说过的话当成小满经历。若内容是玩笑、假设或转述，不要当作真实自述。沿用已有细节，避免前后矛盾；'
        '这是本会话的临时上下文，新会话不会收到这些内容。\n' + '\n'.join(lines)
    )
    result[0]['content'] += addition
    return result, dict(same_chat_memory_retrieved=len(memories),
                        same_chat_memory_source_turns=[item['turn_index'] for item in memories],
                        same_chat_memory_chars=sum(len(item['text']) for item in memories))
