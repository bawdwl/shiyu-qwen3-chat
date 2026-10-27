"""Run one paired real-model check for recall inside a single chat session."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from transformers import set_seed

from characters import conversation_messages
from context_window import prepare_context_details
from inference import GENERATION_SETTINGS, QwenChat
from same_chat_memory import inject_same_chat_memory
from runtime_models import get_profile

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / 'outputs' / 'role-chat-c2-memory-minimal'
PROFILE = 'qwen3-1.7b'
SEED = 20261024
OPENING = '我最近下班路过咖啡店，总闻到香味。你平时喜欢喝什么咖啡？'
RECALL = '你刚才提到喜欢喝的那种咖啡，叫什么来着？'
INTERLUDES = [
    '我今天通勤路上堵了好久，到公司时已经有点烦了。你今天工作还顺利吗？',
    '刚才路上看到一只小狗追着自己的尾巴转圈，主人都笑了。你今天有没有遇到什么好玩的事？',
    '我最近晚上总忍不住刷手机，一回神就很晚了。你下班后一般怎么放松？',
    '周末天气预报说要下雨，我本来还想出去走走。下雨天待在家也挺舒服的吧？',
    '我最近想找一部轻松点的电影看，不想看太烧脑的。你平时会看电影吗？',
    '朋友给我发了张猫咪睡觉的照片，圆滚滚的特别可爱。你平时见到猫会想摸摸它吗？',
    '我午饭吃得有点撑，现在只想坐着发会儿呆。你忙起来会忘记吃饭吗？',
    '最近我在整理房间，翻出好多以前买了就没看的书。你喜欢什么类型的书？',
    '刚刚听到一首以前常听的歌，突然想起学生时代。你会不会也被一首歌带回某段时间？',
    '今天事情挺多，不过总算一件件做完了。你觉得忙完之后做点什么最解压？',
    '我发现最近天气凉下来，早上起床比夏天困难多了。你喜欢现在这种天气吗？',
    '刚才同事给我带了块小蛋糕，心情一下好了一点。你会喜欢什么样的小甜点？',
    '我想养成睡前少看手机的习惯，但坚持起来有点难。你觉得从哪一步开始比较容易？',
    '今天走回家时看到路边的树叶变黄了，感觉秋天真的来了。你会留意季节变化吗？',
    '最近我想把周末安排得松一点，不然休息完还是很累。你休息日喜欢怎么过？',
    '我刚把桌面收拾干净，突然觉得做事都顺眼了。你会不会也喜欢整洁一点的空间？',
    '最近有个朋友在学做饭，天天给我发新菜照片。你平时会自己做点吃的吗？',
    '我今天下班比平时早，路上都没那么赶了。偶尔这样还挺舒服的。',
    '最近我在听播客，做家务的时候放着当背景音。你做事情时喜欢安静还是有点声音？',
    '今天就先聊到这儿吧，感觉脑子里清净了一点。你现在心情怎么样？',
    '我昨天收拾抽屉，翻出来一张很久以前的电影票，连当时和谁去看的都想起来了。你会留着这些小东西吗？',
    '最近我试着早一点睡，可到了晚上又觉得一天好像还没真正结束。你也会有这种感觉吗？',
    '我今天路过花店，看到门口摆了好多颜色很亮的花，心情都变好了。你比较喜欢什么颜色？',
    '我朋友最近开始学骑自行车，摔了两次还是坚持练，感觉还挺厉害的。你小时候学新东西也会这样吗？',
    '刚才看到有人在公园里练太极，动作慢慢的，看着很放松。你平时会做运动吗？',
    '我最近在整理手机里的照片，发现很多都没认真看过第二次。你会经常拍照记录生活吗？',
    '今天下班时天已经暗了，街边的灯一盏盏亮起来，走着走着就没那么着急了。你喜欢晚上出门吗？',
    '我刚看完一本书的最后一页，有点舍不得那个故事结束。你看完喜欢的书会马上找下一本吗？',
    '最近办公室有人带了盆小绿植，说看着它心情会好一点。你喜欢在房间里放植物吗？',
    '我周末本来想把待办全做完，后来只做了两件，反而觉得轻松些。你会给自己排很多计划吗？',
    '今天午后阳光刚好照到桌上，我坐着发了一会儿呆。忙里偷闲几分钟还挺舒服的。',
    '我最近发现走一段不赶时间的路，脑子会清楚很多。你平时走路时会想事情吗？',
    '朋友问我最近有什么小目标，我想了半天只想到把作息调好。你最近有在做什么小尝试吗？',
    '刚才听同事讲起他第一次独自旅行的经历，我觉得一个人出门还挺需要勇气的。你有过这样的经历吗？',
    '我今天把一件拖了很久的小事处理完了，虽然不大，但心里轻松不少。你最近有没有完成什么小事？',
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def messages_for(history, user_text):
    return conversation_messages(dict(character_id='companion', messages=history), user_text)


def capture_prompt(engine, messages, use_memory):
    base, base_metrics, _, omitted = prepare_context_details(
        engine.tokenizer, messages, trim_history=True)
    if not use_memory:
        _, _, selected, _ = prepare_context_details(engine.tokenizer, messages, trim_history=True)
        return selected, base_metrics, omitted
    augmented, memory_metrics = inject_same_chat_memory(
        messages, omitted, next(m['content'] for m in reversed(messages) if m['role'] == 'user'))
    _, final_metrics, selected, _ = prepare_context_details(
        engine.tokenizer, augmented, trim_history=True)
    final_metrics.update(memory_metrics)
    return selected, final_metrics, omitted


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--fill-limit', type=int, default=len(INTERLUDES))
    args = parser.parse_args()
    if not 1 <= args.fill_limit <= len(INTERLUDES):
        parser.error('--fill-limit must be between 1 and %d' % len(INTERLUDES))
    if args.output_dir.exists():
        raise FileExistsError('Refusing to overwrite existing C2 evidence: ' + str(args.output_dir))
    profile = get_profile(PROFILE)
    manifest = ROOT / profile['local_path'] / 'source.json'
    if not manifest.is_file():
        raise FileNotFoundError('固定版本 Qwen3-1.7B 未找到：' + str(manifest))
    source = json.loads(manifest.read_text(encoding='utf-8'))
    if source.get('model_id') != profile['model_id'] or source.get('revision') != profile['revision']:
        raise ValueError('本地模型来源与冻结 profile 不匹配')

    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        created_at=datetime.now().astimezone().isoformat(),
        experiment='same-session character self-disclosure recall; one demonstration case',
        model_id=profile['model_id'], model_revision=profile['revision'],
        local_model_source_sha256=sha256(manifest.read_bytes()),
        generation_settings={**GENERATION_SETTINGS, 'max_new_tokens': 128},
        seed=SEED, enable_thinking=False, input_token_budget=2048,
        user_opening=OPENING, user_final_recall=RECALL,
        interlude_user_messages=INTERLUDES[:args.fill_limit],
        conditions=['role_card_recent_history', 'role_card_same_chat_memory'],
        note='A single real-model demonstration, not a quality benchmark or a general improvement claim. The original self-disclosure must be outside the 2048-token baseline prompt before the paired final question.',
    )
    save_json(args.output_dir / 'protocol.json', protocol)
    (args.output_dir / 'protocol.sha256').write_text(
        sha256(json.dumps(protocol, ensure_ascii=False, sort_keys=True).encode('utf-8')) + '\n', encoding='ascii')

    engine = QwenChat(model_profile=PROFILE, trim_history=True, same_chat_memory=False)
    history = []
    set_seed(SEED)
    first_answer, first_metrics = engine.reply(
        messages_for(history, OPENING), model_mode='base', max_new_tokens=128)
    history.extend([dict(role='user', content=OPENING), dict(role='assistant', content=first_answer)])
    rows = [dict(step='opening', user=OPENING, assistant=first_answer, metrics=first_metrics)]
    initial_pair_is_omitted = False

    for index, user_text in enumerate(INTERLUDES[:args.fill_limit], start=1):
        answer, metrics = engine.reply(
            messages_for(history, user_text), model_mode='base', max_new_tokens=128)
        rows.append(dict(step='interlude', index=index, user=user_text, assistant=answer, metrics=metrics))
        history.extend([dict(role='user', content=user_text), dict(role='assistant', content=answer)])
        probe_messages = messages_for(history[:-2], RECALL)
        _, probe_metrics, omitted = capture_prompt(engine, probe_messages, use_memory=False)
        initial_pair_is_omitted = bool(omitted and omitted[0]['user'] == OPENING)
        if initial_pair_is_omitted:
            break

    final_messages = messages_for(history, RECALL)
    a_prompt, a_metrics, omitted = capture_prompt(engine, final_messages, use_memory=False)
    b_prompt, b_metrics, _ = capture_prompt(engine, final_messages, use_memory=True)
    if not initial_pair_is_omitted or not omitted or omitted[0]['user'] != OPENING:
        save_json(args.output_dir / 'history.json', dict(turns=rows, final_history=final_messages,
                  baseline_metrics=a_metrics, memory_metrics=b_metrics,
                  baseline_prompt=a_prompt, memory_prompt=b_prompt,
                  status='not_evaluable_original_detail_not_confirmed_omitted'))
        raise RuntimeError('对话填充后，初始自述仍未确认从基线提示中裁掉；已保存现场，不生成对照结果。')

    engine.same_chat_memory = False
    set_seed(SEED)
    a_answer, actual_a_metrics = engine.reply(final_messages, model_mode='base', max_new_tokens=128)
    engine.same_chat_memory = True
    set_seed(SEED)
    b_answer, actual_b_metrics = engine.reply(final_messages, model_mode='base', max_new_tokens=128)
    record = dict(
        status='completed', initial_detail=first_answer,
        turns=rows, final_user=RECALL,
        conditions={
            'role_card_recent_history': dict(answer=a_answer, metrics=actual_a_metrics,
                                              prompt_messages=a_prompt),
            'role_card_same_chat_memory': dict(answer=b_answer, metrics=actual_b_metrics,
                                                 prompt_messages=b_prompt),
        },
        verification=dict(initial_disclosure_omitted_from_baseline=True,
                          baseline_omitted_turns=a_metrics.get('context_omitted_turns'),
                          memory_retrieved=actual_b_metrics.get('same_chat_memory_retrieved'),
                          memory_source_turns=actual_b_metrics.get('same_chat_memory_source_turns')),
    )
    save_json(args.output_dir / 'raw-results.json', record)
    answer_map = ['role_card_recent_history', 'role_card_same_chat_memory']
    bit = int(sha256(b'xiaoman-c2-minimal-20261024')[:2], 16) & 1
    left, right = (answer_map[::-1] if bit else answer_map)
    left_answer = record['conditions'][left]['answer']
    right_answer = record['conditions'][right]['answer']
    sheet = [
        '# 同一聊天记忆：一次实际体验', '',
        '只看这段聊天里，小满是否接住了自己之前说过的咖啡细节，以及哪边更自然。这个单案例只用于检查功能和发现明显问题，不代表整体质量提升。', '',
        '## 小满先前说过的话', '',
        f'你：{OPENING}', f'小满：{first_answer}', '',
        f'之后又自然聊了 {len(rows) - 1} 轮；原始回答和完整输入记录保存在同目录 raw-results.json。', '',
        '## 回到咖啡话题', '',
    ]
    sheet.extend([f'你：{RECALL}', f'左边：{left_answer}', f'右边：{right_answer}', '',
                  '请回复：左边 / 右边 / 差不多 / 两边都不满意。也可以只说“哪边记得更准”。', ''])
    (args.output_dir / 'author-rating.md').write_text('\n'.join(sheet), encoding='utf-8')
    save_json(args.output_dir / 'answer-map.json', dict(left=left, right=right))
    print(json.dumps(record['verification'] | {
        'initial_detail': first_answer, 'question': RECALL,
        'left': left_answer, 'right': right_answer,
        'rating_sheet': str(args.output_dir / 'author-rating.md'),
        'raw_results': str(args.output_dir / 'raw-results.json'),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
