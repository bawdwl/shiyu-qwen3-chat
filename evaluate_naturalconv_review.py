"""Generate a blind Base-vs-LoRA packet from frozen NaturalConv contexts.

The corpus and the derived packet are restricted to local non-commercial research/testing.
"""
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

from transformers import set_seed

from characters import conversation_messages
from inference import GENERATION_SETTINGS, QwenChat
from training_data import audit_naturalconv


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def build_packet(cases, results):
    if len(results) != len(cases) * 2:
        raise ValueError('每个场景必须恰有两个模型回答')
    grouped = {case['id']: {} for case in cases}
    for row in results:
        if row['case_id'] not in grouped or row['label'] not in ('A', 'B'):
            raise ValueError('未知题号或答案标签')
        if row['label'] in grouped[row['case_id']] or not row['answer'].strip():
            raise ValueError('重复标签或空回答')
        grouped[row['case_id']][row['label']] = row
    packet, key = [], {}
    for case in cases:
        answers = grouped[case['id']]
        if set(answers) != {'A', 'B'} or {row['mode'] for row in answers.values()} != {'base', 'lora'}:
            raise ValueError('每题必须各有一个 Base 与 LoRA 回答')
        packet.append(dict(case_id=case['id'], topic=case['topic'], messages=case['messages'],
                           answers={label: answers[label]['answer'] for label in ('A', 'B')}))
        key[case['id']] = {label: answers[label]['mode'] for label in ('A', 'B')}
    return packet, key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', default='outputs/lora-lccc-base-v3-lr2e-5')
    parser.add_argument('--frozen-dir', default='outputs/naturalconv-real-chat-review-v1-frozen')
    parser.add_argument('--data-dir', default='data/naturalconv-v1')
    parser.add_argument('--output-dir', default='outputs/naturalconv-real-chat-review-v1')
    parser.add_argument('--seed', type=int, default=20261028)
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    audit = audit_naturalconv(data_root)
    frozen_root = Path(args.frozen_dir)
    frozen = json.loads((frozen_root / 'freeze-manifest.json').read_text(encoding='utf-8'))
    cases_bytes = (frozen_root / 'cases.json').read_bytes()
    if sha256(cases_bytes) != frozen['cases_sha256']:
        raise ValueError('冻结题目指纹不匹配')
    if audit['sha256']['eval'] != frozen['source_eval_sha256']:
        raise ValueError('NaturalConv评估集与冻结时版本不一致')
    cases = json.loads(cases_bytes)
    run_root = Path(args.run_dir)
    run_config = json.loads((run_root / 'run-config.json').read_text(encoding='utf-8'))
    if run_config.get('model_profile') != 'qwen3-1.7b':
        raise ValueError('候选LoRA必须以Qwen3-1.7B训练')
    if run_config.get('base_model') != 'Qwen/Qwen3-1.7B':
        raise ValueError('候选LoRA基座不匹配')
    candidate_adapter = run_root / 'adapter' / 'adapter_model.safetensors'
    if not candidate_adapter.is_file():
        raise FileNotFoundError('候选Adapter权重不存在')

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=False)
    packet_dir = output_root / 'reviewer-packet'
    packet_dir.mkdir()
    engine = QwenChat(adapter_run=run_root, model_profile='qwen3-1.7b', trim_history=True)
    rng = random.SystemRandom()
    results, prompt_info = [], {}
    for case in cases:
        messages = conversation_messages(dict(character_id='companion', messages=case['messages']))
        system = next((row['content'] for row in messages if row['role'] == 'system'), '')
        prompt_info[case['id']] = dict(sha256=sha256(system.encode('utf-8')) if system else None,
                                       included_profile='喜欢喝' in system or '窗边' in system)
        modes = ['base', 'lora']
        rng.shuffle(modes)
        mapping = dict(zip(('A', 'B'), modes))
        for label, mode in mapping.items():
            set_seed(args.seed)
            answer, metrics = engine.reply(messages, model_mode=mode)
            results.append(dict(case_id=case['id'], label=label, mode=mode,
                                answer=answer, metrics=metrics))
            (output_root / 'private-results.json').write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print(f"Generated {case['id']} {label}", flush=True)

    packet, key = build_packet(cases, results)
    instructions = [
        '# 拾语自然聊天盲评', '',
        '这次比较基础模型与一个低学习率 LoRA 版本，题目是公开真人双人对话的多轮上下文。它们由场景引导收集，不是聊天软件的随机日志。',
        '模型 A/B 每题随机；两边使用同一条小满设定和相同生成参数。真人原本的下一句没有展示，也不是标准答案。',
        '',
        '人工评分的用途：确认哪种回答更像你愿意继续聊的角色。你的选择能决定这个 Adapter 是否值得留下；它只代表你的使用偏好，不能单独证明对所有用户都有提升。',
        '',
        '每题先选一个：A 更自然、B 更自然、差不多、两个都不自然。原因可不填。若有问题，可写：没接住上下文、机械敷衍、编造经历/信息、角色不像 AI、答得太长或跑题。',
        '这组只评聊天体验，不要求你判定体育、电影、科技等内容的客观真伪；如果回答明显编造了个人经历，也请标出来。', '',
    ]
    for index, case in enumerate(packet, 1):
        lines = [f'## 第 {index} 题（{case["topic"]}）', '']
        for message in case['messages']:
            lines.append(f"- {'你' if message['role'] == 'user' else '对话中的另一位用户'}：{message['content']}")
        lines.extend(['', '**小满回答 A：** ' + case['answers']['A'], '',
                      '**小满回答 B：** ' + case['answers']['B'], '',
                      '- [ ] A 更自然', '- [ ] B 更自然', '- [ ] 差不多', '- [ ] 两个都不自然', '',
                      '原因或具体问题（可不填）：', '', '---', ''])
        instructions.extend(lines)
    score_rows = [dict(case_id=case['case_id'], preference=None, reason=None) for case in packet]
    (packet_dir / 'score-sheet.md').write_text('\n'.join(instructions), encoding='utf-8')
    (packet_dir / 'scores.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in score_rows), encoding='utf-8')
    (output_root / 'organizer-key.json').write_text(
        json.dumps(dict(mapping=key, note='请在评分完成前不要查看此文件'), ensure_ascii=False, indent=2), encoding='utf-8')
    summary = dict(
        status='review-pending', created_at=datetime.now().astimezone().isoformat(),
        base_model=run_config['base_model'], revision=run_config['revision'], seed=args.seed,
        candidate_run=str(run_root), candidate_learning_rate=run_config['learning_rate'],
        candidate_adapter_sha256=sha256(candidate_adapter.read_bytes()),
        generation_settings=GENERATION_SETTINGS, cases_sha256=frozen['cases_sha256'],
        source_eval_sha256=frozen['source_eval_sha256'], prompt_by_case=prompt_info,
        case_ids=[case['id'] for case in cases], case_count=len(cases),
        use_scope=frozen['use_scope'],
        note='本地小样本个人偏好评估；无用户评分前不报告模型胜负。',
    )
    (output_root / 'run-config.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='review-pending', case_count=len(cases),
                          score_sheet=str(packet_dir / 'score-sheet.md')),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
