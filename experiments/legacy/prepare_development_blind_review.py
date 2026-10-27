"""Create an author-facing blind A/B sheet for two development-only LoRA runs."""
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

from training_data import file_hash


def prepare(cases_path, v1_path, v3_path, output_dir):
    cases_file, v1_file, v3_file = map(Path, (cases_path, v1_path, v3_path))
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError('拒绝覆盖已有评分包：' + str(output))
    if not all(path.is_file() for path in (cases_file, v1_file, v3_file)):
        raise FileNotFoundError('开发题或 v1/v3 原始输出文件不存在')

    packet = json.loads(cases_file.read_text(encoding='utf-8'))
    runs = {
        'v1': json.loads(v1_file.read_text(encoding='utf-8')),
        'v3': json.loads(v3_file.read_text(encoding='utf-8')),
    }
    cases = packet.get('cases', [])
    if len(cases) != 12 or sum(len(case['turns']) for case in cases) != 16:
        raise ValueError('开发评分包预期为12个场景、16个助手轮次')
    by_run = {name: {item['case_id']: item for item in run['cases']}
              for name, run in runs.items()}
    if any(len(index) != 12 for index in by_run.values()):
        raise ValueError('v1 或 v3 输出不完整')

    # Keep A/B identity stable across every turn in one conversation, then
    # randomize independently per conversation. The author-facing file omits
    # model names, source paths, review points, and generation metrics.
    rng = random.SystemRandom()
    key, reviewer_cases, score_rows = {}, [], []
    for index, case in enumerate(cases, 1):
        case_id = case['case_id']
        if set(by_run['v1']) != set(by_run['v3']) or case_id not in by_run['v1']:
            raise ValueError('v1/v3 case IDs do not match the frozen development prompts')
        first, second = rng.sample(['v1', 'v3'], 2)
        mapping = {'A': first, 'B': second}
        key[case_id] = mapping
        responses = {}
        for label, mode in mapping.items():
            rows = by_run[mode][case_id]['outputs']['lora']
            if len(rows) != len(case['turns']):
                raise ValueError('v1/v3 assistant turn count mismatch: ' + case_id)
            if [row['user'] for row in rows] != case['turns']:
                raise ValueError('v1/v3 user turns differ from frozen prompt: ' + case_id)
            responses[label] = [row['answer'] for row in rows]
        case_packet = dict(case_number=index, messages=case['turns'], answers=responses)
        reviewer_cases.append(case_packet)
        score_rows.append(dict(case_number=index, preference=None, reason=None))

    output.mkdir(parents=True, exist_ok=False)
    lines = [
        '# 开发阶段匿名回复对照', '',
        '这份表比较同一基础模型上两版 LoRA 的回复，题目是自然中文开发场景，不是最终冻结评测。每题看完整对话，选择一项；不需要给每个回复打分。',
        '如果两版都不想继续聊，请选“两个都不满意”。回答长度不是评分标准。可以只填选项，也可以补一句原因。', ''
    ]
    for item in reviewer_cases:
        lines += [f"## 第 {item['case_number']} 题", '']
        for turn_index, user_text in enumerate(item['messages']):
            lines += [f'**你说：** {user_text}', '']
            for label in ('A', 'B'):
                lines += [f'**回答 {label}：** {item["answers"][label][turn_index]}', '']
        lines += ['- [ ] A 更自然、也更接得住对话',
                  '- [ ] B 更自然、也更接得住对话',
                  '- [ ] 差不多',
                  '- [ ] 两个都不满意', '',
                  '原因（可不填）：', '', '---', '']
    sheet_path = output / 'score-sheet.md'
    sheet_path.write_text('\n'.join(lines), encoding='utf-8', newline='\n')
    (output / 'scores.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in score_rows),
        encoding='utf-8', newline='\n')
    (output / 'organizer-key.json').write_text(json.dumps(
        dict(mapping=key, note='评分完成前请勿查看：A/B 对应 v1/v3。'),
        ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    summary = dict(
        created_at=datetime.now().astimezone().isoformat(),
        status='review-pending', cases=12, assistant_turns=16,
        prompt_sha256=file_hash(cases_file), v1_output_sha256=file_hash(v1_file),
        v3_output_sha256=file_hash(v3_file),
        score_sheet_sha256=file_hash(sheet_path),
        purpose='Author preference check for development-only v1/v3 comparison; not the preregistered final evaluation.')
    (output / 'run-config.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                            encoding='utf-8', newline='\n')
    return dict(output_dir=str(output), cases=12, assistant_turns=16,
                score_sheet=str(sheet_path), status='review-pending')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cases', default='data/shared-role-chat-v1/development-v3-cases.json')
    parser.add_argument('--v1', default='outputs/lora-shared-role-v1/development-v3-screen-output.json')
    parser.add_argument('--v3', default='outputs/lora-shared-role-v3-data-revision/development-v3-screen-output.json')
    parser.add_argument('--output-dir', default='outputs/lora-shared-role-v3-data-revision/author-blind-review-v1-v3')
    args = parser.parse_args()
    print(json.dumps(prepare(args.cases, args.v1, args.v3, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
