"""Render complete dev answers side by side in prose; no automatic quality score."""
import argparse
import json
from pathlib import Path


def load(root):
    root = Path(root)
    summary = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (root / 'answers.jsonl').open(encoding='utf-8') if line.strip()]
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    base_summary, base = load(args.base)
    candidate_summary, candidate = load(args.candidate)
    for key in ('suite_sha256', 'manifest_sha256', 'protocol', 'split'):
        if base_summary[key] != candidate_summary[key]:
            raise ValueError('Comparison conditions differ: ' + key)
    if base_summary.get('prompt_condition') != candidate_summary.get('prompt_condition'):
        raise ValueError('Prompt conditions differ; cannot attribute change to LoRA')
    if base_summary['split'] != 'development' or len(base) != len(candidate):
        raise ValueError('Expected complete matching development runs')
    text = '# 首轮 LoRA 开发题完整对照\n\n这是公开标签的开发诊断材料，不是盲评，也不是最终测试成绩。没有附加人工评分。\n\n'
    text += '提示条件：' + base_summary.get('prompt_condition', {}).get('mode', 'legacy card only') + '。\n\n'
    for left, right in zip(base, candidate):
        if left['id'] != right['id'] or len(left['turns']) != len(right['turns']):
            raise ValueError('Cases or turns differ')
        text += f"## {left['id']} · {left['role_id']}\n\n"
        for i, (b, c) in enumerate(zip(left['turns'], right['turns']), 1):
            if b['user'] != c['user']:
                raise ValueError('User prompts differ')
            text += f"### 第{i}轮\n\n用户：{b['user']}\n\n基础模型：\n\n{b['answer']}\n\nLoRA：\n\n{c['answer']}\n\n"
            text += f"生成token：基础 {b['generated_tokens']} / LoRA {c['generated_tokens']}；达到上限：{b['reached_token_limit']} / {c['reached_token_limit']}。\n\n"
    text += '多轮后续模型历史包含各自先前回答，所以后续输入token数可能不同。总耗时不能在输出长度不同的情况下直接作为推理加速证据。\n'
    Path(args.output).write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
