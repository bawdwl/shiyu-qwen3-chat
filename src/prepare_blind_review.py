"""生成不含来源映射的评审包；映射单独保存在输出目录。"""
import argparse
import hashlib
import json
import random
import uuid
from pathlib import Path

from compare_evaluation_v2 import read_run

DIMENSIONS = ('correctness', 'completeness', 'relevance', 'factuality')
PAIR_KEYS = ('model', 'revision', 'seed', 'generation_settings', 'enable_thinking', 'dtype',
             'dataset_sha256', 'case_count', 'dataset_role', 'python', 'torch', 'transformers',
             'gpu', 'evaluation_script_sha256', 'inference_script_sha256', 'format_grader_sha256')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def make_packet(cases, base, lora, rng):
    batch_id = str(uuid.uuid4())
    order = list(cases)
    rng.shuffle(order)
    sources = {'base': {r['id']: r for r in base}, 'lora': {r['id']: r for r in lora}}
    packet, mapping, template = [], {}, []
    for case in order:
        modes = ['base', 'lora']
        rng.shuffle(modes)
        item = dict(batch_id=batch_id, case_id=case['id'], category=case['category'],
                    prompts=case['prompts'], reference=case['content_reference'],
                    rubric=case['content_rubric'], samples={})
        mapping[case['id']] = {}
        for label, mode in zip(('A', 'B'), modes):
            row = sources[mode][case['id']]
            mapping[case['id']][label] = mode
            item['samples'][label] = dict(
                failed=row['error'] is not None,
                turns=[dict(prompt=t['prompt'], answer=t['answer'],
                            truncated=not t['ended_with_eos']) for t in row['turns']])
            template.append(dict(batch_id=batch_id, case_id=case['id'], sample_label=label,
                                 scores={key: None for key in DIMENSIONS},
                                 format_pass=None, reason=None, reviewer=None, reviewed_at=None))
        packet.append(item)
    return packet, mapping, template


def summarize_reviews(rows, mapping, batch_id=None):
    if any(set(labels) != {'A', 'B'} or set(labels.values()) != {'base', 'lora'}
           for labels in mapping.values()):
        raise ValueError('每题映射必须恰好包含一个Base和一个LoRA')
    if batch_id is not None and any(r.get('batch_id') != batch_id for r in rows):
        raise ValueError('评分与来源映射不属于同一评审批次')
    expected = {(case_id, label) for case_id, labels in mapping.items() for label in labels}
    actual = [(r['case_id'], r['sample_label']) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('评审必须覆盖映射中的全部样本且不可重复')
    by_model = {mode: {d: [] for d in DIMENSIONS} for mode in ('base', 'lora')}
    formats = {mode: [] for mode in by_model}
    indexed = {}
    reviewers = set()
    for row in rows:
        if set(row['scores']) != set(DIMENSIONS):
            raise ValueError('评分维度不匹配')
        scores = row['scores']
        for score in scores.values():
            if score is not None and (type(score) is not int or score not in (0, 1, 2)):
                raise ValueError('分数必须是整数0、1、2或null，不接受布尔值')
        fmt = row['format_pass']
        if fmt is not None and type(fmt) is not bool:
            raise ValueError('格式必须是布尔值或null')
        if any(v is not None for v in scores.values()) or fmt is not None:
            if not all(isinstance(row.get(k), str) and row[k].strip()
                       for k in ('reason', 'reviewer', 'reviewed_at')):
                raise ValueError('已评分样本必须有理由、真实评审者和时间')
            reviewers.add(row['reviewer'].strip())
        mode = mapping[row['case_id']][row['sample_label']]
        indexed[(row['case_id'], mode)] = scores
        for dimension, score in scores.items():
            if score is not None:
                by_model[mode][dimension].append(score)
        if fmt is not None:
            formats[mode].append(fmt)
    models = {}
    for mode in by_model:
        models[mode] = {d: dict(reviewed=len(v), pending=len(mapping)-len(v),
                               mean=sum(v)/len(v) if v else None)
                        for d, v in by_model[mode].items()}
        models[mode]['format'] = dict(reviewed=len(formats[mode]),
                                      pending=len(mapping)-len(formats[mode]),
                                      passed=sum(formats[mode]))
    paired = {}
    for d in DIMENSIONS:
        differences = [indexed[(cid, 'lora')][d] - indexed[(cid, 'base')][d]
                       for cid in mapping if indexed[(cid, 'lora')][d] is not None
                       and indexed[(cid, 'base')][d] is not None]
        paired[d] = dict(paired=len(differences), improved=sum(v > 0 for v in differences),
                         tied=sum(v == 0 for v in differences), regressed=sum(v < 0 for v in differences),
                         mean_delta=sum(differences)/len(differences) if differences else None)
    return dict(models=models, paired=paired, reviewers=sorted(reviewers),
                note='仅汇总实际填写分数；不自动证明独立性、人工身份或盲评执行情况。')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--lora', required=True)
    parser.add_argument('--dataset-dir', default='data/quality-holdout-v1')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    bc, bs, br = read_run(args.base)
    lc, ls, lr = read_run(args.lora)
    for key in PAIR_KEYS:
        if bc[key] != lc[key]:
            raise ValueError('配对配置不一致：' + key)
    if bc['model_mode'] != 'base' or lc['model_mode'] != 'lora':
        raise ValueError('必须为Base与LoRA配对')
    raw = (Path(args.dataset_dir) / 'test.jsonl').read_bytes()
    manifest = json.loads((Path(args.dataset_dir) / 'freeze-manifest.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest['test']['sha256'] or digest != bc['dataset_sha256']:
        raise ValueError('题集指纹不匹配')
    cases = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    ids = [case['id'] for case in cases]
    if len(ids) != len(set(ids)) or len(ids) != manifest['test']['count'] or len(ids) != bc['case_count']:
        raise ValueError('题集数量或ID异常')
    if ids != [r['id'] for r in br] or ids != [r['id'] for r in lr]:
        raise ValueError('运行题目与冻结集不一致')
    for case, b, l in zip(cases, br, lr):
        for row in (b, l):
            if row['content_reference'] != case['content_reference']:
                raise ValueError('参考答案不一致')
            prompts = [t['prompt'] for t in row['turns']]
            if prompts != case['prompts'][:len(prompts)]:
                raise ValueError('实际问题与题集不一致')
            if row['error'] is None and len(prompts) != len(case['prompts']):
                raise ValueError('成功记录缺少轮次')
    packet, mapping, template = make_packet(cases, br, lr, random.SystemRandom())
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    public = root / 'reviewer-packet'
    public.mkdir()
    save(public / 'questions.json', packet)
    (public / 'scores.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in template), encoding='utf-8')
    lines = ['# 匿名回答评审', '', '每题A/B来源独立随机，不跨题固定。只评末轮，前轮作为上下文。请先读评分协议，填写scores.jsonl；所有未评审分保持null。', '']
    for item in packet:
        lines += [f"## {item['case_id']} · {item['category']}", '',
                  '参考：' + item['reference'], '', '要点：' + '；'.join(item['rubric']), '']
        for label, sample in item['samples'].items():
            lines += [f'### 回答 {label}', '', f"生成失败：{sample['failed']}", '']
            for i, turn in enumerate(sample['turns'], 1):
                lines += [f'第{i}轮问题：', ''] + ['    '+s for s in turn['prompt'].splitlines()]
                lines += ['', f"第{i}轮回答（达到预算而未EOS：{turn['truncated']}）：", '']
                lines += ['    '+s for s in turn['answer'].splitlines()] + ['']
    (public / '阅读与评分.md').write_text('\n'.join(lines), encoding='utf-8')
    (public / '评分协议.md').write_bytes(Path('docs/独立质量评估协议.md').read_bytes())
    save(root / 'organizer-key.json', dict(batch_id=template[0]['batch_id'], mapping=mapping,
        source_hashes={mode: hashlib.sha256((Path(path)/'results.jsonl').read_bytes()).hexdigest()
                       for mode, path in [('base', args.base), ('lora', args.lora)]},
        question_packet_sha256=hashlib.sha256((public/'questions.json').read_bytes()).hexdigest()))
    save(root / 'pending-summary.json', summarize_reviews(template, mapping, template[0]['batch_id']))
    print('Created reviewer-packet; organizer-key.json must be withheld from reviewers.')


if __name__ == '__main__':
    main()
