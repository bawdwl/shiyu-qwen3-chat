"""汇总实际填写的评审；不把缺失值当成0分或通过。"""
import argparse
import json
from pathlib import Path


def summarize_reviews(rows):
    if len({r['case_id'] for r in rows})!=len(rows):
        raise ValueError('评审ID重复')
    content=[];formats=[]
    for r in rows:
        score=r['content_score'];fmt=r['format_pass']
        if score is not None and (type(score) is not int or score not in (0,1,2)):
            raise ValueError('内容分必须为整数0、1、2或null')
        if fmt is not None and type(fmt) is not bool:
            raise ValueError('格式判定必须为布尔值或null')
        if score is not None or fmt is not None:
            if not all(isinstance(r.get(k),str) and r[k].strip() for k in ['reason','reviewer','reviewed_at']):
                raise ValueError('已评审条目必须写理由、评审者和时间')
        if score is not None: content.append(score)
        if fmt is not None: formats.append(fmt)
    return dict(total=len(rows),content_reviewed=len(content),content_pending=len(rows)-len(content),
                mean_content_score=sum(content)/len(content) if content else None,
                fully_correct=sum(x==2 for x in content),
                manually_formatted=len(formats),manual_format_passed=sum(formats),
                note='分母为实际完成评审条目；不将内容分等同自动格式分。')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--reviews',required=True)
    args=p.parse_args()
    root=Path(args.run_dir)
    results=[json.loads(l) for l in (root/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[json.loads(l) for l in Path(args.reviews).read_text(encoding='utf-8').splitlines()]
    if {r['case_id'] for r in rows}!={r['id'] for r in results}:
        raise ValueError('评审题目与运行结果不一致')
    for r in rows:
        if Path(r['run_id']).resolve()!=root.resolve():
            raise ValueError('评审run_id不匹配')
    print(json.dumps(summarize_reviews(rows),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
