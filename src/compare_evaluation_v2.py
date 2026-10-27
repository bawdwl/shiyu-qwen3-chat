"""核对新协议的配对运行，单独呈现自动格式分与原始回答。"""
import argparse
import json
from pathlib import Path


def read_run(path):
    root=Path(path)
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    records=[json.loads(x) for x in (root/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records)!=summary['total'] or len({x['id'] for x in records})!=len(records):
        raise ValueError('运行结果不完整或ID重复')
    return config,summary,records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    parser.add_argument('--lora',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    bc,bs,br=read_run(args.base);lc,ls,lr=read_run(args.lora)
    keys=['model','revision','seed','generation_settings','enable_thinking','dtype','dataset_sha256',
          'case_count','dataset_role','python','torch','transformers','gpu',
          'evaluation_script_sha256','inference_script_sha256','format_grader_sha256']
    for key in keys:
        if bc[key]!=lc[key]:
            raise ValueError(f'对照条件不一致：{key}')
    if bc['model_mode']!='base' or lc['model_mode']!='lora':
        raise ValueError('需要基础与LoRA配对运行')
    if [r['id'] for r in br]!=[r['id'] for r in lr]:
        raise ValueError('题目ID或顺序不同')
    rows=[]
    for b,l in zip(br,lr):
        assert b['content_reference']==l['content_reference']
        rows.append(dict(id=b['id'],category=b['category'],reference=b['content_reference'],
                         base_answer=b['turns'][-1]['answer'] if b['turns'] else None,
                         lora_answer=l['turns'][-1]['answer'] if l['turns'] else None,
                         base_format=b['auto_format_pass'],lora_format=l['auto_format_pass'],
                         base_error=b['error'],lora_error=l['error']))
    result=dict(base_summary=bs,lora_summary=ls,conditions_checked=keys,
                format_improvements=[r['id'] for r in rows if r['base_format'] is False and r['lora_format'] is True],
                format_regressions=[r['id'] for r in rows if r['base_format'] is True and r['lora_format'] is False],
                human_review_status='pending',cases=rows)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# 原始末轮回答对照','',
           '只比较自动严格格式；内容须另外核查。JSON字符串转义保留换行和空格。','',
           '| ID | 基础格式 | LoRA格式 | 基础回答 | LoRA回答 |',
           '|---|---|---|---|---|']
    for r in rows:
        encode=lambda x:json.dumps(x,ensure_ascii=False).replace('|','\\|')
        lines.append(f"| {r['id']} | {r['base_format']} | {r['lora_format']} | {encode(r['base_answer'])} | {encode(r['lora_answer'])} |")
    (out/'cases.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','conditions_checked']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
