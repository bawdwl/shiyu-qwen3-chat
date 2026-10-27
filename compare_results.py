"""核对评估配置并生成逐题对照；不调整题目或评分。"""
import argparse
import json
from pathlib import Path


def read_run(root):
    root = Path(root)
    config = json.loads((root/'config.json').read_text(encoding='utf-8'))
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    results = [json.loads(x) for x in (root/'results.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len({r['id'] for r in results}) == len(results) == summary['total']
    assert sum(r['passed'] for r in results) == summary['passed']
    return config, summary, results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True)
    parser.add_argument('--adapter',required=True)
    parser.add_argument('--output-dir',required=True)
    args=parser.parse_args()
    bc,bs,br=read_run(args.base)
    ac,ass,ar=read_run(args.adapter)
    keys=['model','revision','dataset_sha256','case_count','seed','generation_settings',
          'enable_thinking','dtype','torch','transformers','python','gpu','scope',
          'evaluation_script_sha256','inference_script_sha256']
    for key in keys:
        if bc[key]!=ac[key]:
            raise ValueError(f'对照配置不一致：{key}')
    assert 'adapter_sha256' not in bc and ac.get('adapter_sha256')
    assert [r['id'] for r in br]==[r['id'] for r in ar]
    rows=[]
    for b,a in zip(br,ar):
        assert b['check']==a['check'] and b['category']==a['category']
        change=('改善' if a['passed'] else '退化') if b['passed']!=a['passed'] else '不变'
        rows.append(dict(id=b['id'],category=b['category'],base_pass=b['passed'],
                         adapter_pass=a['passed'],change=change,
                         base_answer=b['turns'][-1]['answer'] if b['turns'] else None,
                         adapter_answer=a['turns'][-1]['answer'] if a['turns'] else None,
                         expected=b['check']['expected']))
    result=dict(base_summary=bs,adapter_summary=ass,matched_configuration_keys=keys,
                improved=[r['id'] for r in rows if r['change']=='改善'],
                regressed=[r['id'] for r in rows if r['change']=='退化'],cases=rows)
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=False)
    (out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    def cell(x):
        # JSON 转义保留行尾空格和换行，避免 Markdown 隐藏格式差别。
        return json.dumps(x,ensure_ascii=False).replace('|','\\|')
    lines=['# 固定题集逐题原始对照','',
           '答案以 JSON 字符串转义展示，保留换行和行尾空格。改善/退化仅指严格规则由失败变通过/通过变失败。','',
           '| ID | 基础通过 | LoRA通过 | 变化 | 基础回答 | LoRA回答 |',
           '|---|---|---|---|---|---|']
    for r in rows:
        lines.append(f"| {r['id']} | {r['base_pass']} | {r['adapter_pass']} | {r['change']} | {cell(r['base_answer'])} | {cell(r['adapter_answer'])} |")
    (out/'cases.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(base=bs['passed'],adapter=ass['passed'],total=bs['total'],
                         improved=result['improved'],regressed=result['regressed']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
