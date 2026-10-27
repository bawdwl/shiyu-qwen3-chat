"""新评估：完整生成、格式自动评分、内容评审留空；不伪造人工评分。"""
import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import set_seed

from evaluate import grade
from inference import MODEL_ID, MODEL_REVISION, GENERATION_SETTINGS, QwenChat


def load_frozen(root):
    root=Path(root)
    raw=(root/'test.jsonl').read_bytes()
    digest=hashlib.sha256(raw).hexdigest()
    manifest=json.loads((root/'freeze-manifest.json').read_text(encoding='utf-8'))
    if digest!=manifest['test']['sha256']:
        raise ValueError('测试集指纹与冻结记录不同')
    cases=[json.loads(x) for x in raw.decode('utf-8').splitlines() if x.strip()]
    if len(cases)!=manifest['test']['count'] or len({x['id'] for x in cases})!=len(cases):
        raise ValueError('测试题数量或ID异常')
    return cases,digest


def evaluate_case(case,engine,mode):
    messages=[]
    turns=[]
    error=None
    try:
        for prompt in case['prompts']:
            messages.append(dict(role='user',content=prompt))
            answer,metrics=engine.reply(messages,model_mode=mode)
            messages.append(dict(role='assistant',content=answer))
            turns.append(dict(prompt=prompt,answer=answer,**metrics))
        check=case['format_check']
        format_pass=grade(turns[-1]['answer'],check) if check else None
    except Exception as exc:
        error=f'{type(exc).__name__}: {exc}'
        format_pass=False if case['format_check'] else None
        if isinstance(exc,torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
    return dict(id=case['id'],category=case['category'],turns=turns,error=error,
                auto_format_pass=format_pass,content_score=None,manual_format_pass=None,
                review_status='pending',content_reference=case['content_reference'],
                content_rubric=case['content_rubric'])


def summarize(records):
    automatic=[r for r in records if r['auto_format_pass'] is not None]
    turns=[t for r in records for t in r['turns']]
    seconds=sum(t['seconds'] for t in turns)
    return dict(total=len(records),errors=sum(r['error'] is not None for r in records),
                auto_format_eligible=len(automatic),auto_format_passed=sum(r['auto_format_pass'] for r in automatic),
                content_reviewed=0,content_pending=len(records),mean_content_score=None,
                manual_format_reviewed=0,measured_turns=len(turns),
                median_generation_seconds=statistics.median(t['seconds'] for t in turns) if turns else None,
                aggregate_tokens_per_second=sum(t['output_tokens'] for t in turns)/seconds if seconds else None,
                peak_allocated_gib=max((t['peak_allocated_gib'] for t in turns),default=None),
                truncated_turns=sum(not t['ended_with_eos'] for t in turns),
                note='自动格式分不等于内容质量；内容与人工格式未评审时为null。')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset-dir',default='data/v2')
    parser.add_argument('--adapter-run',default=None)
    parser.add_argument('--output-dir',default=None)
    parser.add_argument('--check-only',action='store_true',help='只检查冻结集，不生成答案')
    parser.add_argument('--historical-regression',action='store_true',help='单独运行已见的旧16题，不混入新50题')
    args=parser.parse_args()
    if args.historical_regression:
        raw=Path('data/eval.jsonl').read_bytes()
        digest=hashlib.sha256(raw).hexdigest()
        if digest!='48827ddc3908ccd0b827b1f60d9d5749f6ce74e0bd292bb1616a93ebe819afdc':
            raise ValueError('历史题集已被修改')
        original=[json.loads(x) for x in raw.decode('utf-8').splitlines() if x.strip()]
        cases=[dict(id=x['id'],category=x['category'],prompts=x['prompts'],
                    content_reference=x['check']['expected'],format_check=x['check'],
                    content_rubric=['按原材料核查结论，内容与格式分开']) for x in original]
    else:
        cases,digest=load_frozen(args.dataset_dir)
    if args.check_only:
        print(json.dumps(dict(cases=len(cases),sha256=digest,model_executed=False),ensure_ascii=False))
        return
    target=Path(args.output_dir or ('outputs/eval-v2-'+datetime.now().strftime('%Y%m%d-%H%M%S')))
    target.mkdir(parents=True,exist_ok=False)
    mode='lora' if args.adapter_run else 'base'
    config=dict(timestamp=datetime.now().astimezone().isoformat(),model=MODEL_ID,revision=MODEL_REVISION,
                seed=42,generation_settings=GENERATION_SETTINGS,enable_thinking=False,dtype='bfloat16',
                dataset_sha256=digest,case_count=len(cases),model_mode=mode,
                dataset_role='historical-seen-regression' if args.historical_regression else json.loads(
                    (Path(args.dataset_dir)/'freeze-manifest.json').read_text(encoding='utf-8')).get('role','frozen-v2-test'),
                python=platform.python_version(),torch=torch.__version__,transformers=transformers.__version__,
                evaluation_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                inference_script_sha256=hashlib.sha256(Path('inference.py').read_bytes()).hexdigest(),
                format_grader_sha256=hashlib.sha256(Path('evaluate.py').read_bytes()).hexdigest())
    if args.adapter_run:
        root=Path(args.adapter_run)
        config.update(adapter_run=str(root),adapter_sha256=hashlib.sha256((root/'adapter'/'adapter_model.safetensors').read_bytes()).hexdigest())
    def save(name,value):
        (target/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    save('config.json',config)
    save('status.json',dict(status='started'))
    try:
        engine=QwenChat(adapter_run=args.adapter_run)
        config['gpu']=torch.cuda.get_device_name(0)
        save('config.json',config)
        set_seed(42)
        engine.reply([dict(role='user',content='只回复：你好')],model_mode=mode)
        records=[]
        started=time.perf_counter()
        with (target/'results.jsonl').open('w',encoding='utf-8') as log:
            for case in cases:
                set_seed(42)
                result=evaluate_case(case,engine,mode)
                log.write(json.dumps(result,ensure_ascii=False)+'\n');log.flush()
                records.append(result)
                print(f"{case['id']} format={result['auto_format_pass']} content=pending",flush=True)
        summary=summarize(records)
        summary['elapsed_seconds_excluding_load_and_warmup']=time.perf_counter()-started
        save('summary.json',summary)
        # 明确留空的评审表，由实际评审者填写；不把生成器当人工评审者。
        with (target/'review-template.jsonl').open('w',encoding='utf-8') as log:
            for r in records:
                log.write(json.dumps(dict(case_id=r['id'],run_id=str(target),model_label=mode,
                    content_score=None,format_pass=None,reason=None,reviewer=None,reviewed_at=None),ensure_ascii=False)+'\n')
        save('status.json',dict(status='generation-completed-review-pending'))
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    except Exception as exc:
        save('status.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__=='__main__':
    main()
