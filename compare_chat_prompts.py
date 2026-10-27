"""Local prompt-development comparison; no training or blind scores."""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EXTRA={
    'companion':'优先回应用户这句话里的感受和细节，不把用户的感受或经历说成自己的。通常用1到3个自然短句，不用编号。问句不是必需的；用户不想被提问时不要追问。明确不要建议时，只接话和陪伴，不给任务、计划或方案。',
    'study':'根据用户的基础和时间限制控制内容。用户只要一个步骤时，就只给一个可以立即开始的小动作；代码尽量是可直接运行的最小例子。区分常见结论与适用边界，不把简化解释当作没有例外。',
    'interview':'先区分用户是要准备建议、技术解释还是模拟面试。请求建议时先直接回应，不强制进入提问。只有用户明确开始模拟面试时才每轮问一道题，根据已说的经历接着聊，不反复从自我介绍开始。整理经历时保留用户承认的不足，不擅自补充数据或成果。',
}
COMPACT={
    'companion':'你是AI日常聊天搭子小满。直接用中文回应用户当前说的话，每次一到三句。用户分享好事就一起高兴，用户累了就温和陪伴。尊重用户说的不提问、不建议；这种情况下用陈述句回应。只有用户需要建议才提供建议。不要叙述场景、代替用户动作或添加现实经历。角色档案只在用户询问设定时使用。',
    'study':'你是AI学习搭子知夏。用中文直接回答当前问题，默认只给必要解释和一个最小例子。用户限定时间或步骤时照做，不追加学习计划。区分对象本身与对象内部引用的内容，不确定时如实说明。角色档案只在用户询问设定时使用，不叙述环境。',
    'interview':'你是AI面试教练言川。用中文按当前请求行动：要建议就给建议，要整理表达就直接给一段整理后的表述，要模拟面试才问一道题。根据用户已有经历继续，不编造成果或数据，保留用户承认的不足。默认简短。角色档案只在用户询问设定时使用，不叙述环境。',
}


def main():
    from characters import conversation_messages
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',choices=['torch','mnn'],default='torch')
    parser.add_argument('--candidate',choices=['appended','compact'],default='appended')
    parser.add_argument('--only-role',choices=['companion','study','interview'])
    parser.add_argument('--skip-baseline',action='store_true')
    args=parser.parse_args()
    from transformers import AutoTokenizer

    directory=ROOT/'docs/prompt-comparison'
    baseline=json.loads((directory/'baseline-roles.json').read_text(encoding='utf-8'))
    candidate=copy.deepcopy(baseline)
    for role,text in (EXTRA if args.candidate=='appended' else COMPACT).items():
        candidate[role]['prompt']=(baseline[role]['prompt']+'\n'+text) if args.candidate=='appended' else text
    candidate_path=directory/('candidate-roles.json' if args.candidate=='appended' else 'compact-roles.json')
    candidate_path.write_text(json.dumps(candidate,ensure_ascii=False,indent=2),encoding='utf-8')
    scenarios=json.loads((ROOT/'data/chat_scenarios_v1.json').read_text(encoding='utf-8'))
    tokenizer=AutoTokenizer.from_pretrained(ROOT/'outputs/models/qwen3-1.7b',local_files_only=True)
    if args.runtime=='mnn':
        sys.path.insert(0,str(ROOT/'outputs/mnn-runtime'))
        import MNN.llm as mnnllm
        engine=mnnllm.create(str(ROOT/'outputs/models/qwen3-1.7b-mnn/config.json'))
        config=dict(backend_type='cpu',thread_num=4,max_new_tokens=160,sampler_type='greedy',reuse_kv=False,use_template=False)
        if not engine.set_config(config): raise RuntimeError('MNN config rejected')
        engine.load()
        runtime='MNN 3.6.1 Windows CPU / upstream Qwen3-1.7B 4-bit'
        target=directory/'results.json'
    else:
        from inference import QwenChat, GENERATION_SETTINGS
        from transformers import set_seed
        engine=QwenChat(model_profile='qwen3-1.7b')
        config={**GENERATION_SETTINGS,'max_new_tokens':320,'seed':20261023}
        runtime='PyTorch Windows CUDA / Qwen3-1.7B BF16 (app engine)'
        target=directory/'results-torch.json'
    if args.candidate=='compact': target=target.with_stem(target.stem+'-compact')
    records=[]
    variants=[('candidate',candidate)] if args.skip_baseline else [('baseline',baseline),('candidate',candidate)]
    for label,roles in variants:
        for case in scenarios['cases']:
            if args.only_role and case['character_id']!=args.only_role: continue
            history=[];turns=[]
            if args.runtime=='torch': set_seed(config['seed'])
            for question in case['questions']:
                messages=conversation_messages(dict(character_id=case['character_id'],messages=history),question,roles)
                if args.runtime=='torch':
                    answer,metrics=engine.reply(messages,max_new_tokens=config['max_new_tokens'])
                    turns.append(dict(question=question,answer=answer,**metrics,
                        question_marks=answer.count('？')+answer.count('?')))
                    history.extend([dict(role='user',content=question),dict(role='assistant',content=answer)])
                    continue
                prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
                ids=engine.tokenizer_encode(prompt)
                if ids!=tokenizer.encode(prompt,add_special_tokens=False): raise ValueError('Tokenizer mismatch')
                if len(ids)>2048: raise ValueError('Comparison scenario exceeds input budget')
                engine.reset();engine.generate_init();start=time.perf_counter()
                engine.response(prompt,False);context=engine.context
                if context.status not in (mnnllm.LlmStatus.NORMAL_FINISHED,mnnllm.LlmStatus.MAX_TOKENS_FINISHED):
                    raise RuntimeError(str(context.status))
                answer=tokenizer.decode(context.output_tokens,skip_special_tokens=True).strip()
                if not answer: raise ValueError('Empty response')
                turns.append(dict(question=question,answer=answer,input_tokens=len(ids),output_tokens=len(context.output_tokens),
                    seconds=time.perf_counter()-start,status=str(context.status),question_marks=answer.count('？')+answer.count('?')))
                history.extend([dict(role='user',content=question),dict(role='assistant',content=answer)])
            records.append(dict(variant=label,scenario=case['id'],character_id=case['character_id'],turns=turns))
            report=dict(runtime=runtime,config=config,
                note='Development scenarios exposed to prompt author. Later-turn histories differ by variant. Not blind evaluation, training, overall quality score or fair latency benchmark.',records=records)
            report['sources']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [ROOT/'data/chat_scenarios_v1.json',directory/'baseline-roles.json',candidate_path]}
            target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            print('Recorded '+label+' '+case['id'],flush=True)


if __name__=='__main__': main()
