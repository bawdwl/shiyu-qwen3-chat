"""实际 GPU 验收：权重复用、两种模式多轮、往返切换与异常恢复。"""
import json
from pathlib import Path

import torch
from transformers import set_seed

from conversation import submit_message, switch_model
from inference import QwenChat


def main():
    engine=QwenChat(adapter_run='outputs/lora-v2')
    pointers={n:p.data_ptr() for n,p in engine.model.named_parameters() if 'lora_' not in n}
    model_identity=id(engine.model)
    results=[]
    for mode in ['base','lora']:
        history=[]
        for prompt in ['记住临时代号：青石73。只回复收到。','刚才的临时代号是什么？只回复代号。']:
            set_seed(42)
            text,history,display,status=submit_message(prompt,history,engine,mode)
            assert not text and display==history
        assert '青石73' in history[-1]['content'], history
        results.append(dict(mode=mode,history=history,status=status))
    prompt=[dict(role='user',content='计算 17+28，只输出结果数字。')]
    answers={}
    for label,mode in [('base_before','base'),('lora','lora'),('base_after','base')]:
        set_seed(42)
        answer,metrics=engine.reply(prompt,model_mode=mode)
        answers[label]=dict(answer=answer,**metrics)
    assert answers['base_before']['answer']==answers['base_after']['answer']
    # 显式制造生成异常，确认 disable_adapter 上下文恢复。
    original_generate=engine.model.generate
    def fail(**kwargs):
        raise RuntimeError('simulated generation failure')
    engine.model.generate=fail
    try:
        _,h,_,status=submit_message('异常测试',[],engine,'base')
        assert h==[] and '失败' in status
    finally:
        engine.model.generate=original_generate
    set_seed(42)
    recovered,_=engine.reply(prompt,model_mode='lora')
    assert recovered==answers['lora']['answer']
    assert switch_model('lora')[1:3]==([],[])
    assert model_identity==id(engine.model)
    assert pointers=={n:p.data_ptr() for n,p in engine.model.named_parameters() if 'lora_' not in n}
    result=dict(passed=True,shared_base_weight_pointers_unchanged=True,
                base_roundtrip_identical=True,exception_recovery=True,
                model_object_unchanged=True,conversations=results,answers=answers,
                note='固定种子的功能验收；显存仅本进程 PyTorch。异常为模拟，非真实运行故障。')
    Path('docs/model-switch-test.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
