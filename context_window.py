"""按真实聊天模板计算输入预算；只省略旧完整轮次，不改变原始消息。"""

def prepare_context_details(tokenizer,messages,budget=2048,trim_history=False):
    """返回编码结果、指标、实际保留消息，以及因预算裁掉的完整问答对。"""
    if type(budget) is not int or budget<1: raise ValueError('输入预算必须是正整数')
    selected=[dict(role=m['role'],content=m['content']) for m in messages]
    start=1 if selected and selected[0]['role']=='system' else 0
    body=selected[start:]
    if not body or len(body)%2!=1 or any(m['role']!=('user' if i%2==0 else 'assistant') for i,m in enumerate(body)):
        raise ValueError('上下文必须是完整历史问答和最新用户问题')
    def encode(items):
        return tokenizer.apply_chat_template(items,tokenize=True,add_generation_prompt=True,
            enable_thinking=False,return_dict=True,return_tensors='pt')
    inputs=encode(selected);original_tokens=inputs['input_ids'].shape[-1];omitted_pairs=[]
    if original_tokens>budget and not trim_history:
        raise ValueError(f'对话内容超过{budget} token，请缩短输入。')
    while inputs['input_ids'].shape[-1]>budget and len(selected)-start>1:
        omitted_pairs.append(dict(turn_index=len(omitted_pairs)+1,
                                  user=selected[start]['content'],
                                  assistant=selected[start+1]['content']))
        del selected[start:start+2]
        inputs=encode(selected)
    if inputs['input_ids'].shape[-1]>budget:
        raise ValueError(f'最新问题与必要提示超过{budget} token，请缩短本次输入；历史记录保留。')
    metrics=dict(context_policy='recent_complete_turns' if trim_history else 'reject_over_budget',
        context_input_budget=budget,original_input_tokens=original_tokens,
        context_omitted_turns=len(omitted_pairs),context_kept_turns=(len(selected)-start-1)//2)
    return inputs,metrics,selected,omitted_pairs


def prepare_context(tokenizer,messages,budget=2048,trim_history=False):
    """保持原有接口，供训练/评测和不需要检查裁剪来源的调用方使用。"""
    inputs,metrics,_,_=prepare_context_details(tokenizer,messages,budget,trim_history)
    return inputs,metrics
