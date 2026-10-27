"""在新进程重新加载 Adapter，做实际 GPU 生成检查。"""
import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import set_seed

from inference import GENERATION_SETTINGS, load_model
from training_data import file_hash
from runtime_models import get_profile
from shared_role_data import resolve_run_system_prompt


def resolve_profile(config):
    profile=get_profile(config.get('model_profile','qwen3-0.6b'))
    if config['base_model']!=profile['model_id'] or config['revision']!=profile['revision']:
        raise ValueError('训练记录与固定基础模型配置不一致')
    return profile


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    root=Path(args.run_dir)
    config=json.loads((root/'run-config.json').read_text(encoding='utf-8'))
    profile=resolve_profile(config)
    system_prompt = resolve_run_system_prompt(config, root)
    tokenizer,base=load_model(revision=config['revision'],model_id=profile['model_id'],model_path=profile['local_path'])
    model=PeftModel.from_pretrained(base,root/'adapter',is_trainable=False).eval()
    weights=[p for n,p in model.named_parameters() if 'lora_B' in n]
    assert weights and any(torch.count_nonzero(p).item()>0 for p in weights), 'Adapter 未更新'
    prompt='学习本地模型开发时，为什么要记录失败实验？请简短回答。'
    messages=([dict(role='system',content=system_prompt)] if system_prompt else [])+[dict(role='user',content=prompt)]
    inputs=tokenizer.apply_chat_template(messages,tokenize=True,
              add_generation_prompt=True,enable_thinking=False,return_dict=True,return_tensors='pt').to('cuda:0')
    set_seed(42)
    with torch.inference_mode():
        output=model.generate(**inputs,**GENERATION_SETTINGS)
    torch.cuda.synchronize()
    answer=tokenizer.decode(output[0,inputs['input_ids'].shape[-1]:],skip_special_tokens=True)
    assert answer.strip()
    result=dict(reloaded=True,device=str(model.device),model_id=profile['model_id'],revision=config['revision'],
                conditioning=config.get('conditioning'),prompt=prompt,answer=answer,
                adapter_sha256=file_hash(root/'adapter'/'adapter_model.safetensors'),
                lora_B_nonzero=True,note='只验证保存、重新加载和实际生成；不是质量对照评估。')
    (root/'reload-check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
