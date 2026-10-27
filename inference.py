"""Qwen3 单轮推理示例：下载、GPU 推理，并记录本次真实结果。"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from threading import Lock
from contextlib import nullcontext

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from generation_options import for_request
from context_window import prepare_context_details
from same_chat_memory import inject_same_chat_memory
from runtime_models import get_profile

MODEL_ID = 'Qwen/Qwen3-0.6B'
MODEL_REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
GENERATION_SETTINGS = dict(max_new_tokens=256, do_sample=True,
                           temperature=0.7, top_p=0.8, top_k=20)


def adapter_base_matches(adapter_base, model_id, local_path=None):
    """Accept the canonical model ID or the exact local cache used for training."""
    if adapter_base == model_id:
        return True
    if not local_path:
        return False
    try:
        return Path(adapter_base).expanduser().resolve() == Path(local_path).expanduser().resolve()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def load_model(cache_dir=None, revision=MODEL_REVISION, local_files_only=True,model_id=MODEL_ID,model_path=None):
    """复用已下载模型；聊天界面默认离线加载固定版本。"""
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，请确认 NVIDIA 显卡可用并激活 qwen3-im-chat 环境。')
    options = dict(revision=revision, cache_dir=cache_dir, local_files_only=local_files_only)
    source=model_path or model_id
    tokenizer = AutoTokenizer.from_pretrained(source, **options)
    model = AutoModelForCausalLM.from_pretrained(
        source, **options, dtype=torch.bfloat16, device_map='cuda:0',
    ).eval()
    return tokenizer, model


class QwenChat:
    """只负责推理，历史由调用者传入，不把不同用户的对话存在模型中。"""

    def __init__(self, adapter_run=None, trim_history=False,model_profile='qwen3-0.6b',same_chat_memory=False):
        # 默认保持离线评估原有的超长拒绝策略；在线服务单独启用窗口选择。
        self.trim_history=trim_history
        self.same_chat_memory=bool(same_chat_memory)
        profile=get_profile(model_profile)
        self.model_id=profile['model_id'];self.model_revision=profile['revision']
        self.adapter_available = False
        if adapter_run:
            root = Path(adapter_run)
            config = json.loads((root / 'run-config.json').read_text(encoding='utf-8'))
            if config['base_model'] != self.model_id or config['revision'] != self.model_revision:
                raise ValueError('Adapter 训练记录与基础模型版本不匹配')
        self.tokenizer, self.model = load_model(model_id=self.model_id,revision=self.model_revision,model_path=profile['local_path'])
        if adapter_run:
            from peft import PeftConfig, PeftModel
            metadata = PeftConfig.from_pretrained(root / 'adapter')
            if not adapter_base_matches(metadata.base_model_name_or_path, self.model_id, profile['local_path']):
                raise ValueError('Adapter 元数据与基础模型不匹配')
            self.model = PeftModel.from_pretrained(self.model, root / 'adapter', is_trainable=False).eval()
            self.adapter_available = True
        self.lock = Lock()

    def _prepare_inputs(self,messages):
        inputs,metrics,_,omitted_pairs=prepare_context_details(
            self.tokenizer,messages,trim_history=self.trim_history)
        if not self.same_chat_memory:
            return inputs,metrics
        query=next((m['content'] for m in reversed(messages) if m['role']=='user'),'')
        augmented,memory_metrics=inject_same_chat_memory(messages,omitted_pairs,query)
        if not memory_metrics['same_chat_memory_retrieved']:
            metrics.update(memory_metrics)
            return inputs,metrics
        baseline_tokens=metrics['original_input_tokens']
        baseline_omitted=metrics['context_omitted_turns']
        inputs,final_metrics,_,_=prepare_context_details(
            self.tokenizer,augmented,trim_history=self.trim_history)
        final_metrics.update(memory_metrics)
        final_metrics['same_chat_memory_baseline_input_tokens']=baseline_tokens
        final_metrics['same_chat_memory_baseline_omitted_turns']=baseline_omitted
        return inputs,final_metrics

    def reply_stream(self,messages,model_mode,stop_event,emit,max_new_tokens=None,generation_overrides=None):
        """在调用线程生成；逐段回调，停止条件在每次解码后检查。"""
        from transformers import TextStreamer,StoppingCriteria,StoppingCriteriaList
        from stream_job import GenerationStopped
        settings=for_request(GENERATION_SETTINGS,max_new_tokens,generation_overrides)
        with self.lock:
            if model_mode not in ('base','lora') or (model_mode=='lora' and not self.adapter_available):
                raise ValueError('模型模式不可用')
            if stop_event.is_set():
                raise GenerationStopped()
            inputs,context_metrics=self._prepare_inputs(messages)
            input_tokens=inputs['input_ids'].shape[-1]
            if stop_event.is_set(): raise GenerationStopped()
            inputs=inputs.to('cuda:0')
            torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
            started=time.perf_counter();first_text=[None]
            class CallbackStreamer(TextStreamer):
                def on_finalized_text(self,text,stream_end=False):
                    if text:
                        if first_text[0] is None: first_text[0]=time.perf_counter()-started
                        emit(text)
            class Stop(StoppingCriteria):
                def __call__(self,input_ids,scores,**kwargs):
                    return stop_event.is_set()
            streamer=CallbackStreamer(self.tokenizer,skip_prompt=True,skip_special_tokens=True)
            context=self.model.disable_adapter() if self.adapter_available and model_mode=='base' else nullcontext()
            with context,torch.inference_mode():
                generated=self.model.generate(**inputs,**settings,streamer=streamer,
                                              stopping_criteria=StoppingCriteriaList([Stop()]))
            torch.cuda.synchronize()
            if stop_event.is_set():
                raise GenerationStopped()
            tokens=generated[0,input_tokens:]
            eos=self.model.generation_config.eos_token_id
            return self.tokenizer.decode(tokens,skip_special_tokens=True).strip(),dict(
                **context_metrics,
                input_tokens=input_tokens,output_tokens=len(tokens),max_new_tokens=settings['max_new_tokens'],seconds=time.perf_counter()-started,
                first_visible_text_seconds=first_text[0],ended_with_eos=int(tokens[-1]) in ([eos] if isinstance(eos,int) else eos),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3)

    def reply(self, messages, model_mode='base', max_new_tokens=None,generation_overrides=None):
        # 串行使用 GPU，避免多个请求同时生成导致显存峰值叠加。
        settings=for_request(GENERATION_SETTINGS,max_new_tokens,generation_overrides)
        with self.lock:
            if model_mode not in ('base', 'lora'):
                raise ValueError('模型选择无效')
            if model_mode == 'lora' and not self.adapter_available:
                raise ValueError('尚未加载 LoRA Adapter，请检查启动参数')
            inputs,context_metrics=self._prepare_inputs(messages)
            input_tokens = inputs['input_ids'].shape[-1]
            inputs = inputs.to('cuda:0')
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            # 只在本次请求内禁用 Adapter，锁覆盖开关与生成，避免不同会话串用模型。
            context = self.model.disable_adapter() if self.adapter_available and model_mode == 'base' else nullcontext()
            with context, torch.inference_mode():
                generated = self.model.generate(**inputs, **settings)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            new_tokens = generated[0, input_tokens:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if not answer:
                raise RuntimeError('模型返回了空回答，请重新发送。')
            eos = self.model.generation_config.eos_token_id
            ended = int(new_tokens[-1]) in ([eos] if isinstance(eos, int) else eos)
            return answer, {
                **context_metrics,
                'input_tokens': input_tokens, 'output_tokens': len(new_tokens),
                'max_new_tokens': settings['max_new_tokens'],
                'seconds': elapsed, 'ended_with_eos': ended,
                'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
                'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', default='请用中文简短介绍你自己，并解释什么是大语言模型。')
    parser.add_argument('--cache-dir', default=str(Path.home() / '.cache' / 'huggingface' / 'hub'))
    parser.add_argument('--revision', default='main')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--output', default='outputs/first-inference.json')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；本阶段要求 GPU 实测，不自动退回 CPU。')
    model_id = 'Qwen/Qwen3-0.6B'
    # 固定本次下载的提交，方便后续复现同一份模型。
    revision = HfApi().model_info(model_id, revision=args.revision).sha
    print(f'Model: {model_id}, revision: {revision}', flush=True)
    start = time.perf_counter()
    tokenizer, model = load_model(args.cache_dir, revision, local_files_only=False)
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - start
    messages = [{'role': 'user', 'content': args.prompt}]
    # 聊天模板加入角色标记；关闭 thinking，只生成直接回答。
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=True, return_tensors='pt',
    ).to('cuda:0')
    set_seed(42)
    settings = dict(max_new_tokens=args.max_new_tokens, do_sample=True,
                    temperature=0.7, top_p=0.8, top_k=20)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **settings)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    generated = output[0, inputs['input_ids'].shape[-1]:]
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    record = {
        'timestamp': datetime.now().astimezone().isoformat(),
        'model': model_id, 'revision': revision, 'cache_dir': args.cache_dir,
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
        'device': str(model.device), 'dtype': str(model.dtype),
        'seed': 42, 'enable_thinking': False, 'generation_settings': settings,
        'prompt': args.prompt, 'answer': answer,
        'load_seconds_including_download': load_seconds,
        'input_tokens': inputs['input_ids'].shape[-1],
        'output_tokens_including_special_tokens': len(generated),
        'generation_seconds': elapsed,
        'output_tokens_per_second': len(generated) / elapsed,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024**3,
        'ended_with_eos': int(generated[-1]) in (
            [model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int)
            else model.generation_config.eos_token_id),
        'measurement_note': '单次首次生成，无预热；包含提示词处理，不代表稳定性能。显存为本进程 PyTorch 统计。',
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
