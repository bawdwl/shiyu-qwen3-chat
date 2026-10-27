"""Recheck frozen shared-role data and prove one representative BF16 LoRA step."""
import argparse
import gc
import json
import math
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from inference import load_model
from prepare_shared_role_data import write_json
from runtime_models import get_profile
from shared_role_data import (audit_shared_dataset, encode_role_sample,
                              load_registry, render_role_prompt)
from training_data import file_hash, read_records


def _check_data(data_dir, manifest):
    profile = get_profile('qwen3-1.7b')
    if (manifest.get('model', {}).get('model_id') != profile['model_id']
            or manifest.get('model', {}).get('revision') != profile['revision']):
        raise ValueError('冻结数据的基座模型 ID 或 revision 与训练配置不符')
    tokenizer = AutoTokenizer.from_pretrained(
        profile['local_path'], revision=profile['revision'], local_files_only=True)
    registry = load_registry(data_dir / 'role-cards.json')
    records_by_split = {
        split: read_records(data_dir / f'{split}.jsonl')
        for split in ('train', 'validation')
    }
    frozen_token_rows = json.loads((data_dir / 'token-audit.json').read_text(encoding='utf-8'))
    frozen_by_id = {row['id']: row for row in frozen_token_rows}
    if len(frozen_by_id) != 1058:
        raise ValueError('冻结 token 审计必须包含1058个目标')
    checked = 0
    max_train = None
    token_lengths = {'train': [], 'validation': []}
    prompt_lengths = {}
    for role in registry['roles']:
        prompt = render_role_prompt(registry, role['id'])
        prompt_lengths[role['id']] = len(tokenizer.encode(prompt, add_special_tokens=False))
    for split, records in records_by_split.items():
        for record in records:
            encoded = encode_role_sample(record, tokenizer, registry, max_length=1024)
            length = len(encoded['input_ids'])
            target = tokenizer.encode(record['messages'][-1]['content'],
                                       add_special_tokens=False) + [tokenizer.eos_token_id]
            if frozen_by_id.get(record['id'], {}).get('input_tokens') != length:
                raise ValueError('重算 token 数与冻结记录不一致：' + record['id'])
            if [value for value in encoded['labels'] if value != -100] != target:
                raise ValueError('监督目标或 EOS 与预检记录不一致：' + record['id'])
            if encoded['labels'][:-len(target)] != [-100] * (length - len(target)):
                raise ValueError('角色、用户或历史 token 未全部屏蔽：' + record['id'])
            full = tokenizer.apply_chat_template(
                [dict(role='system', content=render_role_prompt(registry, record['role_id']))]
                + record['messages'], tokenize=True, return_dict=False,
                add_generation_prompt=False, enable_thinking=False)
            end = max(index for index, token in enumerate(full)
                      if token == tokenizer.eos_token_id)
            if full[:end + 1] != encoded['input_ids']:
                raise ValueError('完整聊天模板与训练标签不匹配：' + record['id'])
            if length > 1024:
                raise ValueError('样本超出冻结的1024-token预算：' + record['id'])
            token_lengths[split].append(length)
            checked += 1
            if split == 'train' and (max_train is None or length > len(max_train['input_ids'])):
                max_train = encoded
    if checked != 1058 or len(records_by_split['train']) != 932 or len(records_by_split['validation']) != 126:
        raise ValueError('训练/验证目标数量与冻结方案不符')
    return tokenizer, max_train, dict(
        model_id=profile['model_id'], revision=profile['revision'],
        tokenizer=transformers.__version__, checked_targets=checked,
        train_targets=len(records_by_split['train']),
        validation_targets=len(records_by_split['validation']),
        maximum_train_input_tokens=max(token_lengths['train']),
        maximum_validation_input_tokens=max(token_lengths['validation']),
        prompt_tokens_by_role=prompt_lengths,
        all_context_labels_masked=True, full_template_matches=checked,
        target_and_eos_matches=checked, truncated_samples=0,
        planned_optimizer_steps=math.ceil(len(records_by_split['train']) / 4) * 2)


def _gpu_step(sample, profile):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用，无法完成显存预检')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('当前 GPU 不支持训练计划锁定的 BF16')
    free_before, total = torch.cuda.mem_get_info(0)
    minimum_free = 5 * 1024**3
    if free_before < minimum_free:
        raise RuntimeError(
            f'预检开始前可用显存仅 {free_before / 1024**3:.2f} GiB，'
            f'低于安全启动线 {minimum_free / 1024**3:.2f} GiB；未加载基座模型')
    torch.manual_seed(42)
    started = time.perf_counter()
    tokenizer, base = load_model(
        model_id=profile['model_id'], revision=profile['revision'],
        model_path=profile['local_path'])
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        task_type='CAUSAL_LM', r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'], bias='none'))
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-5, weight_decay=0.01)
    batch = {key: torch.tensor([sample[key]], device='cuda:0')
             for key in ('input_ids', 'attention_mask', 'labels')}
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(**batch).loss
    if not torch.isfinite(loss):
        raise RuntimeError('代表性训练步 loss 非有限值')
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
    if not torch.isfinite(grad_norm):
        raise RuntimeError('代表性训练步梯度非有限值')
    optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    report = dict(
        passed=True, device=torch.cuda.get_device_name(0),
        bf16_supported=torch.cuda.is_bf16_supported(),
        free_memory_before_gib=round(free_before / 1024**3, 3),
        total_memory_gib=round(total / 1024**3, 3),
        sample_input_tokens=int(batch['input_ids'].shape[-1]),
        finite_loss=True, finite_gradient=True, optimizer_step_completed=True,
        trainable_parameters=sum(parameter.numel() for parameter in params),
        peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        elapsed_seconds=round(elapsed, 2),
        torch=torch.__version__, peft=peft.__version__,
        note='One disposable training step on the longest training sample; no adapter or optimizer state was saved.')
    del batch, optimizer, params, model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return report


def preflight(base_dir, output_dir):
    base = Path(base_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('拒绝覆盖已有预检包：' + str(output))
    if not base.is_dir() or not output.parent.is_dir():
        raise FileNotFoundError('冻结评测包及输出父目录必须存在')
    audit = audit_shared_dataset(base, require_training_ready=False)
    manifest = json.loads((base / 'freeze-manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('final_evaluation_frozen'):
        raise ValueError('最终评测尚未冻结')
    tokenizer, longest_sample, data_report = _check_data(base, manifest)
    del tokenizer
    profile = get_profile('qwen3-1.7b')
    gpu_report = _gpu_step(longest_sample, profile)
    config = dict(
        model_profile='qwen3-1.7b', revision=profile['revision'],
        epochs=2, learning_rate=2e-5, max_length=1024,
        micro_batch=1, gradient_accumulation=4,
        lora_rank=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'v_proj'], dtype='bfloat16',
        gradient_checkpointing=True, sample_weighting='uniform_per_sample')
    report = dict(
        version='shared-role-training-preflight-v1',
        created_at=datetime.now().astimezone().isoformat(),
        data=data_report, gpu=gpu_report, locked_training_config=config,
        data_audit=audit,
        status='passed_pretraining_capacity_and_data_checks',
        limitation='A single representative step confirms basic fit only; full training can still fail because of runtime or thermal conditions.')
    with tempfile.TemporaryDirectory(prefix='.shared-role-preflight-', dir=output.parent) as temporary:
        stage = Path(temporary) / 'payload'
        shutil.copytree(base, stage)
        write_json(stage / 'gpu-preflight.json', report)
        frozen = json.loads((stage / 'freeze-manifest.json').read_text(encoding='utf-8'))
        frozen.update(
            stage='training_ready_after_full_data_and_gpu_preflight',
            training_ready=True,
            training_preflight=dict(
                file='gpu-preflight.json',
                sha256=file_hash(stage / 'gpu-preflight.json'),
                status=report['status']),
            training_config=config,
            remaining=[
                'Train and reload the first shared-role LoRA candidate',
                'Screen development outputs before the one-time frozen comparison',
                'Run the paired frozen chat and capability evaluation'
            ],
            note='Complete 400/50 data, frozen 30+20 evaluation, tokenizer/label audit, and one BF16 LoRA memory step passed. This is training readiness, not evidence that LoRA improves answers.')
        frozen['code_sha256'] = {
            name: file_hash(name) for name in (
                'shared_role_data.py', 'training_data.py',
                'prepare_shared_role_data.py', 'freeze_shared_role_evaluation.py',
                'preflight_shared_role_training.py')
        }
        write_json(stage / 'freeze-manifest.json', frozen)
        preflight_path = stage / 'preflight.json'
        qc = json.loads(preflight_path.read_text(encoding='utf-8'))
        qc.update(status=report['status'], training_ready=True,
                  gpu_memory_preflight=gpu_report,
                  full_data_token_preflight=data_report)
        write_json(preflight_path, qc)
        ready_audit = audit_shared_dataset(stage, require_training_ready=True)
        report['ready_audit'] = ready_audit
        write_json(stage / 'gpu-preflight.json', report)
        frozen['training_preflight']['sha256'] = file_hash(stage / 'gpu-preflight.json')
        write_json(stage / 'freeze-manifest.json', frozen)
        # The manifest hash references the final report bytes; verify once more.
        if frozen['training_preflight']['sha256'] != file_hash(stage / 'gpu-preflight.json'):
            raise ValueError('预检报告指纹不稳定')
        audit_shared_dataset(stage, require_training_ready=True)
        stage.rename(output)
    return dict(output_dir=str(output), training_ready=True,
                data=data_report, gpu=gpu_report,
                locked_training_config=config,
                planned_optimizer_steps=data_report['planned_optimizer_steps'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', default='outputs/shared-role-build-v12')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    print(json.dumps(preflight(args.base_dir, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
