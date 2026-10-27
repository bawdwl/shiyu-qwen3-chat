"""单 GPU 小规模 LoRA；保留逐步日志，不使用固定测试题选择模型。"""
import argparse
import hashlib
import json
import math
import platform
import shutil
import time
from datetime import datetime
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, get_peft_model
from transformers import set_seed

from inference import load_model
from training_data import (audit_lccc, audit_naturalconv, audit_splits, file_hash,
                           encode_sample, equal_bucket_sample_weights, read_records)
from runtime_models import get_profile
from release_role_data import (DATA_VERSION as RELEASE_ROLE_VERSION,
                               audit_release_dataset, encode_release_sample)
from training_recipe import LORA_MODULES, accumulation_factors, lr_multiplier
from shared_role_data import (DATA_VERSION as SHARED_ROLE_VERSION, audit_shared_dataset,
                              encode_role_sample, load_registry)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--data-dir', default='data', help='data 或 data/v2；不会自动切换历史数据')
    parser.add_argument('--model-profile', choices=['qwen3-0.6b', 'qwen3-1.7b'], default='qwen3-0.6b')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--learning-rate', type=float, default=0.0002,
                        help='AdamW 学习率；默认值保持历史训练设置')
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--gradient-accumulation', type=int, default=4)
    parser.add_argument('--lora-rank', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--target-modules', nargs='+', default=['q_proj', 'v_proj'])
    parser.add_argument('--lr-scheduler', choices=['constant', 'cosine'], default='constant')
    parser.add_argument('--warmup-ratio', type=float, default=0.0)
    parser.add_argument('--loss-normalization', choices=['samples', 'tokens'], default='samples')
    parser.add_argument('--save-half-epoch', action='store_true',
                        help='额外保存半轮Adapter与验证loss，供开发集选择；不是完整优化器恢复点')
    parser.add_argument('--balance-target-length', action='store_true',
                        help='训练时对<=10字与>10字回复做等总样本权重；保留所有回复，不代表长度是质量')
    parser.add_argument('--shared-long-target-weight', type=float, default=1.0,
                        help='共享角色实验单因素选项：把含EOS目标长于32 tokens的样本相对加权；默认1.0即均匀')
    parser.add_argument('--character-id', choices=['companion', 'study', 'interview'], default=None,
                        help='可选：快照 App 的完整角色 system 提示；默认保留历史无 system 的训练行为')
    args = parser.parse_args()
    if args.epochs < 1 or args.max_length < 32 or args.gradient_accumulation < 1:
        parser.error('--epochs、--max-length 和 --gradient-accumulation 必须为正数且长度至少32')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error('--learning-rate 必须是有限正数')
    if args.lora_rank < 1 or args.lora_alpha < 1:
        parser.error('LoRA rank/alpha 必须为正整数')
    if not math.isfinite(args.lora_dropout) or not 0 <= args.lora_dropout < 1:
        parser.error('LoRA dropout 必须在[0,1)')
    if not math.isfinite(args.warmup_ratio) or not 0 <= args.warmup_ratio < 1:
        parser.error('warmup ratio 必须在[0,1)')
    if not set(args.target_modules) <= LORA_MODULES or len(set(args.target_modules)) != len(args.target_modules):
        parser.error('target modules 必须为不重复的Qwen注意力/MLP投影层名称')
    if not math.isfinite(args.shared_long_target_weight) or not 1.0 <= args.shared_long_target_weight <= 3.0:
        parser.error('--shared-long-target-weight 必须在1.0到3.0之间')
    if args.loss_normalization == 'tokens' and (args.balance_target_length or args.shared_long_target_weight != 1.0):
        parser.error('token归一化不能同时叠加旧长度权重实验')
    data_root = Path(args.data_dir)
    freeze_path = data_root / 'freeze-manifest.json'
    freeze = json.loads(freeze_path.read_text(encoding='utf-8')) if freeze_path.exists() else {}
    is_shared = freeze.get('version') == SHARED_ROLE_VERSION
    is_release = freeze.get('version') == RELEASE_ROLE_VERSION
    role_registry = None
    if is_release:
        if args.character_id or args.balance_target_length or args.shared_long_target_weight != 1.0:
            parser.error('新发布实验使用逐条角色条件，不兼容旧角色覆盖/长度加权')
        if args.model_profile != 'qwen3-1.7b':
            parser.error('新发布实验固定 Qwen3-1.7B')
        audit = audit_release_dataset(data_root)
    elif is_shared:
        if args.character_id or args.balance_target_length:
            parser.error('共享角色数据不能覆盖为单一角色或启用旧长度分桶权重')
        audit = audit_shared_dataset(data_root, require_training_ready=True)
        role_registry = load_registry(data_root / 'role-cards.json')
    elif freeze.get('version') == 'naturalconv-v1-pilot':
        audit = audit_naturalconv(data_root)
    elif freeze.get('version') == 'lccc-base-pilot-v1':
        audit = audit_lccc(data_root)
    else:
        audit = audit_splits(data_root)
    if not is_shared and args.shared_long_target_weight != 1.0:
        parser.error('--shared-long-target-weight 只适用于共享角色实验')
    profile = get_profile(args.model_profile)
    system_prompt = None
    conditioning = dict(character_id=None, system_prompt=None, sha256=None,
                        source='legacy alternating user/assistant messages without system')
    if is_release:
        conditioning = dict(mode='per_row_system_prompt',
                            source='frozen role-conditioned release dataset; system labels masked')
    if is_shared:
        conditioning = dict(mode='per_sample_role_card', renderer_version=audit['renderer_version'],
                            role_cards_file='role-cards.json', role_cards_sha256=audit['role_cards_sha256'],
                            verification_role_id=sorted(audit['dialogues_by_role']['train'])[0],
                            source='frozen registry; each sample resolves its own card; system labels masked')
    if args.character_id:
        from characters import conversation_messages
        system_prompt = conversation_messages(dict(character_id=args.character_id, messages=[]))[0]['content']
        conditioning = dict(character_id=args.character_id, system_prompt=system_prompt,
                            sha256=hashlib.sha256(system_prompt.encode('utf-8')).hexdigest(),
                            source='characters.conversation_messages snapshot; system labels are masked')
    is_naturalconv = freeze.get('version') == 'naturalconv-v1-pilot'
    is_lccc = freeze.get('version') == 'lccc-base-pilot-v1'
    trim_history = is_naturalconv or is_lccc
    target = Path(args.output_dir or ('outputs/lora-' + datetime.now().strftime('%Y%m%d-%H%M%S')))
    target.mkdir(parents=True, exist_ok=False)
    config = dict(timestamp=datetime.now().astimezone().isoformat(), model_profile=args.model_profile,
                  base_model=profile['model_id'], revision=profile['revision'],
                  seed=42, epochs=args.epochs, micro_batch=1,
                  gradient_accumulation=args.gradient_accumulation, learning_rate=args.learning_rate,
                  max_length=args.max_length,
                  r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                  target_modules=args.target_modules,
                  dtype='bfloat16', gradient_checkpointing=True,
                  optimizer='AdamW', weight_decay=0.01, max_grad_norm=1.0,
                  schedule=args.lr_scheduler, warmup_ratio=args.warmup_ratio,
                  loss_normalization=args.loss_normalization, save_half_epoch=args.save_half_epoch,
                  loss=f'只监督助手回答和结束 token；每次更新最多{args.gradient_accumulation}条样本；按{args.loss_normalization}归一化',
                  history_policy=('超长样本仅移除最早的完整 user/assistant 轮次；不截断消息文本'
                                  if trim_history else '拒绝超长样本，不静默截断'),
                  selection=(f'固定{args.epochs}轮，保存半轮和最终Adapter；仅用验证/开发集选候选，最终题不参与'
                             if args.save_half_epoch else
                             f'固定{args.epochs}轮，保存最终Adapter；不根据测试集或验证损失选择轮数'),
                  sample_weighting=dict(
                      enabled=(args.balance_target_length or
                               (is_shared and args.shared_long_target_weight > 1.0)),
                      method=('equal_total_sample_weight_by_target_character_length'
                              if args.balance_target_length else
                              'relative_long_target_weight' if is_shared and args.shared_long_target_weight > 1.0
                              else 'uniform_per_sample'),
                      threshold_chars=(10 if args.balance_target_length else None),
                      threshold_target_tokens_including_eos=(32 if is_shared and args.shared_long_target_weight > 1.0 else None),
                      long_target_multiplier=(args.shared_long_target_weight if is_shared else None)),
                  conditioning=conditioning,
                  python=platform.python_version(), torch=torch.__version__,
                  transformers=transformers.__version__, peft=peft.__version__, data=audit)
    if is_release:
        config['training_source_sha256'] = {
            name: file_hash(Path(__file__).resolve().parent / name)
            for name in ('train_lora.py', 'training_recipe.py', 'training_data.py', 'release_role_data.py')
        }
    def save_json(name, value):
        (target/name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    save_json('run-config.json', config)
    if is_release:
        shutil.copyfile(data_root / 'freeze-manifest.json', target / 'freeze-manifest.json')
    if is_shared:
        shutil.copyfile(data_root / 'role-cards.json', target / 'role-cards.json')
    save_json('status.json', dict(status='started'))
    log = (target/'training-log.jsonl').open('w', encoding='utf-8')
    def record(event):
        log.write(json.dumps(event, ensure_ascii=False)+'\n')
        log.flush()
        print(json.dumps(event, ensure_ascii=False), flush=True)
    try:
        set_seed(42)
        tokenizer, base = load_model(model_id=profile['model_id'], revision=profile['revision'],
                                     model_path=profile['local_path'])
        base.config.use_cache = False
        model = get_peft_model(base, LoraConfig(
            task_type='CAUSAL_LM', r=args.lora_rank, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=args.target_modules, bias='none',
        ))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads()
        model.print_trainable_parameters()
        train_rows = read_records(data_root/'train.jsonl')
        def encode_record(row):
            if is_release:
                return encode_release_sample(row, tokenizer, max_length=args.max_length)
            if is_shared:
                return encode_role_sample(row, tokenizer, role_registry, max_length=args.max_length)
            return encode_sample(row, tokenizer, max_length=args.max_length,
                                 trim_history=trim_history, system_prompt=system_prompt)
        train = [encode_record(row) for row in train_rows]
        if args.balance_target_length:
            sample_weights, weighting_audit = equal_bucket_sample_weights(train_rows, threshold_chars=10)
        elif is_shared and args.shared_long_target_weight > 1.0:
            target_lengths = [sum(token != -100 for token in sample['labels'])
                              for sample in train]
            raw_weights = [args.shared_long_target_weight if length > 32 else 1.0
                           for length in target_lengths]
            mean_weight = sum(raw_weights) / len(raw_weights)
            sample_weights = [weight / mean_weight for weight in raw_weights]
            short_count = sum(length <= 32 for length in target_lengths)
            long_count = len(target_lengths) - short_count
            weighting_audit = dict(
                method='relative_long_target_weight',
                threshold_target_tokens_including_eos=32,
                long_target_multiplier=args.shared_long_target_weight,
                counts=dict(short_or_equal_32_tokens=short_count, long_over_32_tokens=long_count),
                normalized_weights=dict(short_or_equal_32_tokens=1.0 / mean_weight,
                                        long_over_32_tokens=args.shared_long_target_weight / mean_weight),
                mean_sample_weight=sum(sample_weights) / len(sample_weights),
                note='Single controlled weighting change; long replies receive modestly higher loss weight. Weighting does not make length a quality metric.')
        else:
            sample_weights = [1.0] * len(train_rows)
            weighting_audit = dict(method='uniform_per_sample', threshold_chars=None,
                                   counts={'all_targets': len(train_rows)}, weights={'all_targets': 1.0},
                                   note='No target-length weighting.')
        config['sample_weighting'].update(weighting_audit)
        save_json('run-config.json', config)
        validation = [encode_record(row) for row in read_records(data_root/'validation.jsonl')]
        save_json('data-audit.json', dict(**audit, sample_weighting=weighting_audit,
                                        train_target_char_lengths=[len(x['messages'][-1]['content']) for x in train_rows],
                                        train_lengths=[len(x['input_ids']) for x in train],
                                        validation_lengths=[len(x['input_ids']) for x in validation],
                                        train_history_pairs_trimmed=sum(x.get('history_pairs_trimmed', 0) for x in train),
                                        validation_history_pairs_trimmed=sum(x.get('history_pairs_trimmed', 0) for x in validation)))
        params = [p for p in model.parameters() if p.requires_grad]
        assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
        config.update(trainable_parameters=sum(p.numel() for p in params),
                      total_parameters=sum(p.numel() for p in model.parameters()),
                      gpu=torch.cuda.get_device_name(0))
        save_json('run-config.json', config)
        optimizer = torch.optim.AdamW(params,lr=config['learning_rate'],weight_decay=0.01)
        updates_per_epoch = math.ceil(len(train) / args.gradient_accumulation)
        total_updates = updates_per_epoch * args.epochs
        target_counts = [sum(token != -100 for token in sample['labels'][1:]) for sample in train]
        if not train or not validation:
            raise ValueError('训练/验证集合不能为空')
        config.update(optimizer_steps_planned=total_updates,
                      supervised_train_tokens_per_epoch=sum(target_counts),
                      checkpoint_policy='Optional half-epoch Adapter plus final Adapter; no optimizer resume state')
        save_json('run-config.json', config)
        def batch(sample):
            return {k:torch.tensor([sample[k]],device='cuda:0') for k in ('input_ids','attention_mask','labels')}
        def validate():
            model.eval()
            weighted, count = 0.0, 0
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for sample in validation:
                    n = sum(x != -100 for x in sample['labels'][1:])
                    loss = model(**batch(sample)).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('验证 loss 非有限值')
                    weighted += loss.item()*n
                    count += n
            return weighted/count
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        before = validate()
        record(dict(event='validation_before',loss=before))
        generator = torch.Generator().manual_seed(42)
        step = 0
        for epoch in range(args.epochs):
            model.train()
            order = torch.randperm(len(train),generator=generator).tolist()
            for offset in range(0,len(order),args.gradient_accumulation):
                group=order[offset:offset+args.gradient_accumulation]
                optimizer.zero_grad(set_to_none=True)
                losses=[]
                weighted_losses=[]
                factors = accumulation_factors([target_counts[idx] for idx in group], args.loss_normalization)
                current_lr = config['learning_rate'] * lr_multiplier(step + 1, total_updates, args.lr_scheduler, args.warmup_ratio)
                for parameter_group in optimizer.param_groups:
                    parameter_group['lr'] = current_lr
                for idx, factor in zip(group, factors):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss=model(**batch(train[idx])).loss
                    if not torch.isfinite(loss):
                        raise RuntimeError('训练 loss 非有限值')
                    losses.append(loss.item())
                    weighted_loss = loss * sample_weights[idx]
                    weighted_losses.append(weighted_loss.item())
                    (weighted_loss*factor).backward()
                norm=torch.nn.utils.clip_grad_norm_(params,1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError('梯度非有限值')
                optimizer.step()
                step+=1
                record(dict(event='train_step',epoch=epoch+1,step=step,
                            mean_sample_loss=sum(losses)/len(losses),
                            weighted_mean_sample_loss=sum(weighted_losses)/len(weighted_losses),
                            optimization_group_loss=sum(loss * factor for loss, factor in zip(weighted_losses, factors)),
                            learning_rate=current_lr, supervised_tokens=sum(target_counts[i] for i in group),
                            grad_norm=float(norm)))
                epoch_update = offset // args.gradient_accumulation + 1
                if args.save_half_epoch and epoch_update == max(1, updates_per_epoch // 2) and epoch_update < updates_per_epoch:
                    midpoint_loss = validate()
                    checkpoint = target / 'checkpoints' / f'epoch-{epoch + 1}-half'
                    model.save_pretrained(checkpoint, safe_serialization=True, save_embedding_layers=False)
                    record(dict(event='half_epoch_checkpoint', epoch=epoch + 1, step=step,
                                validation_loss=midpoint_loss, adapter_path=str(checkpoint)))
                    model.train()
            val=validate()
            record(dict(event='validation_epoch',epoch=epoch+1,loss=val))
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        adapter=target/'adapter'
        model.save_pretrained(adapter, safe_serialization=True, save_embedding_layers=False)
        assert step==math.ceil(len(train)/args.gradient_accumulation)*args.epochs
        summary=dict(status='completed',optimizer_steps=step,train_samples=len(train),
                     validation_samples=len(validation),validation_loss_before=before,
                     validation_loss_after=val,elapsed_seconds_including_validation=elapsed,
                     peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                     peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                     adapter_path=str(adapter),note='loss 不是回答质量评估；未运行固定测试集，未声称微调提升。')
        save_json('summary.json',summary)
        save_json('status.json',dict(status='completed'))
        record(summary)
    except Exception as exc:
        save_json('status.json',dict(status='failed',error=f'{type(exc).__name__}: {exc}'))
        raise
    finally:
        log.close()


if __name__=='__main__':
    main()
