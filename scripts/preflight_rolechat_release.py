"""Temporary GPU capacity probe on original pilot; never saves an Adapter/model."""
import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_models import get_profile
from training_data import encode_sample, file_hash, read_records
from training_recipe import LORA_MODULES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot-audit-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stress-tokens', type=int, default=1024)
    args = parser.parse_args()
    if not 128 <= args.stress_tokens <= 2048:
        parser.error('stress-tokens must be 128–2048 for this bounded probe')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    import torch
    from peft import LoraConfig, get_peft_model
    from inference import load_model

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 is required')
    free, total = torch.cuda.mem_get_info(0)
    if free < 6 * 1024**3:
        raise RuntimeError('Less than 6 GiB free GPU memory; probe stopped before model loading')
    profile = get_profile('qwen3-1.7b')
    audit_root = Path(args.pilot_audit_dir)
    audit = json.loads((audit_root / 'audit.json').read_text(encoding='utf-8'))
    if file_hash('data/rolechat-release-v1/original-pilot.json') != audit['source_sha256']:
        raise ValueError('Pilot source changed after token audit')
    rows = read_records(audit_root / 'pilot-targets.jsonl')
    torch.manual_seed(42)
    tokenizer, base = load_model(model_id=profile['model_id'], revision=profile['revision'], model_path=profile['local_path'])
    samples = [encode_sample(r, tokenizer, max_length=args.stress_tokens, system_prompt=r['system_prompt']) for r in rows]
    actual = max(samples, key=lambda s: len(s['input_ids']))
    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16,
                                          lora_dropout=0.05, target_modules=sorted(LORA_MODULES), bias='none'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-5, weight_decay=0.01)
    stress = {key: list(value) for key, value in actual.items() if key in ('input_ids', 'attention_mask', 'labels')}
    padding = args.stress_tokens - len(stress['input_ids'])
    filler = tokenizer.encode('这是一段仅用于显存形状测试的文本。', add_special_tokens=False)
    stress['input_ids'] = (filler * math.ceil(padding / len(filler)))[:padding] + stress['input_ids']
    stress['attention_mask'] = [1] * padding + stress['attention_mask']
    stress['labels'] = [-100] * padding + stress['labels']
    report = dict(kind='capacity_only_temporary_updates', model=profile, seed=42,
                  source_sha256=audit['source_sha256'], recipe=dict(r=8, alpha=16, dropout=0.05,
                  target_modules=sorted(LORA_MODULES), dtype='bfloat16', learning_rate=5e-5),
                  trainable_parameters=sum(p.numel() for p in parameters),
                  free_memory_before_gib=free / 1024**3, total_memory_gib=total / 1024**3,
                  tests=[], adapter_saved=False,
                  limitation='Synthetic stress prefix is not training data. These updates are discarded; no quality or training-completion claim.')
    for label, sample in [('longest_actual_pilot', actual), ('synthetic_shape_stress', stress)]:
        batch = {k: torch.tensor([sample[k]], device='cuda:0') for k in ('input_ids', 'attention_mask', 'labels')}
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(**batch).loss
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite probe loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite probe gradient')
        optimizer.step()
        torch.cuda.synchronize()
        report['tests'].append(dict(name=label, input_tokens=len(sample['input_ids']),
                                   supervised_tokens=sum(t != -100 for t in sample['labels'][1:]),
                                   finite_loss=float(loss.detach()), finite_gradient=True,
                                   elapsed_seconds=time.perf_counter()-start,
                                   peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                                   peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3))
    (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
