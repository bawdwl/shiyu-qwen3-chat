"""Three paired seeds on a known development failure, not a new benchmark."""
import argparse
import contextlib
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_role_data import audit_release_dataset, training_common_prefix
from training_data import read_records, file_hash
from runtime_models import get_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    data = Path(args.data_dir)
    audit_release_dataset(data)
    suite = json.loads((data / 'development.json').read_text(encoding='utf-8'))
    case = next(c for c in suite['cases'] if c['id'] == 'dev-03')
    if len(case['user_turns']) != 1:
        raise ValueError('Expected one turn')
    system = training_common_prefix(read_records(data / 'train.jsonl')) + suite['roles'][case['role_id']]
    messages = [dict(role='system', content=system), dict(role='user', content=case['user_turns'][0])]
    settings = json.loads((data / 'evaluation-protocol.json').read_text(encoding='utf-8'))['generation']
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    import torch
    from peft import PeftModel
    from inference import load_model
    profile = get_profile('qwen3-1.7b')
    tokenizer, base = load_model(model_id=profile['model_id'], revision=profile['revision'], model_path=profile['local_path'])
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False).eval()
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           enable_thinking=False, return_tensors='pt', return_dict=True).to('cuda:0')
    rows = []
    for mode in ('base', 'lora'):
        for seed in (242, 1242, 2242):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            with (model.disable_adapter() if mode == 'base' else contextlib.nullcontext()):
                with torch.inference_mode():
                    generated = model.generate(**inputs, **settings, pad_token_id=tokenizer.eos_token_id)
            tokens = generated[0, inputs['input_ids'].shape[-1]:]
            row = dict(mode=mode, seed=seed, answer=tokenizer.decode(tokens, skip_special_tokens=True),
                       generated_tokens=len(tokens), reached_token_limit=len(tokens) >= settings['max_new_tokens'])
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report = dict(case_id='dev-03', messages=messages, settings=settings, rows=rows,
                  adapter_sha256=file_hash(Path(args.adapter) / 'adapter_model.safetensors'),
                  input_ids_sha256=hashlib.sha256(json.dumps(inputs['input_ids'].tolist()).encode()).hexdigest(),
                  expected='29元', limitation='One known development arithmetic question, three paired seeds; no population accuracy or causal proof. Base uses disabled Adapter on the same loaded model.')
    (root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
