"""Raw fixed-template evaluation; dev and final use separate frozen files."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from release_role_data import audit_release_dataset, training_common_prefix
from runtime_models import get_profile
from training_data import file_hash, read_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--split', choices=['development', 'final'], default='development')
    parser.add_argument('--adapter', default=None)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--conditioning', choices=['card', 'training'], default='card',
                        help='training is a development-only control using actual frozen common rules')
    args = parser.parse_args()
    if args.conditioning != 'card' and args.split != 'development':
        parser.error('Conditioning control is development-only; final protocol remains frozen')
    data = Path(args.data_dir)
    audit = audit_release_dataset(data)
    suite_path = data / ('development.json' if args.split == 'development' else 'final-evaluation.json')
    suite = json.loads(suite_path.read_text(encoding='utf-8'))
    protocol = json.loads((data / 'evaluation-protocol.json').read_text(encoding='utf-8'))
    prefix = ''
    if args.conditioning == 'training':
        prefix = training_common_prefix(read_records(data / 'train.jsonl'))
        if prefix != training_common_prefix(read_records(data / 'validation.jsonl')):
            raise ValueError('Training and validation common rules differ')
    prompt_condition = dict(mode=args.conditioning, prefix=prefix,
                            prefix_sha256=hashlib.sha256(prefix.encode('utf-8')).hexdigest(),
                            source='frozen train/validation prompts' if prefix else 'frozen suite role card only')
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    import torch
    from inference import load_model
    profile = get_profile('qwen3-1.7b')
    torch.manual_seed(42)
    tokenizer, model = load_model(model_id=profile['model_id'], revision=profile['revision'], model_path=profile['local_path'])
    adapter_hash = None
    if args.adapter:
        from peft import PeftModel
        adapter = Path(args.adapter)
        metadata = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
        if metadata['r'] != 8 or metadata['peft_type'] != 'LORA':
            raise ValueError('Unexpected Adapter metadata')
        adapter_hash = file_hash(adapter / 'adapter_model.safetensors')
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    settings = protocol['generation']
    results = []
    with (root / 'answers.jsonl').open('w', encoding='utf-8') as output:
        for case_index, case in enumerate(suite['cases']):
            system_prompt = prefix + suite['roles'][case['role_id']]
            messages = [dict(role='system', content=system_prompt)]
            turns = []
            for turn_index, user in enumerate(case['user_turns']):
                # Pair seeds by case and turn so earlier output lengths cannot
                # consume randomness needed by later, otherwise identical tests.
                seed = 42 + case_index * 100 + turn_index
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                messages.append(dict(role='user', content=user))
                inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                       enable_thinking=False, return_tensors='pt', return_dict=True).to('cuda:0')
                if inputs['input_ids'].shape[-1] + settings['max_new_tokens'] > 2048:
                    raise ValueError('Evaluation context budget exceeded; no silent trimming')
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs, **settings,
                                               pad_token_id=tokenizer.eos_token_id)
                torch.cuda.synchronize()
                tokens = generated[0, inputs['input_ids'].shape[-1]:]
                answer = tokenizer.decode(tokens, skip_special_tokens=True)
                turns.append(dict(user=user, answer=answer, input_tokens=inputs['input_ids'].shape[-1],
                                  seed=seed,
                                  generated_tokens=len(tokens), elapsed_seconds=time.perf_counter()-start,
                                  reached_token_limit=len(tokens) >= settings['max_new_tokens']))
                messages.append(dict(role='assistant', content=answer))
            result = dict(id=case['id'], role_id=case['role_id'], system_prompt=system_prompt, turns=turns)
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
            output.flush()
            results.append(result)
            print(json.dumps(dict(completed=len(results), total=len(suite['cases']), id=case['id'])), flush=True)
    summary = dict(status='raw_generation_completed_not_quality_scored', model=profile,
                   adapter=args.adapter, adapter_sha256=adapter_hash, split=args.split,
                   suite_sha256=file_hash(suite_path), manifest_sha256=audit['manifest_sha256'],
                   protocol=protocol, cases=len(results), turns=sum(len(r['turns']) for r in results),
                   prompt_condition=prompt_condition,
                   note='No application retrieval/summary/postprocessing; generation time excludes loading. One paired seed per case/turn is not a robustness benchmark.')
    (root / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
