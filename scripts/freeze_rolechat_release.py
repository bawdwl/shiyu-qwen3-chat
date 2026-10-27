"""Build the first bounded, AI-reviewed role-conditioned adaptation dataset.

Public text stays in ignored outputs. Original sources and review are tracked.
No model weights are loaded; tokenizer audits must pass before manifest exists.
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from release_role_data import audit_release_dataset, encode_release_sample
from training_data import file_hash, read_records


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    source = Path('data/rolechat-release-v1')
    pilot = json.loads((source / 'original-pilot.json').read_text(encoding='utf-8'))
    expansion = json.loads((source / 'original-expansion.json').read_text(encoding='utf-8'))
    evaluation = json.loads((source / 'evaluation-draft.json').read_text(encoding='utf-8'))
    roles = dict(pilot['roles'], **expansion['validation_roles'])
    validation_ids = set(expansion['validation_roles'])
    splits = dict(train=[], validation=[])
    dialogue_ids, scene_groups = set(), set()
    for dialogue in pilot['dialogues'] + expansion['dialogues']:
        if dialogue['id'] in dialogue_ids or dialogue['scene_group'] in scene_groups:
            raise ValueError('Duplicate dialogue/scene')
        dialogue_ids.add(dialogue['id'])
        scene_groups.add(dialogue['scene_group'])
        rid = dialogue['role_id']
        split = 'validation' if rid in validation_ids else 'train'
        prompt = pilot['common_rules'] + '\n【当前角色】\n' + roles[rid]
        messages = []
        for i, (user, assistant) in enumerate(dialogue['turns'], 1):
            messages.extend([dict(role='user', content=user), dict(role='assistant', content=assistant)])
            splits[split].append(dict(id=f"{dialogue['id']}-turn-{i}",
                                     source_dialogue_id=dialogue['id'], scene_group=dialogue['scene_group'],
                                     role_id=rid, category=dialogue['category'], system_prompt=prompt,
                                     source_kind='original_ai_synthetic',
                                     review_status='AI_content_reviewed_not_human_rated',
                                     training_ready=True, messages=list(messages)))
    public_path = Path('outputs/data-candidates/oasst2-v1/train-candidates.jsonl')
    public = read_records(public_path)
    accepted = {
        'oasst2_008ee03f-5388-44b5-93f8-43c350e27252': 'Kinship relation checked: maternal aunt son is maternal cousin; seniority by age.',
        'oasst2_a90b824b-e9fb-4a66-bf5c-14e3b3efaf7f': 'SymPy Max/Min example reviewed; symbolic extrema remain symbolic until substitution.',
        'oasst2_539c9bbf-83c0-4e48-9f1b-dc3e2031ed90': 'Recursive quicksort partition reviewed; bounded numeric test cases checked separately.'
    }
    selected = [row for row in public if row['id'] in accepted]
    if len(selected) != len(accepted):
        raise ValueError('Missing reviewed public source')
    for item in selected:
        item = dict(item)
        item.update(source_dialogue_id='oasst2-' + item['source_tree_id'],
                    scene_group='oasst2-' + item['source_tree_id'], role_id='general-helper-train',
                    system_prompt='用中文准确回答用户问题，按需要解释，不能确定的内容明确说明。',
                    source_kind='public_crowdsourced', review_status='AI_content_reviewed_not_human_rated',
                    training_ready=True)
        splits['train'].append(item)
    # These are dev prompts only, not references or final test. Never train them.
    development = dict(status='frozen_development_only', roles=expansion['validation_roles'], cases=[
        dict(id='dev-01', role_id='heyu', user_turns=['今天做的饭太咸了，我只是想抱怨一句，不用教我补救。']),
        dict(id='dev-02', role_id='heyu', user_turns=['你最近有没有做一个让自己挺满意的小物件？','我今天换了新鞋。','你刚说做的东西有什么特点？']),
        dict(id='dev-03', role_id='heyu', user_turns=['我买了三支笔，每支七元，付了五十元，应该找回多少？']),
        dict(id='dev-04', role_id='heyu', user_turns=['你平常工作会接触什么材料？']),
        dict(id='dev-05', role_id='chenmo', user_turns=['我把见面改到周日下午四点，原来周六晚上不去了。','最终哪天几点？']),
        dict(id='dev-06', role_id='chenmo', user_turns=['今晚不聊了，别再接着问，晚安。']),
        dict(id='dev-07', role_id='chenmo', user_turns=['你知道我现在拿着的书叫什么吗？没有照片，也没告诉你书名。']),
        dict(id='dev-08', role_id='chenmo', user_turns=['Python里怎样判断一个列表是不是空的？给个短例子。'])])
    evaluation['status'] = 'frozen_final_test_not_for_training_or_checkpoint_selection'
    evaluation['evaluation_rules']['holdout_policy'] = 'Frozen before formal training. No checkpoint selection or tuning on these cases. Old sealed tests unchanged.'
    # Same prompt rendering for each compared model; no app memory or cleanup.
    protocol = dict(comparison='base versus development-selected half/final Adapter',
                    template='Qwen tokenizer apply_chat_template, enable_thinking=False',
                    generation=dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20,
                                    min_p=0.0, max_new_tokens=192, repetition_penalty=1.0),
                    seed_policy='42 + case_index * 100 + turn_index; paired reset before each turn',
                    context='all turns in each short case; reset between cases; no retrieval or dynamic summary',
                    selection='validation loss plus 8 dev cases; final test never selects checkpoints',
                    rubric=dict(naturalness='0 awkward/off-topic, 1 usable but canned, 2 natural and specific',
                                context='0 contradiction/missed correction, 1 incomplete, 2 consistent',
                                persona='0 fixed identity conflict, 1 weak/unclear, 2 consistent with card',
                                intent='0 violates explicit request, 1 partial, 2 follows request',
                                general='0 incorrect, 1 partial, 2 correct; nonapplicable is null'),
                    scoring='AI diagnosis and actual author anonymous preferences reported separately; no fabricated human ratings',
                    failure_rules='report empty/truncated outputs; spontaneous detail not established is separate, not memory success',
                    claim='Small synthetic adaptation experiment, not a RoleLLM paper-score reproduction.')
    save(root / 'final-evaluation.json', evaluation)
    save(root / 'development.json', development)
    save(root / 'evaluation-protocol.json', protocol)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('outputs/models/qwen3-1.7b', local_files_only=True)
    metrics, outputs = {}, {}
    for split, rows in splits.items():
        encoded = [encode_release_sample(row, tokenizer, 1024) for row in rows]
        eos = tokenizer.convert_tokens_to_ids('<|im_end|>')
        if any(item['labels'][-1] != eos for item in encoded):
            raise ValueError('Incorrect target EOS')
        lengths = [len(item['input_ids']) for item in encoded]
        targets = [sum(x != -100 for x in item['labels'][1:]) for item in encoded]
        metrics[split] = dict(targets=len(rows), dialogues=len({r['source_dialogue_id'] for r in rows}),
                              roles=len({r['role_id'] for r in rows}),
                              input_tokens=dict(min=min(lengths), median=statistics.median(lengths), max=max(lengths)),
                              supervised_tokens=dict(total=sum(targets), median=statistics.median(targets), max=max(targets)),
                              categories=dict(Counter(r.get('category', 'general') for r in rows)),
                              source_kinds=dict(Counter(r['source_kind'] for r in rows)))
        path = root / (split + '.jsonl')
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        outputs[split] = dict(count=len(rows), sha256=file_hash(path))
    review = dict(reviewer='Codex AI, not independent human',
                  scope='Read original dialogue content; reviewed three public answers; mechanical template audit',
                  accepted_public=accepted, public_source_candidate_file_sha256=file_hash(public_path),
                  limitations=['Synthetic majority; small scale; no real user distribution claim',
                               'Only three public helper targets, insufficient to guarantee broad capability retention',
                               'Short dialogues do not establish long-context robustness',
                               'Exact split checks cannot prove semantic independence'],
                  source_hashes={p.name: file_hash(p) for p in source.glob('*.json')}, metrics=metrics)
    save(root / 'content-review.json', review)
    manifest = dict(version='rolechat-release-v1', status='frozen_for_training',
                    evaluation_sha256=file_hash(root / 'final-evaluation.json'),
                    protocol_sha256=file_hash(root / 'evaluation-protocol.json'),
                    development_sha256=file_hash(root / 'development.json'),
                    content_review_record='content-review.json', content_review_sha256=file_hash(root / 'content-review.json'),
                    evaluation_role_ids=list(evaluation['roles']), outputs=outputs,
                    limitation='First small controlled synthetic experiment; not a mature production model dataset.')
    save(root / 'freeze-manifest.json', manifest)
    report = audit_release_dataset(root)
    save(root / 'data-audit.json', dict(**report, metrics=metrics))
    print(json.dumps(dict(**report, metrics=metrics), ensure_ascii=False))


if __name__ == '__main__':
    main()
