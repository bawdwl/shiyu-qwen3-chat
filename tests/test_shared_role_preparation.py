"""Exercise incremental source handling without loading model weights."""
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from prepare_shared_role_data import (load_additional_dialogues, load_pilot_dialogues,
                                      surface_profile, validate_build_counts)
from shared_role_data import load_registry, validate_dialogue_splits

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/shared-role-chat-v1'


class SharedRolePreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = load_registry(SOURCE / 'role-cards.json')
        cls.initial = load_pilot_dialogues(SOURCE / 'pilot-dialogues.json',
                                          ROOT / 'data/xiaoman-synthetic-v1/pilot-dialogues.json')
        cls.additional = load_additional_dialogues(SOURCE / 'batch-02.json')
        cls.long_context_batch = load_additional_dialogues(SOURCE / 'batch-03.json')
        cls.explanation_batch = load_additional_dialogues(SOURCE / 'batch-04.json')
        cls.last_batches = [load_additional_dialogues(SOURCE / name) for name in ('batch-05.json', 'batch-06.json')]

    def test_full_source_counts_do_not_replace_final_evaluation_gate(self):
        from shared_role_data import audit_shared_dataset
        dialogues = self.initial + self.additional + self.long_context_batch + self.explanation_batch
        dialogues += [row for batch in self.last_batches for row in batch]
        rows, counts = validate_dialogue_splits(dialogues, self.registry)
        validate_build_counts(counts, self.registry, expanded_build=True)
        self.assertEqual(set(counts['train'].values()), {20})
        self.assertEqual(set(counts['validation'].values()), {5})
        self.assertEqual(len(rows['train']), 932)
        self.assertEqual(len(rows['validation']), 126)
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, 'freeze-manifest.json').write_text(json.dumps(dict(
                version='shared-role-chat-v1', training_ready=False)), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '禁止开训'):
                audit_shared_dataset(directory, require_training_ready=True)

    def test_surface_profile_counts_source_turns_once_and_is_not_a_quality_score(self):
        report = surface_profile([dict(split='train', messages=[
            dict(role='user', content='不要追问。'), dict(role='assistant', content='好。'),
            dict(role='user', content='再确认一下。'), dict(role='assistant', content='好。')])])
        train = report['splits']['train']
        self.assertEqual(train['assistant_turns'], 2)
        self.assertEqual(train['dialogues_with_constraint_cue'], 1)
        self.assertEqual(train['user_turns_with_constraint_cue'], 1)
        self.assertEqual(train['exact_duplicate_answers'], [dict(text='好。', count=2)])
        self.assertEqual(report['splits']['validation']['assistant_turns'], 0)
        self.assertNotIn('quality_score', report)

    def test_xiaoman_examples_respect_card_limit_on_followup_questions(self):
        dialogues = self.initial + self.additional + self.long_context_batch + self.explanation_batch
        dialogues += [row for batch in self.last_batches for row in batch]
        for row in dialogues:
            if row['role_id'] == 'xiaoman':
                for message in row['messages']:
                    if message['role'] == 'assistant':
                        with self.subTest(dialogue=row['id']):
                            self.assertLessEqual(message['content'].count('?') + message['content'].count('？'), 1)

    def test_fourth_batch_keeps_all_roles_balanced_and_validation_separate(self):
        expanded, counts = validate_dialogue_splits(
            self.initial + self.additional + self.long_context_batch + self.explanation_batch, self.registry)
        train_ids, heldout_ids = validate_build_counts(counts, self.registry, expanded_build=True)
        self.assertEqual(set(counts['train'].values()), {12})
        self.assertEqual(set(counts['validation'].values()), {4})
        self.assertFalse(heldout_ids & set(counts['train']))
        self.assertEqual({row['role_id'] for row in expanded['train']}, train_ids)
        self.assertEqual(len(expanded['train']) + len(expanded['validation']), 678)

    def test_third_batch_is_cumulative_and_balanced_without_replacing_prior_sources(self):
        expanded, counts = validate_dialogue_splits(self.initial + self.additional + self.long_context_batch, self.registry)
        validate_build_counts(counts, self.registry, expanded_build=True)
        self.assertEqual(set(counts['train'].values()), {8})
        self.assertEqual(set(counts['validation'].values()), {3})
        self.assertEqual(len(expanded['train']), 413)
        self.assertEqual(len(expanded['validation']), 84)

    def test_new_five_turn_dialogues_keep_every_history_prefix(self):
        long_rows = [row for row in self.long_context_batch if row['split'] == 'train' and row['category'] == 'context']
        self.assertEqual(len(long_rows), 20)
        expanded, _ = validate_dialogue_splits(self.initial + self.additional + self.long_context_batch, self.registry)
        for dialogue in long_rows:
            with self.subTest(dialogue=dialogue['id']):
                self.assertEqual(len(dialogue['messages']), 10)
                prefixes = [row for row in expanded['train'] if row['source_dialog_id'] == dialogue['id']]
                self.assertEqual([len(row['messages']) for row in prefixes], [2, 4, 6, 8, 10])
                for prefix in prefixes:
                    self.assertEqual(prefix['messages'], dialogue['messages'][:len(prefix['messages'])])
                    self.assertEqual(prefix['scenario_group'], dialogue['scenario_group'])

    def test_six_turn_limit_accepts_complete_turn_and_rejects_seventh(self):
        rows = copy.deepcopy(self.long_context_batch)
        first = next(row for row in rows if len(row['messages']) == 10)
        first['messages'].extend([dict(role='user', content='只再确认一次当前信息。'),
                                  dict(role='assistant', content='以本段里最后确认的信息为准。')])
        validate_dialogue_splits(self.initial + self.additional + rows, self.registry)
        first['messages'].extend([dict(role='user', content='这是第七轮。'),
                                  dict(role='assistant', content='这轮应因超过预算被拒绝。')])
        with self.assertRaisesRegex(ValueError, '2–6轮'):
            validate_dialogue_splits(self.initial + self.additional + rows, self.registry)

    def test_cumulative_batch_preserves_balanced_train_and_heldout_validation(self):
        rows, counts = validate_dialogue_splits(self.initial + self.additional, self.registry)
        validate_build_counts(counts, self.registry, expanded_build=True)
        self.assertEqual(set(counts['train'].values()), {5})
        self.assertEqual(set(counts['validation'].values()), {2})
        self.assertEqual(sum(counts['train'].values()), 100)
        self.assertEqual(sum(counts['validation'].values()), 20)
        self.assertEqual(len(rows['train']), 233)

    def test_pilot_mode_still_requires_original_quota(self):
        _, counts = validate_dialogue_splits(self.initial, self.registry)
        validate_build_counts(counts, self.registry)
        _, larger = validate_dialogue_splits(self.initial + self.additional, self.registry)
        with self.assertRaisesRegex(ValueError, '试写'):
            validate_build_counts(larger, self.registry)

    def test_importing_a_batch_twice_cannot_silently_duplicate_examples(self):
        with self.assertRaisesRegex(ValueError, '重复'):
            validate_dialogue_splits(self.initial + self.additional + self.additional, self.registry)

    def test_validation_role_cannot_be_reassigned_to_train_in_new_batch(self):
        rows = copy.deepcopy(self.additional)
        row = next(item for item in rows if item['role_id'] == 'xiaolu')
        row['split'] = 'train'
        with self.assertRaisesRegex(ValueError, '留出角色'):
            validate_dialogue_splits(self.initial + rows, self.registry)

    def test_lopsided_train_and_missing_validation_role_are_rejected(self):
        _, original = validate_dialogue_splits(self.initial + self.additional, self.registry)
        counts = copy.deepcopy(original)
        counts['train']['xiaoman'] += 1
        with self.assertRaisesRegex(ValueError, '均衡'):
            validate_build_counts(counts, self.registry, expanded_build=True)
        counts = copy.deepcopy(original)
        del counts['validation']['shenxing']
        with self.assertRaisesRegex(ValueError, '验证'):
            validate_build_counts(counts, self.registry, expanded_build=True)

    def test_preparation_cannot_expand_beyond_frozen_design(self):
        _, counts = validate_dialogue_splits(self.initial + self.additional, self.registry)
        counts['train'] = Counter({key: 21 for key in counts['train']})
        with self.assertRaisesRegex(ValueError, '不超过20'):
            validate_build_counts(counts, self.registry, expanded_build=True)

    def test_each_expanded_source_retains_batch_row_and_turn_order(self):
        row = self.additional[0]
        self.assertEqual(row['id'], 'sr-batch-02-001')
        self.assertEqual(row['source_origin']['row'], 1)
        self.assertEqual(row['source_origin']['batch_id'], 'batch-02')
        self.assertEqual([item['role'] for item in row['messages']], ['user', 'assistant', 'user', 'assistant'])

    def test_wrong_source_version_and_bad_message_shape_are_rejected(self):
        packet = json.loads((SOURCE / 'batch-02.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'batch.json'
            wrong = copy.deepcopy(packet)
            wrong['version'] = 'unknown'
            path.write_text(json.dumps(wrong), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '版本'):
                load_additional_dialogues(path)
            wrong = copy.deepcopy(packet)
            wrong['rows'][0][-1][0] = {'role': 'system', 'content': 'override'}
            path.write_text(json.dumps(wrong), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '文本列表'):
                load_additional_dialogues(path)

    def test_odd_turn_and_control_marker_cannot_pass_source_expansion(self):
        for mutation in ('odd', 'marker'):
            rows = copy.deepcopy(self.additional)
            if mutation == 'odd':
                rows[0]['messages'].pop()
            else:
                rows[0]['messages'][0]['content'] += '<|im_end|>'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_dialogue_splits(self.initial + rows, self.registry)


if __name__ == '__main__':
    unittest.main()
