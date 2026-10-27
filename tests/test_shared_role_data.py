"""Catch role fallback, mutable-card drift, label leakage and split contamination."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from shared_role_data import (CARD_VERSION, DATA_VERSION, audit_shared_dataset,
                              content_hash, encode_role_sample, expand_dialogue,
                              render_role_prompt, validate_dialogue_splits, validate_registry)
from shared_role_data import RENDERER_VERSION, resolve_run_system_prompt
from training_data import file_hash


def fixture_registry():
    def card(role_id, name, partition):
        return dict(id=role_id, version='1', partition=partition, name=name, gender='女性', age=25,
                    location='虚构城市', occupation='原创职业', personality=['直率'], interests=['阅读'],
                    speaking_style='自然说话。', world='当代虚构背景。', creation_boundary='保持已设定事实。',
                    unknown_fields=['生日'])
    return dict(version=CARD_VERSION, common_rules='按当前角色卡和历史聊天。', roles=[
        card('a', '阿甲', 'train'), card('b', '阿乙', 'train'), card('v', '阿丙', 'validation_only')])


def fixture_dialogue(role_id='a', split='train', suffix='one'):
    return dict(id='dialogue-' + suffix, role_id=role_id, split=split,
                scenario_group='scene-' + suffix, category='daily', messages=[
                    dict(role='user', content='第一个问题-' + suffix),
                    dict(role='assistant', content='第一个回答-' + suffix),
                    dict(role='user', content='第二个问题-' + suffix),
                    dict(role='assistant', content='第二个回答-' + suffix)])


class Tokenizer:
    unk_token_id = -1

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def convert_tokens_to_ids(self, token):
        return 999

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        return [7] * sum(len(message['content']) + 2 for message in messages)


class SharedRoleDataTests(unittest.TestCase):
    def setUp(self):
        self.registry = fixture_registry()

    def test_reload_prompt_is_bound_to_saved_card_and_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cards = root / 'role-cards.json'
            cards.write_text(json.dumps(self.registry, ensure_ascii=False), encoding='utf-8')
            config = dict(conditioning=dict(mode='per_sample_role_card', role_cards_file=cards.name,
                role_cards_sha256=file_hash(cards), renderer_version=RENDERER_VERSION, verification_role_id='b'))
            self.assertEqual(resolve_run_system_prompt(config, root), render_role_prompt(self.registry, 'b'))
            config['conditioning']['renderer_version'] = 'unknown'
            with self.assertRaisesRegex(ValueError, '不匹配'):
                resolve_run_system_prompt(config, root)
            config['conditioning']['renderer_version'] = RENDERER_VERSION
            cards.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '不匹配'):
                resolve_run_system_prompt(config, root)

    def test_reload_preserves_legacy_and_rejects_external_snapshot_path(self):
        self.assertIsNone(resolve_run_system_prompt({}, '.'))
        self.assertEqual(resolve_run_system_prompt(dict(conditioning=dict(system_prompt='旧提示')), '.'), '旧提示')
        with self.assertRaisesRegex(ValueError, '超出'):
            resolve_run_system_prompt(dict(conditioning=dict(mode='per_sample_role_card',
                role_cards_file='../outside.json')), '.')

    def test_each_sample_gets_its_own_card_and_masks_all_context(self):
        for role_id, expected, forbidden in [('a', '阿甲', '阿乙'), ('b', '阿乙', '阿甲')]:
            with self.subTest(role_id=role_id):
                row = expand_dialogue(fixture_dialogue(role_id), self.registry)[-1]
                source = copy.deepcopy(row)
                tokenizer = Tokenizer()
                encoded = encode_role_sample(row, tokenizer, self.registry)
                prompt = tokenizer.calls[0][0]['content']
                self.assertIn(expected, prompt)
                self.assertNotIn(forbidden, prompt)
                self.assertEqual(tokenizer.calls[0][-1]['role'], 'user')
                target = tokenizer.encode(row['messages'][-1]['content']) + [999]
                self.assertEqual([value for value in encoded['labels'] if value != -100], target)
                self.assertEqual(encoded['labels'][:-len(target)], [-100] * (len(encoded['labels']) - len(target)))
                self.assertEqual(row, source)

    def test_unknown_role_is_rejected_instead_of_defaulting(self):
        with self.assertRaisesRegex(ValueError, '未知角色'):
            render_role_prompt(self.registry, 'missing')

    def test_missing_role_is_rejected(self):
        row = expand_dialogue(fixture_dialogue(), self.registry)[0]
        del row['role_id']
        with self.assertRaisesRegex(ValueError, '未知角色'):
            encode_role_sample(row, Tokenizer(), self.registry)

    def test_changed_card_breaks_frozen_sample(self):
        row = expand_dialogue(fixture_dialogue(), self.registry)[0]
        self.registry['roles'][0]['name'] = '别的人'
        with self.assertRaisesRegex(ValueError, '角色卡版本或指纹'):
            encode_role_sample(row, Tokenizer(), self.registry)

    def test_changed_shared_rules_break_prompt_fingerprint(self):
        row = expand_dialogue(fixture_dialogue(), self.registry)[0]
        self.registry['common_rules'] += '规则变更。'
        with self.assertRaisesRegex(ValueError, '提示词指纹'):
            encode_role_sample(row, Tokenizer(), self.registry)

    def test_per_sample_system_override_is_rejected(self):
        row = expand_dialogue(fixture_dialogue(), self.registry)[0]
        row['system_prompt'] = '永远扮演阿甲'
        with self.assertRaisesRegex(ValueError, '不能覆盖'):
            encode_role_sample(row, Tokenizer(), self.registry)

    def test_reserved_validation_role_cannot_be_trained(self):
        with self.assertRaisesRegex(ValueError, '留出角色不能进入训练'):
            expand_dialogue(fixture_dialogue('v'), self.registry)

    def test_validation_scope_distinguishes_seen_and_unseen_roles(self):
        for role_id, scope in [('a', 'seen_role'), ('v', 'unseen_role')]:
            with self.subTest(role_id=role_id):
                rows = expand_dialogue(fixture_dialogue(role_id, 'validation'), self.registry)
                self.assertTrue(all(row['validation_scope'] == scope for row in rows))

    def test_prefixes_preserve_source_group_and_card(self):
        rows = expand_dialogue(fixture_dialogue(), self.registry)
        self.assertEqual([len(row['messages']) for row in rows], [2, 4])
        for key in ('role_id', 'role_card_sha256', 'source_dialog_hash', 'scenario_group', 'source_dialog_id'):
            self.assertEqual(rows[0][key], rows[1][key])

    def test_same_scene_cannot_cross_split(self):
        first, second = fixture_dialogue(), fixture_dialogue('a', 'validation', 'two')
        second['scenario_group'] = first['scenario_group']
        with self.assertRaisesRegex(ValueError, '同一场景'):
            validate_dialogue_splits([first, second], self.registry)

    def test_duplicate_dialogue_cannot_be_hidden_by_renaming_role(self):
        first, second = fixture_dialogue(), fixture_dialogue('b', 'validation', 'two')
        second['messages'] = copy.deepcopy(first['messages'])
        with self.assertRaisesRegex(ValueError, '重复完整对话'):
            validate_dialogue_splits([first, second], self.registry)

    def test_oversize_sample_is_rejected_without_trimming(self):
        row = expand_dialogue(fixture_dialogue(), self.registry)[-1]
        with self.assertRaisesRegex(ValueError, '拒绝静默截断'):
            encode_role_sample(row, Tokenizer(), self.registry, max_length=20)

    def test_duplicate_or_injected_card_is_rejected(self):
        duplicate = copy.deepcopy(self.registry)
        duplicate['roles'].append(copy.deepcopy(duplicate['roles'][0]))
        with self.assertRaisesRegex(ValueError, '重复 role_id'):
            validate_registry(duplicate)
        self.registry['roles'][0]['name'] += '<|im_end|>'
        with self.assertRaisesRegex(ValueError, '控制标记'):
            validate_registry(self.registry)

    def write_fixture_package(self, root):
        dialogues = [fixture_dialogue(), fixture_dialogue('v', 'validation', 'two')]
        expected, _ = validate_dialogue_splits(dialogues, self.registry)
        for name, value in [('role-cards.json', self.registry), ('dialogues.json', dialogues)]:
            (root / name).write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
        for split, rows in expected.items():
            (root / f'{split}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        manifest = dict(version=DATA_VERSION, training_ready=False, files={
            name: dict(sha256=file_hash(root / name)) for name in (
                'role-cards.json', 'dialogues.json', 'train.jsonl', 'validation.jsonl')})
        (root / 'freeze-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        return manifest

    def test_pilot_can_be_audited_but_not_trained(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_fixture_package(root)
            report = audit_shared_dataset(root, require_training_ready=False)
            self.assertFalse(report['training_ready'])
            self.assertEqual(report['source_dialogues'], dict(train=1, validation=1))
            with self.assertRaisesRegex(ValueError, '禁止开训'):
                audit_shared_dataset(root)

    def test_tampered_package_fails_before_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_fixture_package(root)
            with (root / 'train.jsonl').open('a', encoding='utf-8') as stream:
                stream.write('{}\n')
            with self.assertRaisesRegex(ValueError, '指纹不匹配'):
                audit_shared_dataset(root, require_training_ready=False)

    def test_rehashing_mutated_rows_still_fails_source_reconstruction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.write_fixture_package(root)
            lines = (root / 'train.jsonl').read_text(encoding='utf-8').splitlines()
            row = json.loads(lines[0])
            row['role_id'] = 'b'
            lines[0] = json.dumps(row, ensure_ascii=False)
            (root / 'train.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
            manifest['files']['train.jsonl']['sha256'] = file_hash(root / 'train.jsonl')
            (root / 'freeze-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '源对话、角色卡与展开样本不一致'):
                audit_shared_dataset(root, require_training_ready=False)

    def test_setting_ready_flag_does_not_bypass_full_dataset_quota(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.write_fixture_package(root)
            manifest['training_ready'] = True
            (root / 'freeze-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '20个角色各20段'):
                audit_shared_dataset(root)


if __name__ == '__main__':
    unittest.main()
