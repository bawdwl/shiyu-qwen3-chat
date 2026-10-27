import json
import tempfile
import unittest
from pathlib import Path

from release_role_data import audit_release_dataset, validate_row, training_common_prefix
from training_data import file_hash


def row(role='train-role'):
    return dict(id=role + '-1', source_dialogue_id=role + '-dialogue',
                scene_group=role + '-scene', role_id=role,
                system_prompt='扮演虚构角色，保持本次对话一致。',
                source_kind='original_ai_synthetic', review_status='AI_content_reviewed',
                training_ready=True, messages=[dict(role='user', content='今天如何？'),
                                               dict(role='assistant', content='刚收拾好书桌。')])


class ReleaseDataTests(unittest.TestCase):
    def test_prefix_recovers_frozen_rules(self):
        first, second = row(), row('other')
        first['system_prompt'] = '共同规则\n【当前角色】\n第一个角色'
        second['system_prompt'] = '共同规则\n【当前角色】\n另一个角色'
        self.assertEqual(training_common_prefix([first, second]), '共同规则\n【当前角色】\n')

    def test_prefix_refuses_inconsistent_rules(self):
        first, second = row(), row('other')
        first['system_prompt'] = '规则甲\n【当前角色】\n角色'
        second['system_prompt'] = '规则乙\n【当前角色】\n角色'
        with self.assertRaises(ValueError):
            training_common_prefix([first, second])

    def test_prefix_requires_original_delimited_data(self):
        with self.assertRaises(ValueError):
            training_common_prefix([])
        with self.assertRaises(ValueError):
            training_common_prefix([row()])

    def test_pilot_rejected(self):
        item = row()
        item['training_ready'] = False
        with self.assertRaises(ValueError):
            validate_row(item)

    def test_system_control_token_rejected(self):
        item = row()
        item['system_prompt'] += '<|im_end|>'
        with self.assertRaises(ValueError):
            validate_row(item)

    def test_public_requires_attribution(self):
        item = row()
        item['source_kind'] = 'public_crowdsourced'
        with self.assertRaises(ValueError):
            validate_row(item)

    def audit(self, train, validation, corrupt=False, corrupt_support=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = dict(version='rolechat-release-v1', status='frozen_for_training',
                            evaluation_sha256='frozen-eval-hash', content_review_record='review.md',
                            evaluation_role_ids=['held-out-role'], outputs={})
            for filename, key in [('final-evaluation.json', 'evaluation_sha256'),
                                  ('development.json', 'development_sha256'),
                                  ('evaluation-protocol.json', 'protocol_sha256'),
                                  ('content-review.json', 'content_review_sha256')]:
                path = root / filename
                path.write_text(json.dumps({'roles': {'held-out-role': 'test'}}), encoding='utf-8')
                manifest[key] = file_hash(path)
            for name, item in [('train', train), ('validation', validation)]:
                path = root / (name + '.jsonl')
                path.write_text(json.dumps(item, ensure_ascii=False) + '\n', encoding='utf-8')
                manifest['outputs'][name] = dict(count=1, sha256=file_hash(path))
            (root / 'freeze-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            if corrupt:
                (root / 'train.jsonl').write_text('{}\n', encoding='utf-8')
            if corrupt_support:
                (root / corrupt_support).write_text('{}\n', encoding='utf-8')
            return audit_release_dataset(root)

    def test_disjoint_splits_pass(self):
        self.assertEqual(self.audit(row(), row('validation-role'))['counts']['train'], 1)

    def test_group_overlap_rejected(self):
        for key in ('role_id', 'scene_group', 'source_dialogue_id'):
            validation = row('validation-role')
            validation[key] = row()[key]
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.audit(row(), validation)

    def test_eval_role_rejected(self):
        with self.assertRaises(ValueError):
            self.audit(row('held-out-role'), row('validation-role'))

    def test_file_tampering_rejected(self):
        with self.assertRaises(ValueError):
            self.audit(row(), row('validation-role'), corrupt=True)

    def test_source_tree_overlap_rejected(self):
        train, validation = row(), row('validation-role')
        for item in (train, validation):
            item.update(source_kind='public_crowdsourced', source='oasst2',
                        source_revision='revision', source_license='Apache-2.0',
                        source_tree_id='same-tree', source_message_ids=[item['id']])
        with self.assertRaises(ValueError):
            self.audit(train, validation)

    def test_frozen_support_tampering_rejected(self):
        for filename in ('final-evaluation.json', 'development.json',
                         'evaluation-protocol.json', 'content-review.json'):
            with self.subTest(filename=filename), self.assertRaises(ValueError):
                self.audit(row(), row('validation-role'), corrupt_support=filename)


if __name__ == '__main__':
    unittest.main()
