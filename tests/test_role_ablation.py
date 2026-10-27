import copy
import hashlib
import unittest

from evaluate_role_ablation import summarize, validate_control
from runtime_models import get_profile
from verify_adapter import resolve_profile


class RoleAblationTests(unittest.TestCase):
    def setUp(self):
        self.old = dict(base_model='model', revision='revision', seed=42, epochs=2, micro_batch=1,
                        gradient_accumulation=4, learning_rate=0.0002, max_length=512,
                        r=8, lora_alpha=16, lora_dropout=0.05, target_modules=['q_proj', 'v_proj'],
                        dtype='bfloat16', gradient_checkpointing=True, optimizer='AdamW',
                        weight_decay=0.01, max_grad_norm=1.0, schedule='constant',
                        data=dict(sha256=dict(train='train-hash', validation='val-hash', eval='test-hash')))
        self.new = copy.deepcopy(self.old)
        self.new['conditioning'] = dict(system_prompt='角色提示',
                                        sha256=hashlib.sha256('角色提示'.encode('utf-8')).hexdigest())

    def test_only_conditioning_can_change(self):
        self.assertIn('data.sha256', validate_control(self.old, self.new))
        self.new['learning_rate'] = 0.001
        with self.assertRaisesRegex(ValueError, 'learning_rate'):
            validate_control(self.old, self.new)

    def test_dataset_change_cannot_be_mislabeled_as_prompt_only(self):
        self.new['data']['sha256']['train'] = 'different'
        with self.assertRaisesRegex(ValueError, 'data.sha256'):
            validate_control(self.old, self.new)

    def test_missing_or_corrupted_prompt_snapshot_is_rejected(self):
        for conditioning in ({}, dict(system_prompt='changed', sha256='old')):
            self.new['conditioning'] = conditioning
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                validate_control(self.old, self.new)

    def test_adapter_reload_uses_actual_training_profile(self):
        for name in ('qwen3-0.6b', 'qwen3-1.7b'):
            profile = get_profile(name)
            config = dict(model_profile=name, base_model=profile['model_id'], revision=profile['revision'])
            self.assertEqual(resolve_profile(config), profile)
        config['model_profile'] = 'qwen3-0.6b'
        with self.assertRaisesRegex(ValueError, '不一致'):
            resolve_profile(config)

    def test_short_math_answers_do_not_distort_chat_length_statistics(self):
        rows = [dict(variant='base', kind='chat', metrics=dict(output_tokens=25, ended_with_eos=True)),
                dict(variant='base', kind='capability_regression', metrics=dict(output_tokens=1, ended_with_eos=True))]
        result = summarize(rows)['base']
        self.assertEqual(result['chat']['median_output_tokens'], 25)
        self.assertEqual(result['chat']['replies_le5_tokens'], 0)
        self.assertEqual(result['capability_regression']['median_output_tokens'], 1)


if __name__ == '__main__':
    unittest.main()
