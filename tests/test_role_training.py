"""Guard against learning system/history tokens or removing system during trimming."""
import copy
import unittest

from training_data import encode_sample


class RecordingTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def convert_tokens_to_ids(self, token):
        return 999

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        return [100] * (sum(len(m['content']) + 2 for m in messages) + 3)


class RoleTrainingTests(unittest.TestCase):
    def test_system_and_history_are_masked_and_source_is_not_mutated(self):
        row = dict(id='case', messages=[dict(role='user', content='问题'),
                                        dict(role='assistant', content='答案')])
        before = copy.deepcopy(row)
        tokenizer = RecordingTokenizer()
        encoded = encode_sample(row, tokenizer, system_prompt='角色提示')
        self.assertEqual(tokenizer.calls[0][0], dict(role='system', content='角色提示'))
        self.assertEqual(encoded['labels'][-3:], [ord('答'), ord('案'), 999])
        self.assertTrue(all(t == -100 for t in encoded['labels'][:-3]))
        self.assertEqual(row, before)
        self.assertNotIn('答案', [m['content'] for m in tokenizer.calls[0]])

    def test_trimming_preserves_system_and_latest_question(self):
        row = dict(id='long', messages=[dict(role='user', content='旧的问题'),
                                        dict(role='assistant', content='旧的答案'),
                                        dict(role='user', content='最新问题'),
                                        dict(role='assistant', content='回答')])
        tokenizer = RecordingTokenizer()
        encoded = encode_sample(row, tokenizer, max_length=20, trim_history=True, system_prompt='角色')
        self.assertEqual(encoded['history_pairs_trimmed'], 1)
        self.assertEqual(tokenizer.calls[-1], [dict(role='system', content='角色'),
                                              dict(role='user', content='最新问题')])
        self.assertLessEqual(len(encoded['input_ids']), 20)

    def test_oversize_system_and_current_turn_are_rejected(self):
        row = dict(id='large', messages=[dict(role='user', content='问题'),
                                         dict(role='assistant', content='回答')])
        with self.assertRaisesRegex(ValueError, '超过长度上限'):
            encode_sample(row, RecordingTokenizer(), max_length=20, trim_history=True, system_prompt='长' * 30)

    def test_empty_or_nontext_system_is_rejected(self):
        row = dict(id='case', messages=[dict(role='user', content='问题'),
                                        dict(role='assistant', content='回答')])
        for prompt in ('', '  ', 3):
            with self.assertRaisesRegex(ValueError, 'system_prompt'):
                encode_sample(row, RecordingTokenizer(), system_prompt=prompt)

    def test_legacy_prefix_is_unchanged_without_system(self):
        row = dict(id='case', messages=[dict(role='user', content='问题'),
                                        dict(role='assistant', content='回答')])
        tokenizer = RecordingTokenizer()
        encoded = encode_sample(row, tokenizer)
        self.assertEqual(tokenizer.calls[0], [row['messages'][0]])
        self.assertEqual(sum(t != -100 for t in encoded['labels']), 3)

    def test_literal_dialogue_boundaries_are_rejected_in_both_roles(self):
        for marker in ('<|im_start|>', '<|im_end|>', '<|endoftext|>'):
            for role_index in (0, 1):
                with self.subTest(marker=marker, role_index=role_index):
                    row = dict(id='control', messages=[dict(role='user', content='问题'),
                                                       dict(role='assistant', content='回答')])
                    row['messages'][role_index]['content'] += marker
                    with self.assertRaisesRegex(ValueError, '控制标记'):
                        encode_sample(row, RecordingTokenizer())

    def test_template_normalized_newlines_are_not_silently_trained(self):
        for answer in ('\n你好', '\r\n你好'):
            with self.subTest(answer=answer):
                row = dict(id='newline', messages=[dict(role='user', content='问题'),
                                                   dict(role='assistant', content=answer)])
                with self.assertRaisesRegex(ValueError, '换行开头'):
                    encode_sample(row, RecordingTokenizer())

    def test_thinking_blocks_are_not_accepted_as_visible_targets(self):
        for answer in ('<think>推理</think>回答', '推理</think>回答'):
            with self.subTest(answer=answer):
                row = dict(id='thinking', messages=[dict(role='user', content='问题'),
                                                    dict(role='assistant', content=answer)])
                with self.assertRaisesRegex(ValueError, 'thinking'):
                    encode_sample(row, RecordingTokenizer())


if __name__ == '__main__':
    unittest.main()
