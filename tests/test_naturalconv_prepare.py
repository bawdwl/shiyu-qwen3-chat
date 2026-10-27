import unittest

from prepare_naturalconv import eval_record, role_messages, sft_record, stratified_dialog_ids
from training_data import encode_sample


class TinyTokenizer:
    unk_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def convert_tokens_to_ids(self, token):
        return 999 if token == '<|im_end|>' else self.unk_token_id

    def apply_chat_template(self, messages, **kwargs):
        return list(range(sum(len(message['content']) + 2 for message in messages) + 3))


class NaturalConvPreparationTests(unittest.TestCase):
    def setUp(self):
        self.dialogue = dict(
            dialog_id='12_4',
            document_id=12,
            content=[f'第{i}句' for i in range(20)],
        )

    def test_maps_two_human_speakers_to_alternating_training_roles(self):
        messages = role_messages(self.dialogue, 5)
        self.assertEqual([m['role'] for m in messages],
                         ['user', 'assistant', 'user', 'assistant', 'user', 'assistant'])
        self.assertEqual(messages[-1]['content'], '第5句')
        record = sft_record(self.dialogue, '体育', 5)
        self.assertEqual(record['source_dialog_id'], '12_4')
        self.assertEqual(record['topic'], '体育')

    def test_eval_context_excludes_reference_answer(self):
        record = eval_record(self.dialogue, '体育', 5)
        self.assertEqual(record['messages'][-1], dict(role='user', content='第4句'))
        self.assertEqual(record['prompts'], ['第4句'])
        self.assertEqual(record['reference'], '第5句')
        self.assertNotIn(record['reference'], [m['content'] for m in record['messages']])

    def test_topic_stratified_selection_is_repeatable_and_bounded(self):
        ids = ['a', 'b', 'c', 'd', 'e', 'f']
        dialogue_by_id = {
            'a': dict(document_id=1), 'b': dict(document_id=1),
            'c': dict(document_id=2), 'd': dict(document_id=2),
            'e': dict(document_id=3), 'f': dict(document_id=3),
        }
        topic_by_document = {'1': '体育', '2': '娱乐', '3': '科技'}
        first = stratified_dialog_ids(ids, dialogue_by_id, topic_by_document, 4, 7)
        second = stratified_dialog_ids(ids, dialogue_by_id, topic_by_document, 4, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(stratified_dialog_ids(ids, dialogue_by_id, topic_by_document, 99, 7), ids)

    def test_long_history_trims_only_complete_old_turns(self):
        record = dict(id='long', messages=[
            dict(role='user', content='旧问'), dict(role='assistant', content='旧答'),
            dict(role='user', content='新问'), dict(role='assistant', content='新答'),
            dict(role='user', content='当前问题'), dict(role='assistant', content='当前回答'),
        ])
        encoded = encode_sample(record, TinyTokenizer(), max_length=27, trim_history=True)
        self.assertEqual(encoded['history_pairs_trimmed'], 1)
        self.assertLessEqual(len(encoded['input_ids']), 27)
        self.assertEqual(encoded['labels'][-5:-1], [ord(c) for c in '当前回答'])
        self.assertEqual(encoded['labels'][-1], 999)

    def test_long_current_turn_is_rejected_instead_of_truncated(self):
        record = dict(id='too-long', messages=[
            dict(role='user', content='问题' * 20), dict(role='assistant', content='回答' * 20),
        ])
        with self.assertRaisesRegex(ValueError, '超过长度上限'):
            encode_sample(record, TinyTokenizer(), max_length=32, trim_history=True)


if __name__ == '__main__':
    unittest.main()
