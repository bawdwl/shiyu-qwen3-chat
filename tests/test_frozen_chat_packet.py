import unittest

from evaluate_frozen_chat import assemble_review


class FrozenChatPacketTests(unittest.TestCase):
    def setUp(self):
        self.cases = [dict(id='case-1', messages=[dict(role='user', content='周末想出去走走。')])]
        self.results = [dict(case_id='case-1', label='B', mode='base', answer='基础回答'),
                        dict(case_id='case-1', label='A', mode='lora', answer='微调回答')]

    def test_recovery_keeps_exact_answers_mapping_and_case_id(self):
        packet, key = assemble_review(self.cases, self.results)
        self.assertEqual(packet[0]['case_id'], 'case-1')
        self.assertEqual(packet[0]['answers'], dict(A='微调回答', B='基础回答'))
        self.assertEqual(key, {'case-1': dict(A='lora', B='base')})

    def test_partial_batch_cannot_become_a_completed_review(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            assemble_review(self.cases, self.results[:1])

    def test_duplicate_label_and_mode_are_rejected(self):
        self.results[1]['label'] = 'B'
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            assemble_review(self.cases, self.results)
        self.results[1]['label'] = 'A'
        self.results[1]['mode'] = 'base'
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            assemble_review(self.cases, self.results)

    def test_unknown_case_and_blank_answer_are_rejected(self):
        self.results[0]['case_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            assemble_review(self.cases, self.results)
        self.results[0]['case_id'] = 'case-1'
        self.results[0]['answer'] = ' '
        with self.assertRaisesRegex(ValueError, 'empty'):
            assemble_review(self.cases, self.results)


if __name__ == '__main__':
    unittest.main()
