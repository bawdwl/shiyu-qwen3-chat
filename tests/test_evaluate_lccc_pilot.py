import unittest

from evaluate_lccc_pilot import (
    companion_review_messages,
    randomized_mode_mapping,
    render_review,
    select_cases,
    select_cases_by_ids,
)


class LcccBlindReviewTests(unittest.TestCase):
    def test_selects_one_latest_target_per_dialogue_and_balances_strata(self):
        records = [
            dict(id='a1', source_dialog_id='1', stratum='two_utterances', target_turn_index=1),
            dict(id='b1', source_dialog_id='2', stratum='three_utterances', target_turn_index=1),
            dict(id='c1', source_dialog_id='3', stratum='four_plus', target_turn_index=1),
            dict(id='c3', source_dialog_id='3', stratum='four_plus', target_turn_index=3),
        ]
        cases = select_cases(records, case_count=3, seed=7)
        self.assertEqual(len(cases), 3)
        self.assertEqual({row['stratum'] for row in cases},
                         {'two_utterances', 'three_utterances', 'four_plus'})
        self.assertIn('c3', {row['id'] for row in cases})
        self.assertEqual(len({row['source_dialog_id'] for row in cases}), 3)

    def test_multi_turn_sampling_uses_only_four_plus_dialogues(self):
        records = [
            dict(id='short', source_dialog_id='1', stratum='two_utterances', target_turn_index=1),
            dict(id='a1', source_dialog_id='2', stratum='four_plus', target_turn_index=1),
            dict(id='a3', source_dialog_id='2', stratum='four_plus', target_turn_index=3),
            dict(id='b1', source_dialog_id='3', stratum='four_plus', target_turn_index=1),
            dict(id='b3', source_dialog_id='3', stratum='four_plus', target_turn_index=3),
        ]
        cases = select_cases(records, case_count=2, seed=7, strata=('four_plus',))
        self.assertEqual({row['stratum'] for row in cases}, {'four_plus'})
        self.assertEqual({row['id'] for row in cases}, {'a3', 'b3'})

    def test_curated_case_ids_preserve_order_and_reject_non_multiturn(self):
        records = [
            dict(id='a', source_dialog_id='1', stratum='four_plus', target_turn_index=3,
                 messages=[{}, {}, {}]),
            dict(id='b', source_dialog_id='2', stratum='four_plus', target_turn_index=3,
                 messages=[{}, {}, {}]),
            dict(id='short', source_dialog_id='3', stratum='two_utterances', target_turn_index=1,
                 messages=[{}]),
        ]
        cases = select_cases_by_ids(records, ['b', 'a'])
        self.assertEqual([row['id'] for row in cases], ['b', 'a'])
        with self.assertRaisesRegex(ValueError, '多轮测试对话'):
            select_cases_by_ids(records, ['short'])

    def test_multi_turn_review_sheet_explains_context_requirement(self):
        sheet = render_review([], multi_turn_only=True)
        self.assertIn('中文多轮聊天回复盲评', sheet)
        self.assertIn('至少三条上下文消息', sheet)

    def test_companion_review_uses_app_character_prompt(self):
        context = [dict(role='user', content='今天有点累。')]
        messages = companion_review_messages(context)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertIn('小满', messages[0]['content'])
        self.assertEqual(messages[1:], context)

    def test_review_sheet_shows_randomized_labels_without_revealing_model_modes(self):
        sheet = render_review([dict(case_id='case-1', stratum='two_utterances',
                                    messages=[dict(role='user', content='今天有点累。')],
                                    answers={'A': '辛苦了。', 'B': '听起来今天不太轻松。'})])
        self.assertIn('回答 A', sheet)
        self.assertIn('回答 B', sheet)
        self.assertIn('- [ ] 两个都不自然', sheet)
        self.assertNotIn('Base', sheet)
        self.assertNotIn('LoRA', sheet)

    def test_randomized_mapping_keeps_one_base_and_one_lora(self):
        import random
        mapping = randomized_mode_mapping(random.Random(11))
        self.assertEqual(set(mapping), {'A', 'B'})
        self.assertEqual(set(mapping.values()), {'base', 'lora'})


if __name__ == '__main__':
    unittest.main()
