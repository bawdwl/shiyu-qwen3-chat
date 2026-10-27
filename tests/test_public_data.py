import unittest
from prepare_oasst2_candidates import collect_candidates, split_for_root


def message(mid, parent, role, text, **updates):
    row = dict(message_id=mid, parent_id=parent, message_tree_id='tree', role=role,
               text=text, lang='zh', review_result=True, deleted=False,
               synthetic=False, rank=0 if role == 'assistant' else None,
               labels={'quality': {'value': 0.8}})
    row.update(updates)
    return row


class PublicDataTests(unittest.TestCase):
    def test_multi_turn_preserves_context_and_omits_author_id(self):
        rows = [message('u', None, 'prompter', '怎样安排学习？', user_id='private-author'),
                message('a', 'u', 'assistant', '先确定每天的时间。'),
                message('u2', 'a', 'prompter', '每天一个小时。'),
                message('a2', 'u2', 'assistant', '分成阅读与练习两部分。')]
        out, _ = collect_candidates(rows)
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[-1]['messages']), 4)
        self.assertNotIn('user_id', str(out))
        self.assertEqual(out[0]['split'], out[1]['split'])

    def test_flagged_or_unaccepted_ancestor_cannot_be_skipped(self):
        for updates in ({'review_result': False}, {'deleted': True}, {'synthetic': True},
                        {'text': '联系abc@example.com'}, {'lang': 'en'}):
            root = message('u', None, 'prompter', '问题')
            root.update(updates)
            rows = [root, message('a', 'u', 'assistant', '回答')]
            out, _ = collect_candidates(rows)
            self.assertEqual(out, [])

    def test_low_quality_and_nonpreferred_answer_rejected(self):
        for updates in ({'rank': 1}, {'labels': {'quality': {'value': 0.2}}}):
            out, _ = collect_candidates([message('u', None, 'prompter', '问题'),
                                        message('a', 'u', 'assistant', '回答', **updates)])
            self.assertEqual(out, [])

    def test_missing_parent_cycle_and_duplicate_ids_rejected(self):
        out, stats = collect_candidates([message('a', 'missing', 'assistant', '回答')])
        self.assertEqual(out, [])
        self.assertIn('missing_or_non_zh_parent', stats)
        out, stats = collect_candidates([message('a', 'a', 'assistant', '回答')])
        self.assertEqual(out, [])
        self.assertIn('cycle', stats)
        with self.assertRaises(ValueError):
            collect_candidates([message('a', None, 'assistant', '回答')] * 2)

    def test_normalized_root_keeps_duplicate_questions_in_same_split(self):
        self.assertEqual(split_for_root('如何学习 Python？'), split_for_root('如何学习python'))


if __name__ == '__main__':
    unittest.main()
