import unittest
from memory_records import validate_candidate, propose_revision


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{'id': 1, 'role': 'user', 'content': '面试改到周五上午了。'},
                         {'id': 2, 'role': 'assistant', 'content': '我喜欢手冲咖啡。'}]

    def candidate(self):
        return {'kind': 'user_report', 'content': '用户说面试改到周五上午',
                'evidence': [{'message_id': 1, 'quote': '面试改到周五上午了。'}]}

    def test_relative_date_stays_unresolved(self):
        value = validate_candidate(self.candidate(), self.messages)
        self.assertIsNone(value['event_date'])
        self.assertFalse(value['semantic_verified'])

    def test_unstated_calendar_date_rejected(self):
        value = self.candidate(); value['event_date'] = '2026-11-06'
        with self.assertRaises(ValueError): validate_candidate(value, self.messages)

    def test_wrong_speaker_rejected(self):
        value = self.candidate(); value['kind'] = 'character_detail'
        with self.assertRaises(ValueError): validate_candidate(value, self.messages)

    def test_invented_quote_rejected(self):
        value = self.candidate(); value['evidence'][0]['quote'] = '用户习惯早起'
        with self.assertRaises(ValueError): validate_candidate(value, self.messages)

    def test_explicit_date_and_character_evidence(self):
        rows = [{'id': 3, 'role': 'assistant', 'content': '读书会安排在2026-11-06。'}]
        value = {'kind': 'character_detail', 'content': '角色说读书会在2026-11-06',
                 'event_date': '2026-11-06', 'evidence': [{'message_id': 3, 'quote': rows[0]['content']}]}
        self.assertEqual(validate_candidate(value, rows)['status'], 'candidate')

    def test_update_remains_reviewable_and_keeps_old_record(self):
        rows = [{'id': 1, 'role': 'user', 'content': '面试周三。'},
                {'id': 2, 'role': 'user', 'content': '改到周五。'}]
        def item(mid):
            return validate_candidate({'kind': 'user_report', 'content': rows[mid-1]['content'],
                'evidence': [{'message_id': mid, 'quote': rows[mid-1]['content']}]}, rows)
        old, new = item(1), item(2)
        proposed = propose_revision(old, new, same_subject_and_fact=True)
        self.assertEqual(proposed['action'], 'review_supersession')
        self.assertEqual(old['status'], 'candidate')
        self.assertEqual(propose_revision(new, old, same_subject_and_fact=True)['action'], 'review_order')
        self.assertEqual(propose_revision(old, new)['action'], 'review_separately')

    def test_matching_quote_does_not_prove_an_extra_claim(self):
        rows = [{'id': 1, 'role': 'assistant', 'content': '早点去图书馆应该能找到座位。'}]
        candidate = {'kind': 'character_detail', 'content': '角色习惯早起，熟悉图书馆。',
                     'evidence': [{'message_id': 1, 'quote': rows[0]['content']}]}
        result = validate_candidate(candidate, rows)
        self.assertEqual(result['status'], 'candidate')
        self.assertFalse(result['semantic_verified'])

    def test_subject_and_fact_key_are_normalised(self):
        value = self.candidate()
        value.update(subject='user', fact_key='USER.INTERVIEW.TIME', value='周五上午')
        result = validate_candidate(value, self.messages)
        self.assertEqual(result['subject'], 'user')
        self.assertEqual(result['fact_key'], 'user.interview.time')
        self.assertEqual(result['value'], '周五上午')

    def test_invalid_subject_and_fact_key_rejected(self):
        value = self.candidate()
        value['subject'] = 'character'
        with self.assertRaises(ValueError): validate_candidate(value, self.messages)
        value['subject'] = 'user'
        value['fact_key'] = '包含中文'
        with self.assertRaises(ValueError): validate_candidate(value, self.messages)

    def test_automatic_operation_and_target_are_validated(self):
        value = self.candidate()
        value.update(operation='supersede', target_item_id=9)
        result = validate_candidate(value, self.messages)
        self.assertEqual(result['operation'], 'supersede')
        self.assertEqual(result['target_item_id'], 9)
        value['operation'] = 'delete'
        with self.assertRaises(ValueError): validate_candidate(value, self.messages)
        value['operation'] = 'merge'
        value['target_item_id'] = None
        with self.assertRaises(ValueError): validate_candidate(value, self.messages)

    def test_paraphrased_fact_value_rejected(self):
        value = self.candidate()
        value.update(subject='user', fact_key='user.interview.time', value='星期五下午')
        with self.assertRaises(ValueError): validate_candidate(value, self.messages)

    def test_revision_detected_from_structured_fields(self):
        rows = [{'id': 1, 'role': 'user', 'content': '面试周三。'},
                {'id': 2, 'role': 'user', 'content': '改到周五。'}]
        def item(mid):
            return validate_candidate({'kind': 'user_report', 'subject': 'user',
                'fact_key': 'user.interview.time', 'value': '周五' if mid == 2 else '周三',
                'content': rows[mid-1]['content'],
                'evidence': [{'message_id': mid, 'quote': rows[mid-1]['content']}]}, rows)
        self.assertEqual(
            propose_revision(item(1), item(2))['action'],
            'review_supersession',
        )
