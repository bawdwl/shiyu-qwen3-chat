import unittest

from characters import CHARACTERS, conversation_messages
from role_knowledge import (
    build_additive_retrieval_messages,
    build_retrieval_messages,
    build_role_knowledge,
    retrieve_role_facts,
)


class RoleKnowledgeTests(unittest.TestCase):
    def test_retrieves_only_confirmed_coffee_interest_for_direct_question(self):
        rows = retrieve_role_facts('你也喜欢咖啡吗？')
        self.assertEqual([row['id'] for row in rows], ['interest.coffee'])
        self.assertEqual(rows[0]['status'], 'confirmed')

    def test_returns_job_and_unknown_bookstore_name_with_provenance(self):
        rows = retrieve_role_facts('你工作的那家书店叫什么名字？')
        ids = {row['id'] for row in rows}
        self.assertIn('unknown.bookstore_name', ids)
        self.assertIn('identity.occupation', ids)
        self.assertTrue(all(row['source'] for row in rows))
        unknown = next(row for row in rows if row['id'] == 'unknown.bookstore_name')
        self.assertEqual(unknown['status'], 'unconfirmed')

    def test_explicit_experience_query_retrieves_no_story_boundary(self):
        rows = retrieve_role_facts('你在书店工作时，有没有遇到过让你印象很深的客人？')
        row = next(row for row in rows if row['id'] == 'unknown.work_experiences')
        self.assertEqual(row['status'], 'unconfirmed')
        self.assertIn('没有确认过具体', row['text'])

    def test_cat_ownership_question_returns_unknown_without_inference(self):
        rows = retrieve_role_facts('你养猫了吗？猫叫什么名字？')
        ids = {row['id'] for row in rows}
        self.assertIn('unknown.cat_ownership_or_name', ids)
        self.assertNotIn('interest.cats', ids)

    def test_unrelated_daily_chat_retrieves_no_static_profile_fact(self):
        self.assertEqual(retrieve_role_facts('今天下班有点晚，到家以后整个人都不想动了。'), [])

    def test_arithmetic_question_mentioning_coffee_is_not_a_persona_preference_query(self):
        self.assertEqual(retrieve_role_facts('两杯咖啡每杯28元，打八折后一共多少钱？'), [])

    def test_retrieval_messages_keep_history_and_omit_full_static_profile(self):
        history = [dict(role='user', content='之前的话'), dict(role='assistant', content='之前的回答')]
        messages, facts = build_retrieval_messages(history, '你工作的那家书店叫什么名字？')
        self.assertEqual(messages[1:-1], history)
        self.assertEqual(messages[-1], dict(role='user', content='你工作的那家书店叫什么名字？'))
        self.assertNotIn('【固定角色身份档案】', messages[0]['content'])
        self.assertIn('【本轮检索到的角色资料】', messages[0]['content'])
        self.assertTrue(any(row['id'] == 'unknown.bookstore_name' for row in facts))

    def test_unconfirmed_records_come_only_from_canonical_profile_and_explicit_boundary(self):
        records = build_role_knowledge()
        profile = CHARACTERS['companion']['identity']
        labels = {row['text'].removesuffix('尚未设定。') for row in records if row['id'].startswith('unknown.') and row['id'] != 'unknown.work_experiences'}
        self.assertEqual(labels, set(profile['unconfirmed']))

    def test_additive_retrieval_without_match_is_identical_to_a_role_card(self):
        history = [dict(role='user', content='前面的话'), dict(role='assistant', content='之前的回答')]
        query = '今天下班有点晚，到家以后整个人都不想动了。'
        a_messages = conversation_messages(dict(character_id='companion', messages=history), query)
        b_messages, facts = build_additive_retrieval_messages(history, query)
        self.assertEqual(facts, [])
        self.assertEqual(b_messages, a_messages)

    def test_additive_retrieval_keeps_full_a_prompt_and_appends_only_matching_fact(self):
        history = []
        query = '你也喜欢咖啡吗？'
        a_messages = conversation_messages(dict(character_id='companion', messages=history), query)
        b_messages, facts = build_additive_retrieval_messages(history, query)
        self.assertEqual([row['id'] for row in facts], ['interest.coffee'])
        self.assertEqual(b_messages[0]['content'][:len(a_messages[0]['content'])], a_messages[0]['content'])
        self.assertEqual(b_messages[1:], a_messages[1:])
        self.assertIn('[confirmed] 兴趣之一：咖啡。', b_messages[0]['content'])

    def test_additive_retrieval_includes_unknown_boundary_without_inventing_a_value(self):
        query = '你工作的那家书店叫什么名字？'
        messages, facts = build_additive_retrieval_messages([], query)
        self.assertIn('unknown.bookstore_name', {row['id'] for row in facts})
        self.assertIn('[unconfirmed] 书店名称尚未设定。', messages[0]['content'])
        self.assertNotIn('墨香书屋', messages[0]['content'])

    def test_retrieves_confirmed_facts_from_natural_question_phrasings(self):
        cases = {
            '我记得你是在厦门的独立书店当店员，今年24岁，对吧？': {
                'identity.age', 'identity.city', 'identity.occupation',
            },
            '你是厦门本地人吗，还是后来来厦门工作的？': {
                'identity.city', 'unknown.hometown',
            },
            '你平时喜欢去哪儿散步？厦门有没有你特别喜欢的一条路线？': {
                'identity.city', 'interest.walking',
            },
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                actual = {row['id'] for row in retrieve_role_facts(query)}
                self.assertTrue(expected <= actual, (expected, actual))

    def test_unknown_education_retrieval_matches_everyday_wording(self):
        actual = {row['id'] for row in retrieve_role_facts('你大学学的是什么专业？')}
        self.assertIn('unknown.education', actual)

    def test_work_experience_retrieval_matches_everyday_wording(self):
        actual = {row['id'] for row in retrieve_role_facts('你在书店上班的时候，有没有遇到过什么特别好玩的事？')}
        self.assertIn('unknown.work_experiences', actual)


if __name__ == '__main__':
    unittest.main()
