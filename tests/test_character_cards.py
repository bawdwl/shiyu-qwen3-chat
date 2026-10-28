import json
import unittest

from character_cards import (build_generation_messages, card_to_runtime, normalize_card,
                             parse_generated_card, retrieve_worldbook)


class CharacterCardTests(unittest.TestCase):
    def test_runtime_prompt_makes_followup_questions_optional_and_relevant(self):
        runtime = card_to_runtime(normalize_card({
            'name': '林间电台', 'personality': '温柔、善于倾听',
            'speech_style': '亲切自然', 'relationship': '熟悉的朋友',
        }))
        prompt = runtime['prompt']
        self.assertIn('回应用户刚说的具体内容', prompt)
        self.assertIn('不要重复上一轮回答或开场情境', prompt)
        self.assertIn('不要每轮都追问', prompt)

    def test_card_generation_avoids_question_hook_examples(self):
        system = build_generation_messages('海边电台主持人')[0]['content']
        self.assertIn('每条回复末尾都问一句', system)
        self.assertIn('示例回复不要都以问题收尾', system)

    def test_worldbook_matches_enabled_keyword_entries_and_respects_limit(self):
        card = normalize_card({
            'name': '海音',
            'worldbook': [
                {'title': '单词命中', 'keywords': ['电台'], 'content': 'A'},
                {'title': '多词命中', 'keywords': ['电台', '夜班'], 'content': 'B'},
                {'title': '禁用', 'keywords': ['电台'], 'content': 'C', 'enabled': False},
                {'title': '未命中', 'keywords': ['咖啡店'], 'content': 'D'},
            ],
        })
        self.assertEqual([entry['title'] for entry in retrieve_worldbook(card, '夜班电台', limit=1)], ['多词命中'])
        self.assertEqual(retrieve_worldbook(card, '咖啡店')[0]['title'], '未命中')
        self.assertEqual(retrieve_worldbook(card, '没有相关内容'), [])

    def test_generated_details_are_proposals_and_product_name_collision_is_repaired(self):
        model_output = json.dumps({
            'name': '拾语', 'tagline': '海边的声音', 'description': '夜班主持人',
            'background': '拾语在海边电台主持节目。',
            'fixed_facts': [{'key': '城市', 'value': '海边小城', 'status': 'confirmed'}],
            'example_dialogue': [
                {'user': '你好', 'character': '拾语'},
                {'user': '我睡不着。', 'character': '我给你放一首轻一点的歌，慢慢听就好。'},
            ],
        }, ensure_ascii=False)
        card = parse_generated_card(model_output, '夜班电台主持人')
        self.assertEqual(card['name'], '海音')
        self.assertEqual(card['fixed_facts'][0]['status'], 'proposal')
        self.assertIn('已临时改为“海音”', card['draft_notice'])
        self.assertIn('海音在海边电台', card['background'])
        self.assertEqual(len(card['example_dialogue']), 1)
        self.assertEqual(card['example_dialogue'][0]['character'], '我给你放一首轻一点的歌，慢慢听就好。')

    def test_card_limits_reject_unbounded_worldbook(self):
        with self.assertRaisesRegex(ValueError, '最多30项'):
            normalize_card({'name': '山海', 'worldbook': [
                {'title': str(i), 'keywords': ['词'], 'content': '条目'} for i in range(31)
            ]})


if __name__ == '__main__':
    unittest.main()
