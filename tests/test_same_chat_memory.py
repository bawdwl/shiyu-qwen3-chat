import unittest
from types import SimpleNamespace

from inference import QwenChat
from same_chat_memory import build_same_chat_memory,inject_same_chat_memory


class SameChatMemoryTests(unittest.TestCase):
    def setUp(self):
        self.omitted=[
            dict(turn_index=1,user='你平常喝什么咖啡？',
                 assistant='我最近很喜欢燕麦拿铁，常去书店街口那家店买。'),
            dict(turn_index=2,user='你周末喜欢做什么？',
                 assistant='我周末偶尔会去海边散散步，走完再回书店。'),
            dict(turn_index=3,user='我喜欢猫。',
                 assistant='猫确实很可爱，你喜欢什么性格的猫？'),
        ]

    def test_recalls_relevant_first_person_assistant_sentence(self):
        found=build_same_chat_memory(self.omitted,'你之前说的那杯咖啡是什么来着？')
        self.assertEqual(len(found),1)
        self.assertEqual(found[0]['turn_index'],1)
        self.assertIn('燕麦拿铁',found[0]['text'])
        self.assertNotIn('用户说过',found[0]['text'])

    def test_does_not_record_user_fact_as_character_fact(self):
        found=build_same_chat_memory(self.omitted,'你刚才说你喜欢猫，叫什么来着？')
        self.assertEqual(found,[])

    def test_generic_recall_uses_latest_first_person_statement(self):
        found=build_same_chat_memory(self.omitted,'你刚才说的那个叫什么来着？')
        self.assertEqual([item['turn_index'] for item in found],[2])

    def test_no_match_without_recall_or_topic(self):
        self.assertEqual(build_same_chat_memory(self.omitted,'今天天气不错。'),[])

    def test_memory_is_added_only_for_xiaoman_and_is_auditable(self):
        messages=[dict(role='system',content='你是原创AI聊天角色小满。'),
                  dict(role='user',content='之前那杯咖啡呢？')]
        result,metrics=inject_same_chat_memory(messages,self.omitted,'之前那杯咖啡呢？')
        self.assertEqual(metrics['same_chat_memory_retrieved'],1)
        self.assertEqual(metrics['same_chat_memory_source_turns'],[1])
        self.assertIn('燕麦拿铁',result[0]['content'])
        self.assertEqual(messages[0]['content'],'你是原创AI聊天角色小满。')

    def test_other_characters_do_not_receive_xiaoman_memory(self):
        messages=[dict(role='system',content='你是学习搭子知夏。'),
                  dict(role='user',content='咖啡呢？')]
        result,metrics=inject_same_chat_memory(messages,self.omitted,'咖啡呢？')
        self.assertEqual(result,messages)
        self.assertEqual(metrics['same_chat_memory_retrieved'],0)

    def test_inference_recalls_only_after_original_turn_is_truncated(self):
        class TinyTokenizer:
            def apply_chat_template(self,messages,**kwargs):
                count=1+sum(len(message['content'])+1 for message in messages)
                return {'input_ids':SimpleNamespace(shape=(1,count))}

        messages=[dict(role='system',content='你是原创AI聊天角色小满。'+('角色背景。'*35)),
                  dict(role='user',content='你平常喝什么咖啡？'),
                  dict(role='assistant',content='我最近很喜欢燕麦拿铁，常去书店街口那家店买。'),
                  dict(role='user',content='你今天忙吗？'),
                  dict(role='assistant',content='还好，今天书店不太忙。'*120),
                  dict(role='user',content='最近读书有什么安排？'),
                  dict(role='assistant',content='我想慢慢读完手边那本书。'*120),
                  dict(role='user',content='你之前说的那杯咖啡是什么来着？')]
        engine=object.__new__(QwenChat)
        engine.tokenizer=TinyTokenizer();engine.trim_history=True;engine.same_chat_memory=True
        _,metrics=engine._prepare_inputs(messages)
        self.assertGreater(metrics['same_chat_memory_baseline_omitted_turns'],0)
        self.assertEqual(metrics['same_chat_memory_retrieved'],1)
        self.assertEqual(metrics['same_chat_memory_source_turns'],[1])

    def test_inference_leaves_short_conversation_unchanged(self):
        class TinyTokenizer:
            def apply_chat_template(self,messages,**kwargs):
                count=1+sum(len(message['content'])+1 for message in messages)
                return {'input_ids':SimpleNamespace(shape=(1,count))}

        messages=[dict(role='system',content='你是原创AI聊天角色小满。'),
                  dict(role='user',content='你喝什么咖啡？'),
                  dict(role='assistant',content='我喜欢燕麦拿铁。'),
                  dict(role='user',content='你喜欢哪种咖啡？')]
        engine=object.__new__(QwenChat)
        engine.tokenizer=TinyTokenizer();engine.trim_history=True;engine.same_chat_memory=True
        _,metrics=engine._prepare_inputs(messages)
        self.assertEqual(metrics['context_omitted_turns'],0)
        self.assertEqual(metrics['same_chat_memory_retrieved'],0)


if __name__=='__main__':
    unittest.main()
