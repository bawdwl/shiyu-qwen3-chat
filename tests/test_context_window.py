import copy
import unittest
from types import SimpleNamespace
from context_window import prepare_context,prepare_context_details

class Tokenizer:
    def __init__(self): self.calls=[]
    def apply_chat_template(self,messages,**kwargs):
        self.calls.append(copy.deepcopy(messages))
        # 显式模拟：每条消息一个模板开销，再加内容长度。
        count=1+sum(len(m['content'])+1 for m in messages)
        return {'input_ids':SimpleNamespace(shape=(1,count))}

class ContextWindowTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer=Tokenizer()
        self.messages=[dict(role='user',content='旧问题'),dict(role='assistant',content='旧回答'),
                       dict(role='user',content='近问题'),dict(role='assistant',content='近回答'),
                       dict(role='user',content='新问题')]
    def test_short_input_unchanged(self):
        _,m=prepare_context(self.tokenizer,self.messages,100,True)
        self.assertEqual(m['context_omitted_turns'],0)
        self.assertEqual(self.tokenizer.calls[-1],self.messages)
    def test_exact_boundary_fits(self):
        _,m=prepare_context(self.tokenizer,self.messages,21,True)
        self.assertEqual(m['context_omitted_turns'],0)
    def test_recent_complete_pairs_retained_and_source_unchanged(self):
        original=copy.deepcopy(self.messages)
        _,m=prepare_context(self.tokenizer,self.messages,13,True)
        self.assertEqual(m['context_omitted_turns'],1);self.assertEqual(m['context_kept_turns'],1)
        self.assertEqual(self.tokenizer.calls[-1],original[2:]);self.assertEqual(self.messages,original)
    def test_system_prompt_is_always_preserved(self):
        source=[dict(role='system',content='系统')]+self.messages
        _,m=prepare_context(self.tokenizer,source,8,True)
        self.assertEqual(m['context_omitted_turns'],2)
        self.assertEqual(self.tokenizer.calls[-1],[source[0],source[-1]])
    def test_latest_question_is_never_truncated(self):
        source=[dict(role='user',content='问题很长'*10)]
        with self.assertRaisesRegex(ValueError,'最新问题'): prepare_context(self.tokenizer,source,10,True)
        self.assertEqual(self.tokenizer.calls[-1],source)
    def test_offline_default_still_rejects_long_history(self):
        with self.assertRaises(ValueError): prepare_context(self.tokenizer,self.messages,13)
        self.assertEqual(len(self.tokenizer.calls),1)
    def test_invalid_turns_are_not_silently_repaired(self):
        for source in ([],self.messages[:-1],[dict(role='assistant',content='错')]):
            with self.assertRaises(ValueError): prepare_context(self.tokenizer,source,100,True)

    def test_detailed_context_returns_only_complete_omitted_pairs(self):
        inputs,metrics,selected,omitted=prepare_context_details(self.tokenizer,self.messages,13,True)
        self.assertEqual(metrics['context_omitted_turns'],1)
        self.assertEqual(omitted,[dict(turn_index=1,user='旧问题',assistant='旧回答')])
        self.assertEqual(selected,self.messages[2:])
        self.assertEqual(inputs['input_ids'].shape[-1],13)

if __name__=='__main__': unittest.main()
