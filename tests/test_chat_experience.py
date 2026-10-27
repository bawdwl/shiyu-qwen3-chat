"""界面重试与断线恢复测试：真实回调，模拟HTTP边界故障。"""
import copy
import unittest
from app import build_app

class Client:
    def __init__(self):
        self.messages=[];self.calls=0;self.behavior='fail';self.waits=0
    def request(self,method,path,**kwargs):
        if path=='/health': return dict(model_modes=['base'],speech_provider=None,asr_provider=None)
        if path=='/sessions' and method=='POST': return dict(id='test')
        if path=='/sessions/test': return dict(id='test',model_mode='base',messages=copy.deepcopy(self.messages))
        if path=='/sessions': return dict(sessions=[])
        raise AssertionError((method,path))
    def prepare(self,sid): pass
    def wait_idle(self,seconds=10): self.waits+=1
    def stream(self,sid,text,**kwargs):
        self.calls+=1
        if self.behavior=='fail': raise RuntimeError('simulated disconnect before save')
        self.messages.extend([dict(id=len(self.messages)+1,role='user',content=text.strip()),
                              dict(id=len(self.messages)+2,role='assistant',content='回答')])
        if self.behavior=='saved_disconnect': raise RuntimeError('simulated lost done event')
        yield 'done',dict(metrics=dict(seconds=1,output_tokens=256,ended_with_eos=False))

class ChatExperienceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=Client();demo=build_app(cls.client)
        cls.callbacks={v.fn.__name__:v.fn for v in demo.fns.values() if v.fn}
    def setUp(self):
        self.client.messages=[];self.client.calls=0;self.client.behavior='fail';self.client.waits=0
    def call(self,name,text='问题',sid='test'):
        return list(self.callbacks[name](text,sid,'base',256))[-1]
    def test_failed_question_returns_to_draft(self):
        result=self.call('send_with_audio')
        self.assertEqual(result[0],'问题');self.assertEqual(self.client.messages,[])
        self.assertIn('重试',result[3])
    def test_saved_reply_restored_after_lost_completion_event(self):
        self.client.behavior='saved_disconnect'
        result=self.call('send_with_audio')
        self.assertEqual(result[0],'');self.assertEqual(len(result[2]),2)
        self.assertIn('回答已保存',result[3]);self.assertEqual(self.client.calls,1)
    def test_retry_checks_saved_reply_without_duplicate_generation(self):
        self.client.behavior='success';self.call('send_with_audio')
        result=self.call('retry_message')
        self.assertEqual(self.client.calls,1);self.assertEqual(len(self.client.messages),2)
        self.assertEqual(result[0],'');self.assertIn('没有重复发送',result[3])
        self.assertEqual(self.client.waits,1)
    def test_retry_unsaved_question_generates_once(self):
        self.call('send_with_audio');self.client.behavior='success'
        result=self.call('retry_message')
        self.assertEqual(len(self.client.messages),2);self.assertEqual(result[0],'')
        self.assertIn('请继续',result[3])
    def test_empty_retry_does_not_generate(self):
        self.call('retry_message',text=' ')
        self.assertEqual(self.client.calls,0);self.assertEqual(self.client.waits,0)

if __name__=='__main__': unittest.main()
