"""SillyTavern/OpenAI-compatible chat protocol tests without loading a model."""
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from backend import create_app


class FakeEngine:
    model_id='Qwen/Qwen3-1.7B'
    adapter_available=True

    def __init__(self):
        self.calls=[]

    def reply(self,messages,model_mode='base',max_new_tokens=256,generation_overrides=None):
        self.calls.append((messages,model_mode,max_new_tokens,generation_overrides))
        return '本地回答',dict(input_tokens=17,output_tokens=3)

    def reply_stream(self,messages,model_mode,stop_event,emit,max_new_tokens=256,generation_overrides=None):
        self.calls.append((messages,model_mode,max_new_tokens,generation_overrides))
        emit('本地');emit('回答')
        return '本地回答',dict(input_tokens=17,output_tokens=3)


class OpenAICompatTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.engine=FakeEngine()
        self.client=TestClient(create_app(Path(self.temp.name)/'chat.sqlite3',engine_factory=lambda:self.engine))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.temp.cleanup()

    def body(self,**overrides):
        value={'model':'Qwen3-1.7B','messages':[{'role':'system','content':'扮演一个温柔的角色。'},
                                                  {'role':'user','content':'今天有点累'}]}
        value.update(overrides)
        return value

    def test_model_list_exposes_base_and_loaded_adapter(self):
        result=self.client.get('/v1/models').json()
        self.assertEqual([row['id'] for row in result['data']],
                         ['Qwen/Qwen3-1.7B','Qwen/Qwen3-1.7B-lora'])

    def test_license_route_serves_project_agpl_notice(self):
        response=self.client.get('/license')
        self.assertEqual(response.status_code,200)
        self.assertIn('GNU AFFERO GENERAL PUBLIC LICENSE',response.text)

    def test_nonstream_openai_shape_and_request_mapping(self):
        result=self.client.post('/v1/chat/completions',json=self.body(
            max_tokens=96,temperature=0.4,top_p=0.9,top_k=12,repetition_penalty=1.1)).json()
        self.assertEqual(result['object'],'chat.completion')
        self.assertEqual(result['choices'][0]['message']['content'],'本地回答')
        self.assertEqual(result['usage'],{'prompt_tokens':17,'completion_tokens':3,'total_tokens':20})
        messages,mode,limit,settings=self.engine.calls[0]
        self.assertEqual(messages[0]['content'],'扮演一个温柔的角色。')
        self.assertEqual((mode,limit),('base',96))
        self.assertEqual(settings,{'temperature':0.4,'top_p':0.9,'top_k':12,'repetition_penalty':1.1})

    def test_lora_alias_selects_loaded_adapter(self):
        result=self.client.post('/v1/chat/completions',json=self.body(model='qwen3-1.7b-lora')).json()
        self.assertEqual(result['choices'][0]['finish_reason'],'stop')
        self.assertEqual(self.engine.calls[-1][1],'lora')

    def test_stream_uses_openai_sse_and_does_not_persist_duplicate_history(self):
        response=self.client.post('/v1/chat/completions',json=self.body(stream=True))
        self.assertIn('text/event-stream',response.headers['content-type'])
        self.assertIn('"content": "本地"',response.text)
        self.assertIn('"content": "回答"',response.text)
        self.assertIn('"finish_reason": "stop"',response.text)
        self.assertTrue(response.text.rstrip().endswith('data: [DONE]'))
        self.assertEqual(self.client.get('/sessions').json()['sessions'],[])

    def test_invalid_model_and_missing_user_are_rejected(self):
        self.assertEqual(self.client.post('/v1/chat/completions',json=self.body(model='other')).status_code,404)
        self.assertEqual(self.client.post('/v1/chat/completions',json=self.body(
            messages=[{'role':'system','content':'只给系统消息'}])).status_code,422)

    def test_invalid_sampling_parameters_are_rejected_without_generation(self):
        self.assertEqual(self.client.post('/v1/chat/completions',json=self.body(top_p=0)).status_code,422)
        self.assertEqual(self.engine.calls,[])


if __name__=='__main__':
    unittest.main(verbosity=2)
