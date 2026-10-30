import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from backend import create_app
from runtime_models import get_profile

class Engine:
    model_id='Qwen/Qwen3-1.7B'
    adapter_available=False
    def __init__(self): self.calls=0
    def reply(self,messages,**kwargs): self.calls+=1;return '模拟回答',dict(seconds=.1,output_tokens=2)
    def reply_stream(self,messages,*args,**kwargs): return self.reply(messages)

class RuntimeModelTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.engine=Engine()
        self.client=TestClient(create_app(Path(self.temp.name)/'chat.db',engine_factory=lambda:self.engine))
        self.client.__enter__()
    def tearDown(self):
        self.client.__exit__(None,None,None);self.temp.cleanup()
    def test_new_session_records_actual_model_and_generates(self):
        row=self.client.post('/sessions',json={}).json()
        self.assertEqual(row['model_id'],self.engine.model_id)
        for route in ('/chat','/chat/stream'):
            response=self.client.post(route,json=dict(session_id=row['id'],message='你好'))
            self.assertEqual(response.status_code,200)
        self.assertEqual(self.engine.calls,2)
    def test_old_model_sessions_readable_but_not_mixed(self):
        row=self.client.app.state.store.create('base')
        for route in ('/chat','/chat/stream'):
            self.assertEqual(self.client.post(route,json=dict(session_id=row['id'],message='你好')).status_code,422)
        self.assertEqual(self.client.get('/sessions/'+row['id']).status_code,200)
        self.assertFalse(self.client.get('/health').json()['generation_busy'])
        self.assertEqual(self.engine.calls,0)
    def test_adapter_mismatch_rejected_before_model_loading(self):
        from inference import QwenChat,MODEL_ID,MODEL_REVISION
        path=Path(self.temp.name)/'adapter-run';path.mkdir()
        (path/'run-config.json').write_text(json.dumps(dict(base_model=MODEL_ID,revision=MODEL_REVISION)),encoding='utf-8')
        with patch('inference.load_model') as loader:
            with self.assertRaisesRegex(ValueError,'版本不匹配'): QwenChat(path,model_profile='qwen3-1.7b')
            loader.assert_not_called()
    def test_profile_copy_and_offline_defaults_remain_fixed(self):
        from inference import MODEL_ID
        profile=get_profile('qwen3-1.7b');profile['model_id']='modified'
        self.assertEqual(get_profile('qwen3-1.7b')['model_id'],'Qwen/Qwen3-1.7B')
        self.assertEqual(MODEL_ID,'Qwen/Qwen3-0.6B')

    def test_adapter_metadata_accepts_exact_local_base_path(self):
        from inference import adapter_base_matches
        self.assertTrue(adapter_base_matches('outputs/models/qwen3-1.7b',
                                             'Qwen/Qwen3-1.7B',
                                             'outputs/models/qwen3-1.7b'))
        self.assertFalse(adapter_base_matches('outputs/models/qwen3-0.6b',
                                              'Qwen/Qwen3-1.7B',
                                              'outputs/models/qwen3-1.7b'))

if __name__=='__main__': unittest.main()
