"""无GPU的接口与持久化测试，模型故障由明确标注的假引擎模拟。"""
import logging
import importlib.util
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from uuid import uuid4

from fastapi.testclient import TestClient

from backend import create_app
from storage import ChatStore


class FakeEngine:
    adapter_available=True
    def __init__(self):
        self.calls=[];self.error=None;self.answer='测试回答'
        self.entered=None;self.release=None
    def reply(self,messages,model_mode='base',max_new_tokens=256,generation_overrides=None):
        self.last_limit=max_new_tokens
        self.last_overrides=generation_overrides
        self.calls.append((messages,model_mode))
        if self.entered:
            self.entered.set()
            if not self.release.wait(5): raise RuntimeError('test release timed out')
        if self.error: raise self.error
        return self.answer,dict(seconds=0.01,output_tokens=4)


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'chat.sqlite3'
        self.engine=FakeEngine()
        self.client=TestClient(create_app(self.path,engine_factory=lambda:self.engine))
        self.client.__enter__()
    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.temp.cleanup()
    def session(self,mode='base'):
        r=self.client.post('/sessions',json={'model_mode':mode})
        self.assertEqual(r.status_code,201)
        return r.json()['id']
    def send(self,sid,message='问题'):
        return self.client.post('/chat',json={'session_id':sid,'message':message})

    def test_health(self):
        self.assertEqual(self.client.get('/health').json()['model_modes'],['base','lora'])

    def test_custom_engine_failure_does_not_require_torch_import(self):
        from unittest.mock import patch
        sid=self.session();self.engine.error=RuntimeError('CPU engine failure')
        with patch.dict('sys.modules',{'torch':None,'torch.cuda':None}):
            response=self.send(sid)
        self.assertEqual(response.status_code,500)
        self.assertEqual(self.client.get('/sessions/'+sid).json()['messages'],[])

    def test_nonempty_filter_runs_before_pagination(self):
        store=self.client.app.state.store
        sid=store.create('base')['id'];store.save_turn(sid,'旧对话不能被挤掉','回答')
        for _ in range(105): store.create('base')
        rows=self.client.get('/sessions?limit=1&nonempty=true').json()
        self.assertEqual([r['id'] for r in rows['sessions']],[sid])
        self.assertFalse(rows['has_more'])
        self.assertNotEqual(self.client.get('/sessions?limit=1').json()['sessions'][0]['id'],sid)

    def test_history_search_includes_later_questions_and_assistant_answers(self):
        store=self.client.app.state.store
        sid=store.create('base')['id'];store.save_turn(sid,'无关的开场','普通回答')
        store.save_turn(sid,'后续关键字：火山旅行','专有名词：月光车票')
        for query in ('火山旅行','月光车票'):
            rows=self.client.get('/sessions',params={'q':query,'nonempty':True}).json()['sessions']
            self.assertEqual([r['id'] for r in rows],[sid])
        self.assertEqual(self.client.get('/sessions',params={'q':'不存在的内容'}).json()['sessions'],[])

    def test_search_is_literal_and_parameterized(self):
        store=self.client.app.state.store
        hit=store.create('base')['id'];store.save_turn(hit,"100%_ ' OR 1=1 --",'ANSWER')
        other=store.create('base')['id'];store.save_turn(other,'普通问题','普通回答')
        for query in ('%', '_', "' OR 1=1 --", 'answer'):
            rows=self.client.get('/sessions',params={'q':query}).json()['sessions']
            self.assertEqual([r['id'] for r in rows],[hit])

    def test_nonempty_pagination_has_more_and_stable_tie_order(self):
        store=self.client.app.state.store
        ids=[]
        for i in range(3):
            sid=store.create('base')['id'];store.save_turn(sid,str(i),'回答');ids.append(sid)
        with store.connect() as db: db.execute("UPDATE sessions SET updated_at='same'")
        actual=[]
        for offset in range(3):
            page=self.client.get('/sessions',params={'limit':1,'offset':offset,'nonempty':True}).json()
            actual.append(page['sessions'][0]['id'])
            self.assertEqual(page['has_more'],offset<2)
        self.assertEqual(actual,sorted(ids))
        self.assertEqual(self.client.get('/sessions',params={'q':'a'*201}).status_code,422)

    def test_cleared_history_is_hidden_without_deleting_session(self):
        sid=self.session();self.send(sid)
        self.client.delete('/sessions/'+sid+'/messages')
        self.assertEqual(self.client.get('/sessions?nonempty=true').json()['sessions'],[])
        self.assertEqual(self.client.get('/sessions/'+sid).status_code,200)

    def test_delete_session_removes_conversation_and_cascades_messages(self):
        sid=self.session();self.send(sid,'要删除的问题')
        response=self.client.delete('/sessions/'+sid)
        self.assertEqual(response.status_code,200)
        self.assertTrue(response.json()['deleted'])
        self.assertEqual(self.client.get('/sessions/'+sid).status_code,404)
        self.assertNotIn(sid,[row['id'] for row in self.client.get('/sessions').json()['sessions']])
        with self.client.app.state.store.connect() as db:
            self.assertEqual(db.execute('SELECT count(*) FROM messages WHERE session_id=?',(sid,)).fetchone()[0],0)
        self.assertEqual(self.client.delete('/sessions/'+sid).status_code,404)

    def test_delete_session_is_rejected_while_generation_gate_is_busy(self):
        sid=self.session();gate=self.client.app.state.gate;gate.acquire()
        try:
            self.assertEqual(self.client.delete('/sessions/'+sid).status_code,409)
        finally:
            gate.release()
        self.assertEqual(self.client.get('/sessions/'+sid).status_code,200)

    def test_multiturn_and_history(self):
        sid=self.session()
        self.assertEqual(self.send(sid,'首轮').status_code,200)
        self.assertEqual(self.send(sid,'追问').json()['message_count'],4)
        self.assertEqual([m['content'] for m in self.engine.calls[-1][0] if m['role']!='system'],
                         ['首轮','测试回答','追问'])
        saved=self.client.get('/sessions/'+sid).json()['messages']
        self.assertEqual([m['role'] for m in saved],['user','assistant','user','assistant'])
        self.assertTrue(all(m['created_at'] for m in saved))

    def test_session_model_binding_and_isolation(self):
        a=self.session('base');b=self.session('lora')
        self.send(a);self.send(b)
        self.assertEqual([c[1] for c in self.engine.calls],['base','lora'])
        self.assertEqual(self.engine.calls[-1][0][0]['role'],'system')
        self.assertEqual(self.engine.calls[-1][0][-1]['role'],'user')
        r=self.client.post('/chat',json={'session_id':a,'message':'x','model_mode':'lora'})
        self.assertEqual(r.status_code,422)

    def test_invalid_requests(self):
        sid=self.session()
        for message in ['', '  \n ', 'x'*4097, 123, None]:
            with self.subTest(message_type=type(message).__name__):
                self.assertEqual(self.send(sid,message).status_code,422)
        self.assertEqual(self.client.post('/sessions',json={'model_mode':'bad'}).status_code,422)
        self.assertEqual(self.client.post('/chat',json={'message':'x'}).status_code,422)
        self.assertEqual(len(self.engine.calls),0)

    def test_missing_and_invalid_session(self):
        self.assertEqual(self.send(str(uuid4())).status_code,404)
        self.assertEqual(self.client.get('/sessions/not-uuid').status_code,422)
        self.assertEqual(self.client.delete('/sessions/'+str(uuid4())+'/messages').status_code,404)

    def test_generation_error_does_not_write(self):
        sid=self.session();self.send(sid)
        self.engine.error=RuntimeError('simulated')
        self.assertEqual(self.send(sid).status_code,500)
        self.assertEqual(len(self.client.get('/sessions/'+sid).json()['messages']),2)
        self.engine.error=None
        self.assertEqual(self.send(sid).status_code,200)

    def test_context_error_does_not_write(self):
        sid=self.session();self.engine.error=ValueError('上下文超限')
        self.assertEqual(self.send(sid).status_code,422)
        self.assertEqual(self.client.get('/sessions/'+sid).json()['messages'],[])

    @unittest.skipUnless(importlib.util.find_spec('torch') is not None,'PyTorch OOM type requires the inference dependencies')
    def test_simulated_oom_returns_503_no_partial_turn(self):
        from torch.cuda import OutOfMemoryError
        sid=self.session();self.engine.error=OutOfMemoryError('simulated OOM')
        self.assertEqual(self.send(sid).status_code,503)
        self.assertEqual(self.client.get('/sessions/'+sid).json()['messages'],[])

    def test_storage_failure_returns_503(self):
        from unittest.mock import patch
        sid=self.session()
        with patch.object(self.client.app.state.store,'save_turn',side_effect=sqlite3.OperationalError('simulated')):
            self.assertEqual(self.send(sid).status_code,503)
        self.assertEqual(self.client.get('/sessions/'+sid).json()['messages'],[])

    def test_empty_answer_does_not_write(self):
        sid=self.session();self.engine.answer='  '
        self.assertEqual(self.send(sid).status_code,500)
        self.assertEqual(self.client.get('/sessions/'+sid).json()['messages'],[])

    def test_clear_and_reuse(self):
        sid=self.session();self.send(sid)
        self.assertEqual(self.client.delete('/sessions/'+sid+'/messages').json()['messages'],[])
        self.send(sid)
        self.assertEqual([m['role'] for m in self.engine.calls[-1][0]],['system','user'])

    def test_list_and_pagination_validation(self):
        self.session();self.session()
        self.assertEqual(len(self.client.get('/sessions?limit=1').json()['sessions']),1)
        self.assertEqual(self.client.get('/sessions?limit=0').status_code,422)
        self.assertEqual(self.client.get('/sessions?offset=-1').status_code,422)

    def test_busy_request_and_clear_rejected_health_responsive(self):
        sid=self.session();self.engine.entered=Event();self.engine.release=Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(self.send,sid)
            try:
                self.assertTrue(self.engine.entered.wait(3))
                self.assertEqual(self.send(sid).status_code,409)
                self.assertEqual(self.client.delete('/sessions/'+sid+'/messages').status_code,409)
                self.assertTrue(self.client.get('/health').json()['generation_busy'])
                self.assertEqual(self.client.get('/sessions/'+sid).json()['messages'],[])
            finally:
                self.engine.release.set()
            self.assertEqual(future.result().status_code,200)

    def test_no_lora_when_not_loaded(self):
        self.engine.adapter_available=False
        self.assertEqual(self.client.post('/sessions',json={'model_mode':'lora'}).status_code,400)

    def test_persistent_after_new_app(self):
        sid=self.session();self.send(sid,'持久化检查')
        other=FakeEngine()
        with TestClient(create_app(self.path,engine_factory=lambda:other)) as client:
            self.assertEqual(client.get('/sessions/'+sid).json()['messages'][0]['content'],'持久化检查')
            self.assertEqual(client.post('/chat',json={'session_id':sid,'message':'重启后追问'}).status_code,200)
        self.assertEqual(len(other.calls[0][0]),4)

    def test_store_transaction_rollback(self):
        sid=self.session();store=ChatStore(self.path)
        with self.assertRaises(sqlite3.IntegrityError):
            store.save_turn(sid,'不应留下孤立用户消息',None)
        self.assertEqual(store.get(sid)['messages'],[])

    def test_model_startup_failure_health_503_history_readable(self):
        sid=self.session();self.send(sid)
        def fail(): raise RuntimeError('simulated startup failure')
        with TestClient(create_app(self.path,engine_factory=fail)) as client:
            self.assertEqual(client.get('/health').status_code,503)
            self.assertEqual(client.post('/sessions',json={}).status_code,503)
            self.assertEqual(len(client.get('/sessions/'+sid).json()['messages']),2)


if __name__=='__main__':
    logging.disable(logging.CRITICAL)
    unittest.main(verbosity=2)
