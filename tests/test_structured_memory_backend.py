import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Thread

from fastapi.testclient import TestClient

from backend import create_app


class FakeEngine:
    adapter_available=False

    def __init__(self):
        self.calls=[]

    def reply(self,messages,model_mode='base',max_new_tokens=256,generation_overrides=None):
        self.calls.append(messages)
        return '测试回答',dict(seconds=0.01,output_tokens=4)

    def reply_stream(self,messages,model_mode='base',stop_event=None,emit=None,
                     max_new_tokens=256,generation_overrides=None):
        self.calls.append(messages)
        if emit:
            emit('测试回答')
        return '测试回答',dict(seconds=0.01,output_tokens=4)


class FakeDeepSeekProvider:
    model_id='deepseek-flash'

    def __init__(self):
        self.last_existing_facts=[]

    def reply(self,messages,max_new_tokens=512,generation_overrides=None):
        request=json.loads(messages[-1]['content'])
        rows=request['messages']
        self.last_existing_facts=request['existing_facts']
        target=next(item for item in rows if '改到周五' in item['content'])
        answer=json.dumps({'candidates':[{
            'kind':'user_report','subject':'user','fact_key':'user.interview.time',
            'value':'周五','content':'面试改到周五',
            'event_date':None,
            'evidence':[{'message_id':target['id'],'role':'user','quote':target['content']}],
        }]},ensure_ascii=False)
        return answer,dict(seconds=0.01,output_tokens=30,prompt_tokens=80,
                           total_tokens=110,model='deepseek-flash')


class ReverseOrderProvider(FakeDeepSeekProvider):
    def reply(self,messages,max_new_tokens=512,generation_overrides=None):
        request=json.loads(messages[-1]['content'])
        rows=request['messages']
        old=next(item for item in rows if '周三' in item['content'])
        new=next(item for item in rows if '周五' in item['content'])
        candidates=[
            {
                'kind':'user_report','subject':'user',
                'fact_key':'user.interview.time','value':'周五',
                'content':'面试改到周五','event_date':None,
                'evidence':[{'message_id':new['id'],'role':'user',
                             'quote':new['content']}],
            },
            {
                'kind':'user_report','subject':'user',
                'fact_key':'user.interview.time','value':'周三',
                'content':'面试原本在周三','event_date':None,
                'evidence':[{'message_id':old['id'],'role':'user',
                             'quote':old['content']}],
            },
        ]
        return json.dumps({'candidates':candidates},ensure_ascii=False),dict(
            seconds=0.01,output_tokens=60,prompt_tokens=100,
            total_tokens=160,model='deepseek-flash',
        )


class AutomaticFakeDeepSeekProvider:
    model_id='deepseek-flash'

    def reply(self,messages,max_new_tokens=512,generation_overrides=None):
        request=json.loads(messages[-1]['content'])
        rows=request['messages']
        existing=request['existing_facts']
        correction=next((item for item in reversed(rows)
                         if '改到周五' in item['content']),None)
        initial=next((item for item in reversed(rows)
                      if '面试在周三' in item['content']),None)
        target=correction or initial
        if target is None:
            return '{"candidates":[]}',dict(
                seconds=0.01,output_tokens=8,prompt_tokens=50,
                total_tokens=58,model='deepseek-flash',
            )
        value='周五' if correction else '周三'
        old=next((item for item in existing
                  if item['state']=='active'
                  and item['fact_key']=='user.interview.time'),None)
        operation='supersede' if correction and old else 'add'
        answer=json.dumps({'candidates':[{
            'kind':'user_report','subject':'user',
            'fact_key':'current.interview.day',
            'value':value,'content':target['content'],
            'event_date':None,'operation':operation,
            'target_item_id':old['item_id'] if operation=='supersede' else None,
            'evidence':[{
                'message_id':target['id'],'role':'user',
                'quote':target['content'],
            }],
        }]},ensure_ascii=False)
        return answer,dict(seconds=0.01,output_tokens=30,prompt_tokens=80,
                           total_tokens=110,model='deepseek-flash')


class AutomaticStateProvider:
    model_id='deepseek-flash'

    def reply(self,messages,max_new_tokens=512,generation_overrides=None):
        request=json.loads(messages[-1]['content'])
        target=next(
            (item for item in request['messages']
             if '考试通过' in item['content']),
            None,
        )
        if target is None:
            payload={'candidates':[]}
        else:
            payload={
                'candidates':[],
                'state_update':{
                    'significant':True,
                    'event_type':'user_shared_good_news',
                    'importance':'medium',
                    'mood_delta':{'valence':0.2,'arousal':0.2},
                    'relationship_delta':{
                        'familiarity':0.1,
                        'trust':0.1,
                        'warmth':0.1,
                    },
                    'boundary':None,
                    'reason':'用户主动分享了考试通过的消息',
                    'evidence':[{
                        'message_id':target['id'],
                        'role':target['role'],
                        'quote':'考试通过了',
                    }],
                },
            }
        return json.dumps(payload,ensure_ascii=False),dict(
            seconds=0.01,output_tokens=35,prompt_tokens=90,
            total_tokens=125,model='deepseek-flash',
        )


class BlockingDeepSeekProvider:
    model_id='deepseek-flash'

    def __init__(self):
        self.entered=Event()
        self.release=Event()

    def reply(self,messages,max_new_tokens=512,generation_overrides=None):
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError('blocking provider timed out')
        return '{"candidates":[]}',dict(
            seconds=0.01,output_tokens=8,prompt_tokens=20,
            total_tokens=28,model='deepseek-flash',
        )


class CountingEmptyDeepSeekProvider:
    model_id='deepseek-flash'

    def __init__(self):
        self.calls=0
        self.batch_sizes=[]

    def reply(self,messages,max_new_tokens=512,generation_overrides=None):
        request=json.loads(messages[-1]['content'])
        self.calls+=1
        self.batch_sizes.append(len(request['messages']))
        return '{"candidates":[]}',dict(
            seconds=0.01,output_tokens=8,prompt_tokens=20,
            total_tokens=28,model='deepseek-flash',
        )


class StructuredMemoryBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.engine=FakeEngine()
        self.deepseek=FakeDeepSeekProvider()
        self.client=TestClient(create_app(
            self.root/'chat.sqlite3',
            engine_factory=lambda:self.engine,
            deepseek_provider=self.deepseek,
            structured_memory=True,
            structured_memory_db_path=self.root/'structured.sqlite3',
            auto_memory_worker_enabled=False,
        ))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None,None,None)
        self.temp.cleanup()

    def open_conversation(self):
        response=self.client.put(
            '/characters/companion/conversation',
            params={'provider':'local'},
        )
        self.assertEqual(response.status_code,200)
        return response.json()

    def send(self,session_id,text):
        response=self.client.post('/chat',json={'session_id':session_id,'message':text})
        self.assertEqual(response.status_code,200,response.text)
        return response.json()

    def candidate(self,message_id,role,quote,value,fact_key='user.interview.time'):
        return {
            'kind':'user_report',
            'subject':'user',
            'fact_key':fact_key,
            'value':value,
            'content':quote,
            'event_date':None,
            'evidence':[{'message_id':message_id,'role':role,'quote':quote}],
        }

    def test_review_revision_injection_history_and_delete(self):
        current=self.open_conversation()
        self.send(current['id'],'面试在周三')
        self.send(current['id'],'面试改到周五')
        session=self.client.get('/sessions/'+current['id']).json()
        old_message,new_message=session['messages'][1],session['messages'][3]

        old_response=self.client.post(
            '/characters/companion/structured-memory/candidates',
            json=self.candidate(old_message['id'],'user','面试在周三','周三'),
        )
        new_response=self.client.post(
            '/characters/companion/structured-memory/candidates',
            json=self.candidate(new_message['id'],'user','面试改到周五','周五'),
        )
        self.assertEqual(old_response.status_code,201,old_response.text)
        self.assertEqual(new_response.status_code,201,new_response.text)
        old_id,new_id=old_response.json()['item']['id'],new_response.json()['item']['id']
        self.assertEqual(old_response.json()['action'],'added')
        self.assertEqual(new_response.json()['action'],'superseded')

        self.send(current['id'],'面试什么时候')
        prompt=self.engine.calls[-1][0]['content']
        self.assertIn('面试改到周五',prompt)
        self.assertIn('若本轮消息与旧记忆冲突，以本轮消息为准',prompt)
        self.assertNotIn('历史旧值',prompt)
        self.send(current['id'],'面试原来是什么时候')
        history_prompt=self.engine.calls[-1][0]['content']
        self.assertIn('历史旧值（已被替代）',history_prompt)
        self.assertIn('面试在周三',history_prompt)

        rows=self.client.get(
            '/characters/companion/structured-memory',
            params={'include_inactive':'true'},
        ).json()['records']
        states={row['id']:row['state'] for row in rows}
        self.assertEqual(states[old_id],'superseded')
        self.assertEqual(states[new_id],'active')

        self.assertEqual(
            self.client.post('/characters/companion/structured-memory/review',
                             json={'action':'delete','item_id':new_id}).status_code,
            200,
        )
        suppressed=self.client.post(
            '/characters/companion/structured-memory/candidates',
            json=self.candidate(new_message['id'],'user','面试改到周五','周五'),
        )
        self.assertEqual(suppressed.status_code,201,suppressed.text)
        self.assertEqual(suppressed.json()['action'],'ignored_deleted')
        self.send(current['id'],'面试什么时候')
        self.assertNotIn('面试改到周五',self.engine.calls[-1][0]['content'])

    def test_invalid_quote_rejected_and_reset_clears_scope(self):
        current=self.open_conversation()
        self.send(current['id'],'面试在周三')
        message=self.client.get('/sessions/'+current['id']).json()['messages'][1]
        invalid=self.candidate(message['id'],'user','不存在的原文','周五')
        response=self.client.post(
            '/characters/companion/structured-memory/candidates',
            json=invalid,
        )
        self.assertEqual(response.status_code,422)

        valid=self.client.post(
            '/characters/companion/structured-memory/candidates',
            json=self.candidate(message['id'],'user','面试在周三','周三'),
        )
        self.assertEqual(valid.status_code,201)
        self.client.delete('/characters/companion/conversation/messages')
        records=self.client.get(
            '/characters/companion/structured-memory',
            params={'include_inactive':'true'},
        ).json()['records']
        self.assertEqual(records,[])
        self.assertEqual(
            self.client.app.state.structured_memory.job_counts(),
            {},
        )

    def test_disabled_feature_documents_capability(self):
        with TestClient(create_app(
            self.root/'plain.sqlite3',
            engine_factory=lambda:FakeEngine(),
        )) as client:
            self.assertFalse(client.get('/health').json()['structured_memory_enabled'])
            response=client.get('/characters/companion/structured-memory')
            self.assertEqual(response.status_code,404)

    def test_manual_test_mode_does_not_claim_automatic_worker_is_running(self):
        status=self.client.get(
            '/characters/companion/structured-memory/status'
        )
        self.assertEqual(status.status_code,200,status.text)
        self.assertFalse(status.json()['worker_running'])
        self.assertFalse(status.json()['automatic'])

    def test_extract_endpoint_auto_activates_without_review(self):
        current=self.open_conversation()
        self.send(current['id'],'面试在周三')
        self.send(current['id'],'面试改到周五')
        self.assertEqual(
            self.client.put('/settings/deepseek-key',
                            json={'api_key':'sk-test-placeholder'}).status_code,
            200,
        )
        response=self.client.post('/characters/companion/structured-memory/extract')
        self.assertEqual(response.status_code,200,response.text)
        body=response.json()
        self.assertEqual(len(body['created']),1)
        self.assertEqual(body['created'][0]['state'],'active')
        self.assertEqual(body['created'][0]['fact_key'],'user.interview.time')
        self.assertEqual(body['rejected'],[])
        self.send(current['id'],'面试什么时候')
        self.assertIn('面试改到周五',self.engine.calls[-1][0]['content'])
        self.assertEqual(
            self.deepseek.last_existing_facts,
            [],
        )
        second=self.client.post('/characters/companion/structured-memory/extract')
        self.assertEqual(second.status_code,200,second.text)
        self.assertTrue(
            any(item['fact_key']=='user.interview.time'
                for item in self.deepseek.last_existing_facts)
        )

    def test_extraction_is_applied_in_message_order(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            provider=ReverseOrderProvider()
            engine=FakeEngine()
            with TestClient(create_app(
                root/'chat.sqlite3',
                engine_factory=lambda:engine,
                deepseek_provider=provider,
                structured_memory=True,
                structured_memory_db_path=root/'structured.sqlite3',
                auto_memory_worker_enabled=False,
            )) as client:
                client.put(
                    '/settings/deepseek-key',
                    json={'api_key':'sk-test-placeholder'},
                )
                current=client.put(
                    '/characters/companion/conversation',
                    params={'provider':'local'},
                ).json()
                client.post(
                    '/chat',
                    json={'session_id':current['id'],'message':'面试在周三'},
                )
                client.post(
                    '/chat',
                    json={'session_id':current['id'],
                          'message':'面试改到周五'},
                )
                response=client.post(
                    '/characters/companion/structured-memory/extract'
                )
                self.assertEqual(response.status_code,200,response.text)
                records=client.get(
                    '/characters/companion/structured-memory',
                    params={'include_inactive':'true'},
                ).json()['records']
                by_value={item['value']:item['state'] for item in records}
                self.assertEqual(by_value,{'周三':'superseded','周五':'active'})

    def test_structured_memory_suppresses_legacy_excerpt_even_after_delete(self):
        current=self.open_conversation()
        self.send(current['id'],'面试在周三')
        message=self.client.get(
            '/sessions/'+current['id']
        ).json()['messages'][1]
        created=self.client.post(
            '/characters/companion/structured-memory/candidates',
            json=self.candidate(message['id'],'user','面试在周三','周三'),
        )
        self.assertEqual(created.status_code,201,created.text)
        self.client.app.state.store.save_memory(
            current['id'],
            {
                'items':[{
                    'message_id':message['id'],'role':'user',
                    'quote':'面试在周三',
                }],
                'covered_message_id':self.client.get(
                    '/sessions/'+current['id']
                ).json()['messages'][2]['id'],
                'notes':'',
                'auto_enabled':True,
            },
            expected_revision=0,
        )
        self.send(current['id'],'面试什么时候')
        self.assertNotIn(
            '仅本会话的早期记忆',self.engine.calls[-1][0]['content']
        )

        item_id=created.json()['item']['id']
        self.client.post(
            '/characters/companion/structured-memory/review',
            json={'action':'delete','item_id':item_id},
        )
        self.send(current['id'],'还记得面试时间吗')
        prompt=self.engine.calls[-1][0]['content']
        self.assertNotIn('仅本会话的早期记忆',prompt)
        self.assertNotIn('面试在周三',prompt)

    def test_background_worker_commits_memory_without_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            provider=FakeDeepSeekProvider()
            engine=FakeEngine()
            with TestClient(create_app(
                root/'chat.sqlite3',
                engine_factory=lambda:engine,
                deepseek_provider=provider,
                structured_memory=True,
                structured_memory_db_path=root/'structured.sqlite3',
                auto_memory_worker_enabled=True,
                memory_job_ready_seconds=0.1,
            )) as client:
                self.assertEqual(
                    client.put('/settings/deepseek-key',
                               json={'api_key':'sk-test-placeholder'}).status_code,
                    200,
                )
                current=client.put(
                    '/characters/companion/conversation',
                    params={'provider':'local'},
                ).json()
                response=client.post(
                    '/chat',
                    json={'session_id':current['id'],'message':'面试改到周五'},
                )
                self.assertEqual(response.status_code,200,response.text)
                self.assertGreaterEqual(
                    client.app.state.structured_memory.job_counts().get(
                        'queued',0
                    ) + client.app.state.structured_memory.job_counts().get(
                        'running',0
                    ),
                    1,
                )
                deadline=time.monotonic()+5
                records=[]
                while time.monotonic()<deadline:
                    records=client.get(
                        '/characters/companion/structured-memory',
                        params={'include_inactive':'true'},
                    ).json()['records']
                    if any(item['state']=='active'
                           and item['fact_key']=='user.interview.time'
                           for item in records):
                        break
                    time.sleep(0.1)
                self.assertTrue(
                    any(item['state']=='active'
                        and item['fact_key']=='user.interview.time'
                        for item in records),
                    (
                        records,
                        client.app.state.structured_memory.job_counts(),
                        client.app.state.memory_worker is not None,
                        (
                            client.app.state.memory_worker._thread.is_alive()
                            if client.app.state.memory_worker is not None else False
                        ),
                        [
                            dict(row)
                            for row in client.app.state.structured_memory.db.execute(
                                'SELECT * FROM memory_jobs'
                            ).fetchall()
                        ],
                    ),
                )
                active=next(
                    item for item in records
                    if item['state']=='active'
                    and item['fact_key']=='user.interview.time'
                )
                self.assertIsNotNone(active['origin_job_id'])
                status=client.get(
                    '/characters/companion/structured-memory/status'
                )
                self.assertEqual(status.status_code,200,status.text)
                status_body=status.json()
                self.assertTrue(status_body['worker_running'])
                self.assertEqual(status_body['daily_token_limit'],30000)
                self.assertGreaterEqual(
                    status_body['usage_today']['total_tokens'], 1
                )
                health=client.get('/health')
                if health.status_code in (200,503):
                    self.assertEqual(
                        health.json()['deepseek_usage']['total_tokens'], 0
                    )

    def test_stream_completion_enqueues_automatic_memory_job(self):
        current=self.open_conversation()
        with self.client.stream(
            'POST',
            '/chat/stream',
            json={'session_id':current['id'],'message':'面试改到周五'},
        ) as response:
            body=''.join(response.iter_text())
        self.assertEqual(response.status_code,200)
        self.assertIn('event: done',body)
        self.assertEqual(
            self.client.app.state.structured_memory.job_counts().get('queued',0),
            1,
        )

    def test_background_budget_defers_without_touching_chat_flow(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            provider=FakeDeepSeekProvider()
            engine=FakeEngine()
            with TestClient(create_app(
                root/'chat.sqlite3',
                engine_factory=lambda:engine,
                deepseek_provider=provider,
                structured_memory=True,
                structured_memory_db_path=root/'structured.sqlite3',
                auto_memory_worker_enabled=True,
                memory_job_ready_seconds=0.1,
                memory_daily_token_limit=1000,
            )) as client:
                client.put(
                    '/settings/deepseek-key',
                    json={'api_key':'sk-test-placeholder'},
                )
                store=client.app.state.structured_memory
                store.record_memory_usage(
                    'companion','deepseek','deepseek-flash',500,500,
                )
                current=client.put(
                    '/characters/companion/conversation',
                    params={'provider':'local'},
                ).json()
                response=client.post(
                    '/chat',
                    json={'session_id':current['id'],
                          'message':'面试改到周五'},
                )
                self.assertEqual(response.status_code,200,response.text)
                deadline=time.monotonic()+3
                row=None
                while time.monotonic()<deadline:
                    row=store.db.execute(
                        '''SELECT status,last_error,ready_at
                           FROM memory_jobs ORDER BY id DESC LIMIT 1'''
                    ).fetchone()
                    if row and 'budget' in (row['last_error'] or ''):
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(row)
                self.assertEqual(row['status'],'queued')
                self.assertIn('budget',row['last_error'])
                self.assertEqual(
                    store.memory_usage_for_day('companion')['total_tokens'],
                    1000,
                )
                records=client.get(
                    '/characters/companion/structured-memory',
                    params={'include_inactive':'true'},
                ).json()['records']
                self.assertEqual(records,[])

    def test_background_worker_automatically_supersedes_correction(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            provider=AutomaticFakeDeepSeekProvider()
            engine=FakeEngine()
            with TestClient(create_app(
                root/'chat.sqlite3',
                engine_factory=lambda:engine,
                deepseek_provider=provider,
                structured_memory=True,
                structured_memory_db_path=root/'structured.sqlite3',
                auto_memory_worker_enabled=True,
                memory_job_ready_seconds=0.05,
            )) as client:
                client.put(
                    '/settings/deepseek-key',
                    json={'api_key':'sk-test-placeholder'},
                )
                current=client.put(
                    '/characters/companion/conversation',
                    params={'provider':'local'},
                ).json()

                def wait_for(value):
                    deadline=time.monotonic()+5
                    records=[]
                    while time.monotonic()<deadline:
                        records=client.get(
                            '/characters/companion/structured-memory',
                            params={'include_inactive':'true'},
                        ).json()['records']
                        if any(item['state']=='active'
                               and item['value']==value for item in records):
                            return records
                        time.sleep(0.05)
                    self.fail(records)

                self.assertEqual(
                    client.post('/chat',json={
                        'session_id':current['id'],
                        'message':'面试在周三',
                    }).status_code,
                    200,
                )
                wait_for('周三')
                self.assertEqual(
                    client.post('/chat',json={
                        'session_id':current['id'],
                        'message':'面试改到周五',
                    }).status_code,
                    200,
                )
                records=wait_for('周五')
                active=[item for item in records if item['state']=='active']
                superseded=[item for item in records
                            if item['state']=='superseded']
                self.assertEqual(len(active),1)
                self.assertEqual(len(superseded),1)
                self.assertEqual(active[0]['fact_key'],superseded[0]['fact_key'])
                self.assertEqual([item['value'] for item in active],['周五'])
                self.assertEqual([item['value'] for item in superseded],['周三'])
                self.assertEqual(active[0]['supersedes'],superseded[0]['id'])

    def test_background_worker_updates_internal_state_and_renders_natural_prompt(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            provider=AutomaticStateProvider()
            engine=FakeEngine()
            with TestClient(create_app(
                root/'chat.sqlite3',
                engine_factory=lambda:engine,
                deepseek_provider=provider,
                structured_memory=True,
                structured_memory_db_path=root/'structured.sqlite3',
                auto_memory_worker_enabled=True,
                memory_job_ready_seconds=0,
            )) as client:
                client.put(
                    '/settings/deepseek-key',
                    json={'api_key':'sk-test-placeholder'},
                )
                current=client.put(
                    '/characters/companion/conversation',
                    params={'provider':'local'},
                ).json()
                self.assertEqual(
                    client.post('/chat',json={
                        'session_id':current['id'],
                        'message':'我考试通过了，终于松了口气。',
                    }).status_code,
                    200,
                )
                store=client.app.state.structured_memory
                deadline=time.monotonic()+5
                transitions=[]
                while time.monotonic()<deadline:
                    transitions=store.db.execute(
                        'SELECT * FROM state_transitions WHERE character_id=?',
                        ('companion',),
                    ).fetchall()
                    if transitions:
                        break
                    time.sleep(0.05)
                self.assertEqual(len(transitions),1)
                snapshot=store.internal_state.snapshot('companion')
                self.assertGreater(snapshot['mood']['valence'],0)
                self.assertGreater(snapshot['relationship']['familiarity'],0.15)

                self.assertEqual(
                    client.post('/chat',json={
                        'session_id':current['id'],
                        'message':'你还记得刚才发生了什么吗？',
                    }).status_code,
                    200,
                )
                system_prompt=engine.calls[-1][0]['content']
                self.assertIn('角色内部状态',system_prompt)
                self.assertIn('你此刻的心情',system_prompt)
                self.assertIn('不要展示内部数值',system_prompt)
                self.assertNotIn('0.398',system_prompt)
                self.assertNotIn('user_shared_good_news',system_prompt)

    def test_background_model_call_does_not_block_chat(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            provider=BlockingDeepSeekProvider()
            engine=FakeEngine()
            with TestClient(create_app(
                root/'chat.sqlite3',
                engine_factory=lambda:engine,
                deepseek_provider=provider,
                structured_memory=True,
                structured_memory_db_path=root/'structured.sqlite3',
                auto_memory_worker_enabled=True,
                memory_job_ready_seconds=0,
            )) as client:
                client.put(
                    '/settings/deepseek-key',
                    json={'api_key':'sk-test-placeholder'},
                )
                current=client.put(
                    '/characters/companion/conversation',
                    params={'provider':'local'},
                ).json()
                first=client.post(
                    '/chat',
                    json={'session_id':current['id'],'message':'第一次消息'},
                )
                self.assertEqual(first.status_code,200,first.text)
                self.assertTrue(provider.entered.wait(2))

                result={}
                def send_again():
                    result['response']=client.post(
                        '/chat',
                        json={
                            'session_id':current['id'],
                            'message':'后台分析时仍然可以聊天',
                        },
                    )

                thread=Thread(target=send_again,daemon=True)
                thread.start()
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())
                self.assertEqual(
                    result['response'].status_code,200,
                    result['response'].text,
                )
                provider.release.set()

    def test_backlog_larger_than_window_is_processed_in_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            provider=CountingEmptyDeepSeekProvider()
            engine=FakeEngine()
            with TestClient(create_app(
                root/'chat.sqlite3',
                engine_factory=lambda:engine,
                deepseek_provider=provider,
                structured_memory=True,
                structured_memory_db_path=root/'structured.sqlite3',
                auto_memory_worker_enabled=True,
                memory_job_ready_seconds=0,
            )) as client:
                client.put(
                    '/settings/deepseek-key',
                    json={'api_key':'sk-test-placeholder'},
                )
                current=client.put(
                    '/characters/companion/conversation',
                    params={'provider':'local'},
                ).json()
                for index in range(15):
                    client.app.state.store.save_turn(
                        current['id'],
                        f'问题{index}',
                        f'回答{index}',
                    )
                session=client.app.state.store.get(current['id'])
                end_id=session['messages'][-1]['id']
                store=client.app.state.structured_memory
                store.enqueue_auto_extraction(
                    'companion',end_id,ready_in=0,
                )
                client.app.state.memory_worker.wake()
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    if store.job_counts().get('completed',0)==1:
                        break
                    time.sleep(0.05)
                self.assertEqual(store.job_counts().get('completed',0),1)
                self.assertEqual(store.last_processed_message_id('companion'),end_id)
                self.assertGreaterEqual(provider.calls,2)
                self.assertLessEqual(max(provider.batch_sizes),24)
                self.assertGreaterEqual(max(provider.batch_sizes),16)


if __name__ == '__main__':
    unittest.main()
