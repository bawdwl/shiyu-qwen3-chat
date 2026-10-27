"""SSE/取消/超时与提交语义测试，使用模拟流式引擎。"""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

from fastapi.testclient import TestClient
from backend import create_app


class FakeStream:
    adapter_available=True
    def __init__(self):
        self.slow=False;self.fail=False;self.entered=Event();self.allow_exit=None
    def reply_stream(self,messages,mode,stop,emit,max_new_tokens=256,generation_overrides=None):
        self.last_limit=max_new_tokens
        self.last_overrides=generation_overrides
        self.entered.set();emit('中文');emit('\n回答')
        if self.slow:
            stop.wait(3)
            if self.allow_exit: self.allow_exit.wait(3)
        if self.fail: raise RuntimeError('simulated')
        return '中文\n回答',dict(seconds=0.1,output_tokens=3,ended_with_eos=True)


def parse(text):
    results=[]
    for block in text.split('\n\n'):
        fields=block.splitlines()
        if not fields: continue
        event=next(x[7:] for x in fields if x.startswith('event: '))
        data=json.loads(next(x[6:] for x in fields if x.startswith('data: ')))
        results.append((event,data))
    return results


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.engine=FakeStream()
        self.client=TestClient(create_app(Path(self.temp.name)/'chat.sqlite3',engine_factory=lambda:self.engine))
        self.client.__enter__()
        self.sid=self.client.post('/sessions',json={}).json()['id']
    def tearDown(self):
        self.client.__exit__(None,None,None);self.temp.cleanup()
    def send(self,timeout=2):
        return self.client.post('/chat/stream',json=dict(session_id=self.sid,message='问题',timeout_seconds=timeout))
    def history(self):
        return self.client.get('/sessions/'+self.sid).json()['messages']

    def test_success_stream_order_and_persistence(self):
        r=self.send();self.assertIn('text/event-stream',r.headers['content-type'])
        events=parse(r.text)
        self.assertEqual([x[0] for x in events],['start','delta','delta','done'])
        self.assertEqual(''.join(d['text'] for e,d in events if e=='delta'),'中文\n回答')
        self.assertEqual(len(self.history()),2)

    def test_timeout_no_partial_commit_and_reuse(self):
        self.engine.slow=True
        events=parse(self.send(.1).text)
        self.assertEqual(events[-1][1]['code'],'timeout');self.assertEqual(self.history(),[])
        self.engine.slow=False
        self.assertEqual(parse(self.send().text)[-1][0],'done')

    def test_stop_waits_for_actual_worker_exit(self):
        self.engine.slow=True;self.engine.allow_exit=Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(self.send)
            try:
                self.assertTrue(self.engine.entered.wait(2))
                r=self.client.post('/sessions/'+self.sid+'/stop')
                self.assertTrue(r.json()['stop_requested'])
                self.assertTrue(self.client.get('/health').json()['generation_busy'])
                self.assertEqual(self.client.delete('/sessions/'+self.sid+'/messages').status_code,409)
            finally: self.engine.allow_exit.set()
            events=parse(future.result().text)
        self.assertEqual(events[-1][1]['code'],'stopped');self.assertEqual(self.history(),[])

    def test_error_event_not_done_not_saved(self):
        self.engine.fail=True
        events=parse(self.send().text)
        self.assertEqual(events[-1][0],'error');self.assertNotIn('done',[e for e,d in events])
        self.assertEqual(self.history(),[])

    def test_timeout_bounds_and_invalid_input(self):
        for value in [0,301]:
            r=self.client.post('/chat/stream',json=dict(session_id=self.sid,message='x',timeout_seconds=value))
            self.assertEqual(r.status_code,422)
        self.assertEqual(self.client.post('/chat/stream',json=dict(session_id=self.sid,message=' ')).status_code,422)

    def test_stop_idle_and_other_session(self):
        self.assertFalse(self.client.post('/sessions/'+self.sid+'/stop').json()['stop_requested'])
        self.engine.slow=True
        other=self.client.post('/sessions',json={}).json()['id']
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(self.send)
            self.assertTrue(self.engine.entered.wait(2))
            self.assertFalse(self.client.post('/sessions/'+other+'/stop').json()['stop_requested'])
            self.client.post('/sessions/'+self.sid+'/stop')
            self.assertEqual(parse(future.result().text)[-1][1]['code'],'stopped')


if __name__=='__main__': unittest.main(verbosity=2)
