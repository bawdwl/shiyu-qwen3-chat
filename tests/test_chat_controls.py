"""回答长度请求隔离、标题回读及客户端参数传递。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from api_client import BackendClient
from backend import create_app
from generation_options import for_request


class RecordingEngine:
    adapter_available = True

    def __init__(self):
        self.limits = []

    def reply(self, messages, model_mode='base', max_new_tokens=256):
        self.limits.append(max_new_tokens)
        return '回答', {'max_new_tokens': max_new_tokens}

    def reply_stream(self, messages, mode, stop, emit, max_new_tokens=256):
        self.limits.append(max_new_tokens)
        emit('回答')
        return '回答', {'max_new_tokens': max_new_tokens}


class ChatControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.engine = RecordingEngine()
        self.client = TestClient(create_app(Path(self.temp.name)/'chat.sqlite3', engine_factory=lambda: self.engine))
        self.client.__enter__()
        self.sid = self.client.post('/sessions', json={}).json()['id']

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def test_sync_limit_is_per_request_and_legacy_default_is_256(self):
        body = dict(session_id=self.sid, message='问题', max_new_tokens=512)
        self.assertEqual(self.client.post('/chat', json=body).status_code, 200)
        del body['max_new_tokens']
        self.assertEqual(self.client.post('/chat', json=body).status_code, 200)
        self.assertEqual(self.engine.limits, [512, 256])

    def test_stream_limit_and_extended_timeout_reach_worker(self):
        body = dict(session_id=self.sid, message='问题', max_new_tokens=1024, timeout_seconds=240)
        self.assertIn('event: done', self.client.post('/chat/stream', json=body).text)
        del body['max_new_tokens']
        self.assertIn('event: done', self.client.post('/chat/stream', json=body).text)
        self.assertEqual(self.engine.limits, [1024, 256])

    def test_invalid_limits_never_generate_or_save(self):
        for endpoint in ('/chat', '/chat/stream'):
            for limit in (0, 1025, True, 512.0, '512', None):
                response = self.client.post(endpoint, json=dict(session_id=self.sid, message='问题', max_new_tokens=limit))
                self.assertEqual(response.status_code, 422)
        self.assertEqual(self.engine.limits, [])
        self.assertEqual(self.client.get('/sessions/'+self.sid).json()['messages'], [])

    def test_first_question_title_and_new_session_preserve_old_history(self):
        self.assertEqual(self.client.get('/sessions').json()['sessions'][0]['title'], '新对话')
        question = '明天面试\n    如何准备？'
        self.client.post('/chat', json=dict(session_id=self.sid, message=question))
        new_id = self.client.post('/sessions', json={}).json()['id']
        self.client.post('/chat', json=dict(session_id=new_id, message='另外一个话题'))
        listed = {r['id']: r for r in self.client.get('/sessions').json()['sessions']}
        self.assertEqual(listed[self.sid]['title'], '明天面试 如何准备？')
        self.assertEqual(len(self.client.get('/sessions/'+self.sid).json()['messages']), 2)
        self.client.delete('/sessions/'+new_id+'/messages')
        listed = {r['id']: r for r in self.client.get('/sessions').json()['sessions']}
        self.assertEqual(listed[new_id]['title'], '新对话')
        self.assertEqual(listed[self.sid]['title'], '明天面试 如何准备？')

    def test_generation_settings_copy_does_not_change_experiment_defaults(self):
        defaults = dict(max_new_tokens=256, temperature=0.7)
        self.assertEqual(for_request(defaults, 1024)['max_new_tokens'], 1024)
        self.assertEqual(defaults['max_new_tokens'], 256)
        self.assertEqual(for_request(defaults)['max_new_tokens'], 256)

    def test_client_sends_selected_limit_and_timeout(self):
        received = []
        real_client = httpx.Client

        def respond(request):
            received.append(json.loads(request.content))
            return httpx.Response(200, text='event: done\ndata: {}\n\n', headers={'content-type': 'text/event-stream'})

        with patch('api_client.httpx.Client', side_effect=lambda **kwargs: real_client(
                transport=httpx.MockTransport(respond), **kwargs)):
            self.assertEqual(list(BackendClient().stream('fixture', '问题', timeout=240, max_new_tokens=1024)), [('done', {})])
        self.assertEqual(received[0]['max_new_tokens'], 1024)
        self.assertEqual(received[0]['timeout_seconds'], 240)


if __name__ == '__main__':
    unittest.main()
