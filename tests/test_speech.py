import io
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from backend import create_app
from speech import GPTSoVITSSpeech, ProfiledSpeech, speech_text, validate_wav


def wav_bytes():
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as audio:
        audio.setnchannels(1);audio.setsampwidth(2);audio.setframerate(16000)
        audio.writeframes(b'\x01\x00'*160)
    return buffer.getvalue()


class FakeSpeech:
    label = 'Explicit simulated voice'

    def __init__(self):
        self.calls = [];self.profile_calls=[];self.error = None;self.content = wav_bytes()

    def synthesize(self, text):
        self.calls.append(text)
        if self.error:
            raise self.error
        return self.content

    def synthesize_for_profile(self,text,profile):
        self.profile_calls.append((text,profile['reference_text']))
        return self.synthesize(text)


class FakeEngine:
    adapter_available = True


class SpeechTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.provider = FakeSpeech()
        self.client = TestClient(create_app(Path(self.temp.name)/'chat.sqlite3',
            engine_factory=FakeEngine, speech_factory=lambda: self.provider))
        self.client.__enter__()
        self.sid = self.client.post('/sessions', json={'character_id':'companion'}).json()['id']
        self.store = self.client.app.state.store
        self.store.save_turn(self.sid, '问题', '已保存的回答')
        self.messages = self.store.get(self.sid)['messages']

    def tearDown(self):
        self.client.__exit__(None, None, None);self.temp.cleanup()

    def request(self, sid=None, message_id=None):
        return self.client.post('/sessions/'+(sid or self.sid)+'/speech',
            json={'message_id': message_id if message_id is not None else self.messages[-1]['id']})

    def test_saved_assistant_audio_matches_id_without_changing_history(self):
        response = self.request()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers['x-message-id'], str(self.messages[-1]['id']))
        self.assertEqual(response.headers['content-type'], 'audio/wav')
        self.assertEqual(response.content, wav_bytes())
        self.assertEqual(self.provider.calls, ['已保存的回答'])
        self.assertEqual(self.store.get(self.sid)['messages'], self.messages)

    def test_tts_skips_parenthetical_stage_directions(self):
        self.provider.calls.clear()
        saved = self.store.save_turn(
            self.sid,
            '你开心吗？',
            '（别过头，却忍不住笑）还行吧。',
        )
        assistant_id = saved['assistant_message_id']
        response = self.request(message_id=assistant_id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.provider.calls, ['还行吧。'])

    def test_user_and_other_session_message_cannot_be_spoken(self):
        self.assertEqual(self.request(message_id=self.messages[0]['id']).status_code, 404)
        other = self.client.post('/sessions', json={}).json()['id']
        self.assertEqual(self.request(sid=other).status_code, 404)
        self.assertEqual(self.provider.calls, [])

    def test_xiaoman_voice_profile_can_be_saved_and_wav_uploaded(self):
        profile_url='/characters/companion/voice-profile'
        initial=self.client.get(profile_url).json()
        self.assertEqual(initial['display_name'],'小满 · 日常音色')
        self.assertFalse(initial['has_reference_audio'])
        saved=self.client.put(profile_url,json={
            'display_name':'小满 · 温柔日常','voice_style':'温柔自然，语速适中',
            'reference_text':'你好，我是小满。','language':'zh'})
        self.assertEqual(saved.status_code,200)
        uploaded=self.client.put(profile_url+'/reference-audio',content=wav_bytes(),
                                 headers={'Content-Type':'audio/wav'})
        self.assertEqual(uploaded.status_code,200,uploaded.text)
        current=self.client.get(profile_url).json()
        self.assertTrue(current['gpt_sovits_ready'])
        stored=self.client.app.state.store.get_voice_profile('companion')
        self.assertEqual(Path(stored['reference_audio_path']).read_bytes(),wav_bytes())
        self.provider.supports_voice_profiles=True
        spoken=self.request()
        self.assertEqual(spoken.status_code,200)
        self.assertEqual(self.provider.profile_calls,[('已保存的回答','你好，我是小满。')])

    def test_xiaoman_gpt_sovits_refuses_unconfigured_profile_instead_of_using_wrong_voice(self):
        self.client.app.state.speech.supports_voice_profiles=True
        response=self.request()
        self.assertEqual(response.status_code,409)
        self.assertEqual(self.provider.calls,[])

    def test_busy_and_invalid_inputs_do_not_synthesize(self):
        for value in (0, True, '1', 1.0):
            response = self.client.post('/sessions/'+self.sid+'/speech', json={'message_id': value})
            self.assertEqual(response.status_code, 422)
        gate = self.client.app.state.gate
        gate.acquire()
        try:
            self.assertEqual(self.request().status_code, 409)
        finally:
            gate.release()
        self.assertEqual(self.provider.calls, [])

    def test_provider_failure_and_invalid_audio_preserve_text_and_release_gate(self):
        self.provider.error = RuntimeError('simulated TTS outage')
        self.assertEqual(self.request().status_code, 503)
        self.provider.error = None;self.provider.content = b'not wav'
        self.assertEqual(self.request().status_code, 422)
        self.assertEqual(self.store.get(self.sid)['messages'], self.messages)
        self.assertFalse(self.client.app.state.gate.locked())

    def test_missing_provider_keeps_text_health_available(self):
        self.client.app.state.speech = None
        self.assertEqual(self.request().status_code, 503)
        self.assertEqual(self.client.get('/health').status_code, 200)

    def test_markdown_cleanup_and_no_silent_overlength_truncation(self):
        self.assertEqual(speech_text('# 标题\n**你好** [说明](https://example.com)'), '标题 你好 说明')
        self.assertNotIn('print', speech_text('```python\nprint(1)\n```'))
        with self.assertRaises(ValueError):
            speech_text('字'*2001)
        with self.assertRaises(ValueError):
            validate_wav(b'')

    def test_gpt_sovits_contract_is_local_and_uses_nonstreaming_wav(self):
        requests = []
        real_client = httpx.Client

        def respond(request):
            import json
            requests.append((request.url.path, json.loads(request.content)))
            return httpx.Response(200, content=wav_bytes())

        provider = GPTSoVITSSpeech('http://127.0.0.1:9880', 'reference.wav', '参考音频文字')
        with patch('speech.httpx.Client', side_effect=lambda **kw: real_client(
                transport=httpx.MockTransport(respond), **kw)):
            self.assertEqual(provider.synthesize('你好'), wav_bytes())
        self.assertEqual(requests[0][0], '/tts')
        self.assertFalse(requests[0][1]['streaming_mode'])
        self.assertEqual(requests[0][1]['ref_audio_path'], 'reference.wav')
        self.assertEqual(requests[0][1]['media_type'], 'wav')
        with self.assertRaises(ValueError):
            GPTSoVITSSpeech('https://example.com', 'reference.wav', '文字')
        profile_only=GPTSoVITSSpeech('http://127.0.0.1:9880',None,'')
        self.assertTrue(profile_only.supports_voice_profiles)
        with self.assertRaisesRegex(ValueError,'参考音频'):
            profile_only.synthesize('你好')

    def test_gpt_sovits_uses_selected_character_profile(self):
        requests=[]
        real_client=httpx.Client
        def respond(request):
            import json
            requests.append(json.loads(request.content))
            return httpx.Response(200,content=wav_bytes())
        with tempfile.TemporaryDirectory() as folder:
            audio=Path(folder)/'xiaoman.wav';audio.write_bytes(wav_bytes())
            provider=GPTSoVITSSpeech('http://127.0.0.1:9880','default.wav','默认文本')
            profile={'reference_audio_path':str(audio),'reference_text':'小满专属逐字稿','language':'zh'}
            with patch('speech.httpx.Client',side_effect=lambda **kw: real_client(
                    transport=httpx.MockTransport(respond),**kw)):
                self.assertEqual(provider.synthesize_for_profile('今天辛苦了。',profile),wav_bytes())
        self.assertEqual(requests[0]['ref_audio_path'],str(audio))
        self.assertEqual(requests[0]['prompt_text'],'小满专属逐字稿')
        self.assertEqual(requests[0]['text'],'今天辛苦了。')

    def test_failed_speech_initialization_does_not_prevent_text_startup(self):
        def fail():
            raise RuntimeError('simulated invalid speech configuration')
        with TestClient(create_app(Path(self.temp.name)/'other.sqlite3', engine_factory=FakeEngine,
                                  speech_factory=fail)) as client:
            self.assertEqual(client.get('/health').status_code, 200)
            self.assertIsNone(client.get('/health').json()['speech_provider'])

    def test_profiled_speech_routes_ready_voice_to_gpt_sovits(self):
        fallback=FakeSpeech();profile_provider=FakeSpeech()
        with tempfile.TemporaryDirectory() as folder:
            audio=Path(folder)/'reference.wav';audio.write_bytes(wav_bytes())
            provider=ProfiledSpeech(fallback,profile_provider)
            profile={'reference_audio_path':str(audio),'reference_text':'逐字稿'}
            provider.synthesize_for_profile('你好。',profile)
        self.assertEqual(profile_provider.profile_calls,[('你好。','逐字稿')])
        self.assertEqual(fallback.calls,[])

    def test_profiled_speech_falls_back_when_role_has_no_reference(self):
        fallback=FakeSpeech();profile_provider=FakeSpeech()
        provider=ProfiledSpeech(fallback,profile_provider)
        provider.synthesize_for_profile('你好。',{'reference_audio_path':None,'reference_text':''})
        self.assertEqual(fallback.calls,['你好。'])
        self.assertEqual(profile_provider.calls,[])

    def test_profiled_speech_endpoint_uses_voice_when_configured_and_falls_back_otherwise(self):
        fallback=FakeSpeech();profile_provider=FakeSpeech()
        provider=ProfiledSpeech(fallback,profile_provider)
        db=Path(self.temp.name)/'profiled.sqlite3'
        with tempfile.TemporaryDirectory() as folder:
            reference=Path(folder)/'reference.wav';reference.write_bytes(wav_bytes())
            with TestClient(create_app(db,engine_factory=FakeEngine,speech_factory=lambda:provider)) as client:
                session=client.post('/sessions',json={'character_id':'interview'}).json()
                store=client.app.state.store
                store.save_voice_profile('interview','言川','自然','参考逐字稿')
                store.set_voice_reference_audio('interview',reference)
                store.save_turn(session['id'],'问候','今天过得还好吗？')
                answer=store.get(session['id'])['messages'][-1]
                response=client.post(f"/sessions/{session['id']}/speech",json={'message_id':answer['id']})
                self.assertEqual(response.status_code,200)
                self.assertEqual(profile_provider.profile_calls,[('今天过得还好吗？','参考逐字稿')])

                other=client.post('/sessions',json={'character_id':'companion'}).json()
                store.save_voice_profile('companion','小满','温柔','')
                store.save_turn(other['id'],'问候','辛苦啦。')
                other_answer=store.get(other['id'])['messages'][-1]
                response=client.post(f"/sessions/{other['id']}/speech",json={'message_id':other_answer['id']})
                self.assertEqual(response.status_code,200)
                self.assertEqual(fallback.calls,['辛苦啦。'])


if __name__ == '__main__':
    unittest.main()
