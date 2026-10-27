import io
import tempfile
import unittest
import wave
from pathlib import Path
from fastapi.testclient import TestClient
from backend import create_app
from recognition import validate_input,MAX_AUDIO_BYTES

def audio(seconds=.1,silent=False):
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as output:
        output.setnchannels(1);output.setsampwidth(2);output.setframerate(16000)
        output.writeframes((b'\0\0' if silent else b'\x00\x10')*int(seconds*16000))
    return buffer.getvalue()

class Engine:
    adapter_available=True

class Recognizer:
    label='Simulated ASR for contract tests only'
    def __init__(self): self.calls=0;self.error=None
    def transcribe(self,content):
        self.calls+=1
        if self.error: raise self.error
        return dict(text='测试文字',audio_seconds=validate_input(content),provider=self.label)

class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.asr=Recognizer()
        self.client=TestClient(create_app(Path(self.temp.name)/'chat.db',engine_factory=Engine,asr_factory=lambda:self.asr))
        self.client.__enter__()
    def tearDown(self):
        self.client.__exit__(None,None,None);self.temp.cleanup()
    def test_transcription_does_not_create_conversation(self):
        before=self.client.get('/sessions').json()
        response=self.client.post('/transcribe',content=audio())
        self.assertEqual(response.status_code,200);self.assertEqual(response.json()['text'],'测试文字')
        self.assertEqual(self.client.get('/sessions').json(),before)
        self.assertFalse(self.client.get('/health').json()['generation_busy'])
    def test_invalid_silent_truncated_and_long_audio_rejected(self):
        for content in (b'',audio(silent=True),audio()[:-2],audio(30.1)):
            with self.subTest(size=len(content)):
                self.assertEqual(self.client.post('/transcribe',content=content).status_code,422)
        self.assertEqual(self.asr.calls,0)
    def test_oversized_body_rejected(self):
        self.assertEqual(self.client.post('/transcribe',content=b'x'*(MAX_AUDIO_BYTES+1)).status_code,413)
        self.assertEqual(self.asr.calls,0)
    def test_busy_rejected_and_failure_releases_gate(self):
        self.client.app.state.gate.acquire()
        try: self.assertEqual(self.client.post('/transcribe',content=audio()).status_code,409)
        finally: self.client.app.state.gate.release()
        self.asr.error=RuntimeError('simulated failure')
        self.assertEqual(self.client.post('/transcribe',content=audio()).status_code,503)
        self.assertFalse(self.client.get('/health').json()['generation_busy'])
    def test_unconfigured_asr_preserves_text_service(self):
        self.client.app.state.asr=None
        self.assertEqual(self.client.post('/transcribe',content=audio()).status_code,503)
        self.assertEqual(self.client.get('/health').status_code,200)

if __name__=='__main__': unittest.main()
