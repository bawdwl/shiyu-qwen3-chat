"""真实本地模型回答→TTS接口→PCM WAV验收。"""
import io
import json
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
import httpx
from speech import validate_wav


def main():
    with httpx.Client(base_url='http://127.0.0.1:8000', timeout=135) as client:
        health=client.get('/health');health.raise_for_status()
        assert health.json()['speech_provider']
        created=client.post('/sessions',json={});created.raise_for_status()
        sid=created.json()['id']
        response=client.post('/chat',json=dict(session_id=sid,message='请只回复：你好，我可以帮助你学习。',max_new_tokens=64))
        response.raise_for_status()
        saved=client.get('/sessions/'+sid).json()
        assert len(saved['messages'])==2
        target=saved['messages'][-1]
        assert client.post('/sessions/'+sid+'/speech',json={'message_id':saved['messages'][0]['id']}).status_code==404
        start=time.perf_counter()
        audio=client.post('/sessions/'+sid+'/speech',json={'message_id':target['id']})
        audio.raise_for_status()
        elapsed=time.perf_counter()-start
        validate_wav(audio.content)
        assert audio.headers['x-message-id']==str(target['id'])
        with wave.open(io.BytesIO(audio.content),'rb') as wav:
            sample_rate=wav.getframerate();duration=wav.getnframes()/sample_rate
            assert any(wav.readframes(wav.getnframes()))
        assert client.get('/sessions/'+sid).json()['messages']==saved['messages']
        assert not client.get('/health').json()['generation_busy']
        root=Path('outputs/speech-demo');root.mkdir(exist_ok=True)
        (root/'chat-answer.wav').write_bytes(audio.content)
        record=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),provider=health.json()['speech_provider'],
            question=saved['messages'][0]['content'],answer=target['content'],message_id=target['id'],
            bytes=len(audio.content),sample_rate=sample_rate,audio_seconds=duration,
            request_wall_seconds=elapsed,history_unchanged=True,gate_released=True,
            note='Real local Qwen answer and MeloTTS CPU synthesis; no GPT-SoVITS server, no human listening score or real-time latency claim.')
        Path('docs/speech-smoke-result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
        client.delete('/sessions/'+sid+'/messages').raise_for_status()
        print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
