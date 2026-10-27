"""Run with the new environment's Python; exercise the real GPU through the API."""
import importlib.metadata
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    import torch
    from fastapi.testclient import TestClient
    from backend import create_app
    from inference import QwenChat

    prefix=Path(sys.prefix)
    config=(prefix/'pyvenv.cfg').read_text(encoding='utf-8')
    if 'include-system-site-packages = false' not in config.lower():
        raise RuntimeError('This check requires an isolated venv without system packages')
    if not Path(torch.__file__).resolve().is_relative_to(prefix.resolve()):
        raise RuntimeError('Torch was imported from outside the fresh environment')
    check=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    if check.returncode: raise RuntimeError(check.stdout+check.stderr)
    start=time.perf_counter()
    engine=QwenChat(model_profile='qwen3-1.7b',trim_history=True)
    load_seconds=time.perf_counter()-start
    with tempfile.TemporaryDirectory() as directory:
        with TestClient(create_app(Path(directory)/'check.sqlite3',engine_factory=lambda:engine)) as client:
            health=client.get('/health');health.raise_for_status()
            home=client.get('/ui/');assert home.status_code==200 and '拾语' in home.text
            session=client.post('/sessions',json={'character_id':'companion'});session.raise_for_status()
            sid=session.json()['id']
            result=client.post('/chat',json=dict(session_id=sid,message='请用一句简短的话打个招呼。',max_new_tokens=64))
            result.raise_for_status()
            stream=client.post('/chat/stream',json=dict(session_id=sid,message='再用一句话说说你能陪我聊什么。',max_new_tokens=64,timeout_seconds=120))
            stream.raise_for_status()
            if 'event: done' not in stream.text or 'event: error' in stream.text:
                raise RuntimeError('Real GPU streaming did not finish normally')
            saved=client.get('/sessions/'+sid).json()
            assert len(saved['messages'])==4
            report=dict(date='2026-10-23',scope='Same Windows computer, fresh Python venv/site-packages; existing driver, fixed model files and base Python stdlib reused. Not a new machine or Linux deployment.',
                python=sys.version,prefix=sys.prefix,base_prefix=sys.base_prefix,venv_config=config,
                torch_path=torch.__file__,cuda_available=torch.cuda.is_available(),cuda_version=torch.version.cuda,
                packages={name:importlib.metadata.version(name) for name in ('torch','transformers','accelerate','safetensors','fastapi','uvicorn','httpx','gradio','peft')},
                pip_check=check.stdout.strip(),model_load_seconds=load_seconds,
                health=health.json(),sync_result=result.json(),stream_done=True,saved_messages=saved['messages'],
                limitations='Speech runtimes, microphone and MNN are not part of this clean-install GPU/API check. No user database used.')
    target=Path('docs/clean-environment-validation.json')
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Verified fresh environment, real GPU sync/SSE chat and four saved messages: '+str(target),flush=True)


if __name__=='__main__': main()
