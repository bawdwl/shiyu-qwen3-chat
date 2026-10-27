"""真实HTTP/GPU验证：逐段输出、停止、超时、断开后清理。"""
import json
import time
from pathlib import Path
import httpx


def events(response):
    kind=None;data=[]
    for line in response.iter_lines():
        if line.startswith('event:'): kind=line[6:].strip()
        elif line.startswith('data:'): data.append(line[5:].strip())
        elif not line and data:
            yield kind,json.loads('\n'.join(data));data=[]


def main():
    result={}
    with httpx.Client(base_url='http://127.0.0.1:8000',timeout=90) as client:
        def session(mode='base'):
            r=client.post('/sessions',json={'model_mode':mode});r.raise_for_status();return r.json()['id']
        def history(sid): return client.get('/sessions/'+sid).json()['messages']
        def idle():
            started=time.monotonic()
            while time.monotonic()-started<15:
                if not client.get('/health').json()['generation_busy']: return time.monotonic()-started
                time.sleep(.05)
            raise AssertionError('worker did not exit')
        successes=[]
        for mode in client.get('/health').json()['model_modes']:
            sid=session(mode);chunks=[];done=None;started=time.perf_counter();first=None
            with client.stream('POST','/chat/stream',json=dict(session_id=sid,message='用两句话解释为什么要记录实验配置。',timeout_seconds=60)) as response:
                response.raise_for_status()
                for kind,data in events(response):
                    if kind=='delta':
                        first=first or time.perf_counter()-started;chunks.append(data['text'])
                    if kind=='error': raise AssertionError(data)
                    if kind=='done': done=data
            assert done and len(chunks)>1
            assert ''.join(chunks).strip()==done['answer'] and len(history(sid))==2
            successes.append(dict(mode=mode,chunk_count=len(chunks),first_client_text_seconds=first,done=done))
        result['success']=successes
        long_prompt='请详细写一篇关于本地模型开发流程的长文，逐项展开数据、推理、训练和评估，不要只给概括。'
        sid=session();stop_response=None;terminal=None
        with client.stream('POST','/chat/stream',json=dict(session_id=sid,message=long_prompt,timeout_seconds=60)) as response:
            response.raise_for_status()
            for kind,data in events(response):
                if kind=='delta' and stop_response is None:
                    stop_response=client.post('/sessions/'+sid+'/stop').json()
                    assert stop_response['stop_requested']
                if kind in ('done','error'): terminal=(kind,data)
        stop_wait=idle()
        assert terminal[0]=='error' and terminal[1]['code']=='stopped' and history(sid)==[]
        result['stop']=dict(stop_response=stop_response,terminal=terminal,history_empty=True,wait_idle_seconds=stop_wait)
        sid=session()
        with client.stream('POST','/chat/stream',json=dict(session_id=sid,message=long_prompt,timeout_seconds=.1)) as response:
            response.raise_for_status();timed=list(events(response))
        idle();assert timed[-1][0]=='error' and timed[-1][1]['code']=='timeout' and history(sid)==[]
        result['timeout']=dict(terminal=timed[-1],history_empty=True)
        sid=session();saw_text=False
        with client.stream('POST','/chat/stream',json=dict(session_id=sid,message=long_prompt,timeout_seconds=60)) as response:
            response.raise_for_status()
            for kind,data in events(response):
                if kind=='delta': saw_text=True;break
        disconnected_wait=idle()
        assert saw_text and history(sid)==[]
        result['disconnect']=dict(first_text_observed=True,history_empty=True,worker_exited=True,wait_idle_seconds=disconnected_wait)
        # 停止/异常后再次成功生成，确认门锁和Adapter开关恢复。
        sid=session('lora')
        with client.stream('POST','/chat/stream',json=dict(session_id=sid,message='只回复：你好',timeout_seconds=60)) as response:
            response.raise_for_status();restored=list(events(response))
        assert restored[-1][0]=='done' and len(history(sid))==2
        result['recovery']=dict(lora_completed=True)
    Path('docs/streaming-smoke-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
