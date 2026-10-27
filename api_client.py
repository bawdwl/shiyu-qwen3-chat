"""网页HTTP客户端，不加载权重。"""
import json
import time
from threading import Event,Lock
import httpx


class BackendClient:
    def __init__(self,url='http://127.0.0.1:8000'):
        self.url=url.rstrip('/')
        self._stops={};self._lock=Lock()

    def prepare(self,sid):
        # 在网页首次yield前登记，覆盖“按钮已出现、SSE尚未发出”的停止窗口。
        with self._lock: self._stops[sid]=Event()

    def stop(self,sid):
        with self._lock:
            pending=self._stops.get(sid)
            if pending: pending.set()
        result=self.request('POST',f'/sessions/{sid}/stop')
        result['stop_requested']=result['stop_requested'] or pending is not None
        return result

    def request(self,method,path,**kwargs):
        with httpx.Client(base_url=self.url,timeout=10) as client:
            response=client.request(method,path,**kwargs)
            if response.is_error:
                raise RuntimeError(f'后端返回{response.status_code}：{response.text}')
            return response.json()

    def wait_idle(self,seconds=10):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if not self.request('GET','/health')['generation_busy']: return
            time.sleep(0.05)
        raise RuntimeError('后端仍在退出生成，请稍后再试。')

    def transcribe(self,path):
        from pathlib import Path
        from recognition import MAX_AUDIO_BYTES
        source=Path(path)
        if source.stat().st_size>MAX_AUDIO_BYTES: raise ValueError('音频超过8MiB')
        content=source.read_bytes()
        with httpx.Client(base_url=self.url,timeout=135) as client:
            response=client.post('/transcribe',content=content,headers={'Content-Type':'audio/wav'})
            if response.is_error: raise RuntimeError(f'识别服务返回{response.status_code}：{response.text}')
            return response.json()

    def speech(self,sid,message_id):
        with httpx.Client(base_url=self.url,timeout=135) as client:
            response=client.post(f'/sessions/{sid}/speech',json={'message_id':message_id})
            if response.is_error:
                raise RuntimeError(f'语音服务返回{response.status_code}：{response.text}')
            if response.headers.get('X-Message-ID')!=str(message_id):
                raise RuntimeError('语音与所选回答不匹配')
            from speech import validate_wav
            return validate_wav(response.content)

    def stream(self,sid,message,timeout=60,max_new_tokens=256):
        terminal=False
        with self._lock: pending=self._stops.get(sid) or Event()
        stop_sent=False
        try:
            if pending.is_set(): raise RuntimeError('生成已停止，尚未发送到模型。')
            with httpx.Client(base_url=self.url,timeout=httpx.Timeout(timeout+15,connect=5)) as client:
                with client.stream('POST','/chat/stream',json=dict(session_id=sid,message=message,timeout_seconds=timeout,
                                                               max_new_tokens=max_new_tokens)) as response:
                    if response.is_error:
                        response.read();raise RuntimeError(f'后端返回{response.status_code}：{response.text}')
                    event=None;data=[]
                    for line in response.iter_lines():
                        if pending.is_set() and not stop_sent:
                            self.request('POST',f'/sessions/{sid}/stop');stop_sent=True
                        if line.startswith('event:'): event=line[6:].strip()
                        elif line.startswith('data:'): data.append(line[5:].strip())
                        elif not line and data:
                            payload=json.loads('\n'.join(data));data=[]
                            if event in ('done','error'): terminal=True
                            yield event,payload
                    if not terminal: raise RuntimeError('连接中断，未收到完成事件；请重新读取历史。')
        finally:
            with self._lock: self._stops.pop(sid,None)
            if not terminal:
                try: self.request('POST',f'/sessions/{sid}/stop')
                except Exception: pass
