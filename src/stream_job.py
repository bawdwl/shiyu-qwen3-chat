"""后台生成任务：协作停止，任务结束后才释放服务门锁。"""
import json
import logging
from queue import Queue
from threading import Event, Lock, Thread, Timer
from characters import conversation_messages


class GenerationStopped(RuntimeError):
    pass


class StreamJob:
    def __init__(self,engine,store,current,message,gate,timeout,max_new_tokens=256,policy_reply=None,response_source='identity_policy',
                 characters=None,prompt_additions=None):
        self.engine=engine;self.store=store;self.current=current;self.message=message;self.gate=gate
        self.max_new_tokens=max_new_tokens;self.policy_reply=policy_reply;self.response_source=response_source
        self.characters=characters;self.prompt_additions=prompt_additions or []
        self.events=Queue();self.stop=Event();self.finished=Event();self.state_lock=Lock()
        self.reason=None;self.committed=False
        self.timer=Timer(timeout,lambda:self.cancel('timeout'))
        self.timer.daemon=True
        self.thread=Thread(target=self.run,daemon=True)

    def cancel(self,reason='stopped'):
        with self.state_lock:
            if self.finished.is_set() or self.committed:
                return False
            self.reason=self.reason or reason
            self.stop.set()
            return True

    def start(self):
        self.timer.start();self.thread.start()

    def emit(self,text):
        if text and not self.stop.is_set():
            self.events.put(dict(event='delta',data={'text':text}))

    def run(self):
        try:
            if self.policy_reply is not None:
                answer=self.policy_reply
                metrics=dict(seconds=0.0,output_tokens=0,response_source=self.response_source)
                self.emit(answer)
            else:
                messages=conversation_messages(self.current,self.message,self.characters)
                if messages and self.prompt_additions:
                    messages[0]['content'] += '\n\n' + '\n\n'.join(self.prompt_additions)
                answer,metrics=self.engine.reply_stream(messages,self.current['model_mode'],self.stop,self.emit,
                                                       max_new_tokens=self.max_new_tokens)
            with self.state_lock:
                if self.stop.is_set():
                    raise GenerationStopped()
                if not answer.strip():
                    raise RuntimeError('空回答')
                # 与停止请求串行确定提交点；先完成提交的请求不会再被停止撤销。
                self.store.save_turn(self.current['id'],self.message,answer)
                self.committed=True
            self.events.put(dict(event='done',data=dict(answer=answer,metrics=metrics,
                                session_id=self.current['id'],model_mode=self.current['model_mode'])))
        except Exception as exc:
            if self.stop.is_set():
                code=self.reason or 'stopped'
                detail='生成超时，未保存本轮。' if code=='timeout' else '生成已停止，未保存本轮。'
            elif isinstance(exc,ValueError):
                code='invalid_context';detail=str(exc)
            else:
                logging.exception('流式生成失败')
                code='generation_failed';detail='生成失败，未保存本轮，请查看终端。'
            self.events.put(dict(event='error',data=dict(code=code,detail=detail)))
        finally:
            self.timer.cancel()
            self.finished.set()
            self.gate.release()


def encode_sse(event):
    return f"event: {event['event']}\ndata: {json.dumps(event['data'],ensure_ascii=False)}\n\n"
