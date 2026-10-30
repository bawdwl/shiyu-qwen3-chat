"""后台生成任务：协作停止，任务结束后才释放服务门锁。"""
import json
import logging
from queue import Queue
from threading import Event, Lock, Thread, Timer
from characters import conversation_messages
from providers import ProviderConfigurationError, ProviderError


class GenerationStopped(RuntimeError):
    pass


class StreamJob:
    def __init__(self,engine,store,current,message,gate,timeout,max_new_tokens=256,policy_reply=None,response_source='identity_policy',
                 characters=None,prompt_additions=None,prompt_builder=None,provider_name=None,
                 request_id=None,record_usage=None,validate_messages=None,image_id=None,
                 buffer_until_validated=False,transform_answer=None,after_save=None):
        self.engine=engine;self.store=store;self.current=current;self.message=message;self.gate=gate
        self.max_new_tokens=max_new_tokens;self.policy_reply=policy_reply;self.response_source=response_source
        self.characters=characters;self.prompt_additions=prompt_additions or []
        self.prompt_builder=prompt_builder
        self.provider_name=provider_name
        self.request_id=request_id
        self.record_usage=record_usage
        self.validate_messages=validate_messages
        self.image_id=image_id
        self.buffer_until_validated=bool(buffer_until_validated)
        self.transform_answer=transform_answer
        self.after_save=after_save
        self.stream_buffer=[]
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
            if self.buffer_until_validated:
                self.stream_buffer.append(text)
            else:
                self.events.put(dict(event='delta',data={'text':text}))

    def run(self):
        try:
            if self.policy_reply is not None:
                answer=self.policy_reply
                metrics=dict(seconds=0.0,output_tokens=0,response_source=self.response_source)
                self.emit(answer)
            else:
                trace={}
                if self.prompt_builder:
                    messages,trace=self.prompt_builder(self.stop)
                else:
                    messages=conversation_messages(self.current,self.message,self.characters)
                if self.stop.is_set():
                    raise GenerationStopped()
                if messages and self.prompt_additions:
                    messages[0]['content'] += '\n\n' + '\n\n'.join(self.prompt_additions)
                if self.validate_messages:
                    self.validate_messages(messages)
                options={'generation_overrides':trace['generation_preset']} if trace.get('generation_preset') else {}
                answer,metrics=self.engine.reply_stream(messages,self.current['model_mode'],self.stop,self.emit,
                                                       max_new_tokens=self.max_new_tokens,**options)
                if self.transform_answer:
                    answer,guard_metrics=self.transform_answer(answer)
                    metrics.update(guard_metrics or {})
                if trace.get('memory_warning'):
                    metrics['memory_warning']=trace['memory_warning']
                if metrics.get('context_omitted_turns'):
                    metrics['memory_warning']='本轮仍有部分旧对话超过上下文预算，可在会话记忆中继续整理。'
            saved_answer=answer
            if self.buffer_until_validated and answer and not self.stop.is_set():
                self.events.put(dict(event='delta',data={'text':answer}))
            with self.state_lock:
                if self.stop.is_set():
                    raise GenerationStopped()
                if not saved_answer.strip():
                    raise RuntimeError('空回答')
                # 与停止请求串行确定提交点；先完成提交的请求不会再被停止撤销。
                saved=self.store.save_turn(
                    self.current['id'],self.message,saved_answer,
                    image_id=self.image_id,
                )
                self.committed=True
            if self.record_usage:
                self.record_usage(self.current['id'],self.provider_name,self.request_id,metrics)
            if self.after_save:
                try:
                    self.after_save(self.current,saved,self.message,saved_answer)
                except Exception:
                    logging.exception('聊天已保存，但自动记忆任务入队失败')
            self.events.put(dict(event='done',data=dict(answer=saved_answer,display=None,metrics=metrics,
                                session_id=self.current['id'],model_mode=self.current['model_mode'],
                                provider=self.provider_name)))
        except Exception as exc:
            if self.stop.is_set():
                code=self.reason or 'stopped'
                detail='生成超时，未保存本轮。' if code=='timeout' else '生成已停止，未保存本轮。'
            elif isinstance(exc,ValueError):
                code='invalid_context';detail=str(exc)
            elif isinstance(exc,ProviderConfigurationError):
                code='provider_config';detail=str(exc)
            elif isinstance(exc,ProviderError):
                code='provider_error';detail=str(exc)
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
