"""单进程本地FastAPI后端；与Gradio独立启动，不同时占用两份GPU模型。"""
import argparse
import os
import sys
import logging
import sqlite3
import asyncio
import json
import time
import re
import hmac
import ipaddress
import secrets
from queue import Queue, Empty as QueueEmpty
from pathlib import Path
from contextlib import asynccontextmanager,contextmanager
from threading import Event, Lock, Thread
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse,StreamingResponse,Response,FileResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from storage import ChatStore
from stream_job import StreamJob,encode_sse
from characters import (CHARACTERS,DEFAULT_CHARACTER_IDS,ai_identity_reply,catalogue,character_profile_reply,conversation_messages,
                        is_explicit_ai_identity_question)
from character_cards import (build_generation_messages,card_to_runtime,normalize_card,
                             parse_generated_card,retrieve_worldbook,build_enrichment_messages,apply_enrichment)
from prompt_pipeline import assemble_prompt
from session_memory import refresh_memory

logger=logging.getLogger(__name__)


class SessionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model_mode: Literal['base','lora']='base'
    character_id: str=Field(default='interview',min_length=1,max_length=80)


class CharacterIdeaRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    inspiration: str=Field(strict=True,min_length=1,max_length=1800)

    @field_validator('inspiration')
    @classmethod
    def nonblank(cls,value):
        if not value.strip():
            raise ValueError('角色灵感不能为空白')
        return value.strip()


class CharacterCardRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    card: dict


class MemoryEditRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    notes: str=Field(default='',strict=True,max_length=1200)
    auto_enabled: bool=Field(default=True,strict=True)
    revision: int=Field(strict=True,ge=0)
    kept_items: list[int] | None=Field(default=None,max_length=12)

    @field_validator('kept_items')
    @classmethod
    def valid_memory_indices(cls,value):
        if value is not None and (any(type(i) is not int or i<0 for i in value) or len(set(value))!=len(value)):
            raise ValueError('记忆条目索引无效')
        return value


class VoiceProfileRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    display_name: str=Field(strict=True,min_length=1,max_length=80)
    voice_style: str=Field(strict=True,min_length=1,max_length=300)
    reference_text: str=Field(default='',strict=True,max_length=1000)
    language: Literal['zh']='zh'

    @field_validator('display_name','voice_style')
    @classmethod
    def nonblank(cls,value):
        if not value.strip():
            raise ValueError('音色名称和声音描述不能为空白')
        return value.strip()

    @field_validator('reference_text')
    @classmethod
    def clean_reference_text(cls,value):
        return value.strip()


class ChatRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    session_id: UUID
    message: str=Field(strict=True,min_length=1,max_length=4096)
    max_new_tokens: int=Field(default=256,strict=True,ge=1,le=1024)

    @field_validator('message')
    @classmethod
    def nonblank(cls,value):
        if not value.strip():
            raise ValueError('消息不能为空白')
        return value.strip()


class StreamRequest(ChatRequest):
    timeout_seconds: float=Field(default=60,ge=0.1,le=300,allow_inf_nan=False)


class SpeechRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    message_id: int=Field(strict=True,gt=0)


class CompletionMessage(BaseModel):
    model_config=ConfigDict(extra='ignore')
    role: Literal['system','user','assistant']
    content: str=Field(strict=True,max_length=32768)


class CompletionRequest(BaseModel):
    model_config=ConfigDict(extra='ignore')
    model: str=Field(default='qwen3-1.7b',min_length=1,max_length=100)
    messages: list[CompletionMessage]=Field(min_length=1,max_length=128)
    stream: bool=False
    max_tokens: int=Field(default=256,ge=1,le=1024)
    temperature: float | None=Field(default=None,ge=0,le=2,allow_inf_nan=False)
    top_p: float | None=Field(default=None,gt=0,le=1,allow_inf_nan=False)
    top_k: int | None=Field(default=None,ge=1,le=100)
    repetition_penalty: float | None=Field(default=None,ge=0.5,le=2,allow_inf_nan=False)


class LanPairRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    pair_code: str=Field(strict=True,min_length=6,max_length=6,pattern=r'^\d{6}$')
    device_name: str=Field(default='Android device',strict=True,min_length=1,max_length=64)


class LanPairResponse(BaseModel):
    device_id: str
    device_token: str
    expires_on_restart: bool=True


def create_app(db_path='outputs/chat.sqlite3',adapter_run=None,engine_factory=None,speech_factory=None,asr_factory=None,
               model_profile='qwen3-0.6b',same_chat_memory=False,lan_enabled=False,
               lan_pair_code=None,lan_pair_ttl=300):
    if engine_factory is None:
        def engine_factory():
            from inference import QwenChat
            return QwenChat(adapter_run=adapter_run,trim_history=True,model_profile=model_profile,
                            same_chat_memory=same_chat_memory,input_budget=4096)

    @asynccontextmanager
    async def lifespan(app):
        app.state.store=ChatStore(db_path)
        app.state.gate=Lock()
        app.state.engine=None
        app.state.job=None
        app.state.speech=None
        app.state.asr=None
        app.state.lan_enabled=bool(lan_enabled)
        app.state.lan_pair_code=lan_pair_code
        app.state.lan_pair_expires_at=0
        app.state.lan_pair_attempts=0
        app.state.lan_devices={}
        try:
            app.state.asr=asr_factory() if asr_factory else None
        except Exception:
            logger.exception('语音输入不可用，继续提供文字服务')
        try:
            app.state.speech=speech_factory() if speech_factory else None
        except Exception:
            logger.exception('语音配置不可用，继续提供文字服务')
        try:
            app.state.engine=engine_factory()
        except Exception:
            logger.exception('模型启动失败，保留历史读取功能')
        # Start the pairing window only after model and speech startup, just before
        # the HTTP server begins accepting requests.
        app.state.lan_pair_expires_at=time.monotonic()+max(1,int(lan_pair_ttl)) if lan_pair_code else 0
        try:
            yield
        finally:
            if app.state.job and not app.state.job.finished.is_set():
                app.state.job.cancel('shutdown')
                await asyncio.to_thread(app.state.job.thread.join)
        app.state.engine=None

    app=FastAPI(title='Qwen3 IM Chat API',version='0.1.0',lifespan=lifespan)

    def is_loopback_client(request):
        host=request.client.host if request.client else ''
        if host == 'testclient':
            return True
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return False

    @app.middleware('http')
    async def restrict_lan_surface(request,call_next):
        if not is_loopback_client(request):
            if not getattr(app.state,'lan_enabled',False):
                return JSONResponse(status_code=403,content={'detail':'LAN access is disabled'})
            if request.url.path not in ('/lan/pair','/lan/status'):
                return JSONResponse(status_code=403,content={'detail':'This endpoint is local-only'})
        return await call_next(request)

    def require_lan_device(request):
        if not app.state.lan_enabled:
            raise HTTPException(404,'LAN access is disabled')
        header=request.headers.get('authorization','')
        scheme,_,token=header.partition(' ')
        if scheme.lower()!='bearer' or not token:
            raise HTTPException(401,'Device token required',headers={'WWW-Authenticate':'Bearer'})
        for device_id,record in app.state.lan_devices.items():
            if hmac.compare_digest(record['token'],token):
                return device_id,record
        raise HTTPException(401,'Device token is invalid or revoked',headers={'WWW-Authenticate':'Bearer'})

    @app.post('/lan/pair',response_model=LanPairResponse)
    def lan_pair(body:LanPairRequest):
        if not app.state.lan_enabled:
            raise HTTPException(404,'LAN access is disabled')
        expected=app.state.lan_pair_code
        expired=time.monotonic()>app.state.lan_pair_expires_at
        if expected is None or expired or app.state.lan_pair_attempts>=5:
            app.state.lan_pair_code=None
            raise HTTPException(403,'Pairing code is unavailable or expired')
        if not hmac.compare_digest(expected,body.pair_code):
            app.state.lan_pair_attempts+=1
            if app.state.lan_pair_attempts>=5:
                app.state.lan_pair_code=None
            raise HTTPException(403,'Pairing code is invalid')
        app.state.lan_pair_code=None
        device_id=uuid4().hex
        token=secrets.token_urlsafe(32)
        app.state.lan_devices[device_id]={'name':body.device_name.strip() or 'Android device','token':token}
        return LanPairResponse(device_id=device_id,device_token=token)

    @app.get('/lan/status')
    def lan_status(request:Request):
        device_id,device=require_lan_device(request)
        engine=app.state.engine
        return dict(status='ready' if engine is not None else 'model_unavailable',
                    device_id=device_id,device_name=device['name'],
                    model_id=getattr(engine,'model_id',None) if engine else None,
                    speech_ready=app.state.speech is not None,
                    asr_ready=app.state.asr is not None)

    @app.get('/lan/devices')
    def lan_devices():
        if not app.state.lan_enabled:
            raise HTTPException(404,'LAN access is disabled')
        return {'devices':[{'device_id':key,'device_name':value['name']}
                           for key,value in app.state.lan_devices.items()]}

    @app.post('/lan/devices/{device_id}/revoke')
    def lan_revoke_device(device_id:str):
        if not app.state.lan_enabled:
            raise HTTPException(404,'LAN access is disabled')
        if app.state.lan_devices.pop(device_id,None) is None:
            raise HTTPException(404,'Device not found')
        return {'revoked':True,'device_id':device_id}
    web_dir=Path(__file__).resolve().parent.parent / 'web'
    if web_dir.is_dir():
        app.mount('/ui/assets',StaticFiles(directory=web_dir),name='app-assets')

        @app.get('/',include_in_schema=False)
        def home(): return RedirectResponse('/ui/')

        @app.get('/ui/',include_in_schema=False)
        def application(): return FileResponse(web_dir/'index.html',headers={'Cache-Control':'no-cache'})

    @app.get('/license',include_in_schema=False)
    def project_license():
        return FileResponse(Path(__file__).resolve().parent.parent / 'LICENSE',media_type='text/plain',
                            filename='LICENSE-AGPL-3.0.txt')

    @app.get('/characters')
    def characters():
        rows=catalogue(DEFAULT_CHARACTER_IDS)
        for card in app.state.store.list_character_cards():
            rows.append(dict(id=card['id'],name=card['name'],tag='自定义角色',color='peach',
                             description=card['description'],greeting=card['greeting'],starters=[],
                             profile=dict(line=card['tagline'],tags=[],scene='',about=card['description']),
                             custom=True))
        return dict(characters=rows)

    def runtime_characters():
        rows=dict(CHARACTERS)
        rows.update({card['id']:card_to_runtime(card) for card in app.state.store.list_character_cards()})
        return rows

    def prepare_chat(current,question,registry,stop=None):
        memory=app.state.store.get_memory(current['id'])
        warning=None
        try:
            draft,_=assemble_prompt(current,question,registry,memory)
            tokenizer=getattr(app.state.engine,'tokenizer',None)
            budget=getattr(app.state.engine,'input_budget',4096)
            should_refresh=False
            if tokenizer:
                encoded=tokenizer.apply_chat_template(draft,tokenize=True,add_generation_prompt=True,
                                                      enable_thinking=False,return_dict=True,return_tensors='pt')
                should_refresh=encoded['input_ids'].shape[-1] >= int(budget*0.75)
            if should_refresh:
                memory=refresh_memory(app.state.store,current,app.state.engine,stop)
        except ValueError as exc:
            warning=str(exc)
        except Exception:
            if stop is not None and stop.is_set():
                raise
            logger.exception('会话摘要失败，继续使用原始历史')
            warning='摘要暂未更新，继续使用原始历史。'
        messages,trace=assemble_prompt(current,question,registry,memory)
        trace['memory_warning']=warning
        return messages,trace

    @app.post('/characters/generate')
    def generate_character(body:CharacterIdeaRequest):
        engine=require_engine()
        if not app.state.gate.acquire(blocking=False):
            raise HTTPException(409,'模型正在生成回答，请稍后再创建角色。')
        try:
            generated,_=engine.reply(build_generation_messages(body.inspiration),model_mode='base',max_new_tokens=900)
            card=parse_generated_card(generated,body.inspiration)
            try:
                enrichment,_=engine.reply(build_enrichment_messages(card),model_mode='base',max_new_tokens=1000)
                card=apply_enrichment(card,enrichment)
            except (ValueError,TypeError) as exc:
                card['draft_notice']+=' 人物补充未通过格式检查，已保留第一步草稿，请手动补充。'
                logger.warning('角色补充失败：%s',exc)
            return {'card':card,'notice':card['draft_notice']}
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        except Exception as exc:
            logger.exception('本地角色卡生成失败')
            raise HTTPException(500,'本地模型生成角色卡失败，请查看终端日志。') from exc
        finally:
            app.state.gate.release()

    @app.post('/characters',status_code=201)
    def save_character(body:CharacterCardRequest):
        try:
            card=normalize_card(body.card)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        if card['id'] in {'assistant','companion','study','interview'}:
            raise HTTPException(409,'不能覆盖内置角色。')
        app.state.store.save_character_card(card)
        return {'card':card,'saved':True}

    @app.get('/characters/{character_id}')
    def get_character_card(character_id:str):
        try:
            return {'card':app.state.store.get_character_card(character_id)}
        except KeyError:
            raise HTTPException(404,'自定义角色不存在。') from None

    @app.delete('/characters/{character_id}')
    def delete_character(character_id:str):
        if character_id in {'assistant','companion','study','interview'}:
            raise HTTPException(409,'不能删除内置角色。')
        try:
            app.state.store.delete_character_card(character_id)
        except KeyError:
            raise HTTPException(404,'自定义角色不存在。') from None
        return {'deleted':True,'id':character_id}

    @app.get('/characters/{character_id}/voice-profile')
    def get_voice_profile(character_id:str):
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在。')
        profile=app.state.store.get_voice_profile(character_id)
        if profile is None:
            profile=dict(character_id=character_id,display_name=runtime_characters()[character_id]['name']+' · 音色',
                         voice_style='',reference_text='',reference_audio_path=None,language='zh')
        return {key:value for key,value in profile.items() if key!='reference_audio_path'} | {
            'has_reference_audio':bool(profile.get('reference_audio_path') and Path(profile['reference_audio_path']).is_file()),
            'gpt_sovits_ready':bool(profile.get('reference_audio_path') and Path(profile['reference_audio_path']).is_file()
                                    and profile.get('reference_text'))}

    @app.put('/characters/{character_id}/voice-profile')
    def save_voice_profile(character_id:str,body:VoiceProfileRequest):
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在。')
        profile=app.state.store.save_voice_profile(character_id,body.display_name,body.voice_style,
                                                   body.reference_text,body.language)
        return {key:value for key,value in profile.items() if key!='reference_audio_path'} | {
            'has_reference_audio':bool(profile.get('reference_audio_path') and Path(profile['reference_audio_path']).is_file()),
            'gpt_sovits_ready':bool(profile.get('reference_audio_path') and Path(profile['reference_audio_path']).is_file()
                                    and profile.get('reference_text'))}

    @app.put('/characters/{character_id}/voice-profile/reference-audio')
    async def upload_voice_reference(character_id:str,request:Request):
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在。')
        profile=app.state.store.get_voice_profile(character_id)
        if profile is None:
            raise HTTPException(409,'请先保存音色名称、声音描述和参考音频逐字稿。')
        size=int(request.headers.get('content-length','0') or 0)
        if size>20*1024*1024:
            raise HTTPException(413,'参考音频不能超过20 MiB。')
        chunks=[];total=0
        async for chunk in request.stream():
            total+=len(chunk)
            if total>20*1024*1024:
                raise HTTPException(413,'参考音频不能超过20 MiB。')
            chunks.append(chunk)
        content=b''.join(chunks)
        try:
            from speech import validate_wav
            validate_wav(content)
        except ValueError as exc:
            raise HTTPException(422,'请上传有效的 PCM WAV 参考音频。') from exc
        if not profile.get('reference_text'):
            raise HTTPException(422,'请先填写与参考音频完全对应的逐字稿。')
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',character_id):
            raise HTTPException(422,'角色编号格式不支持音色文件保存。')
        root=Path(db_path).resolve().parent/'voice-profiles'
        folder=root/character_id
        folder.mkdir(parents=True,exist_ok=True)
        target=(folder/'reference.wav').resolve()
        target.relative_to(root.resolve())
        temporary=folder/'reference.uploading'
        temporary.write_bytes(content)
        temporary.replace(target)
        app.state.store.set_voice_reference_audio(character_id,target)
        return {'uploaded':True,'has_reference_audio':True,'gpt_sovits_ready':True}

    @app.exception_handler(sqlite3.Error)
    async def database_error(request,exc):
        logger.exception('数据库操作失败',exc_info=exc)
        return JSONResponse(status_code=503,content={'detail':'会话存储暂不可用，请重试并查看终端日志。'})

    def require_engine():
        if app.state.engine is None:
            raise HTTPException(503,'模型未就绪，请查看启动终端。')
        return app.state.engine

    def session(sid):
        try:
            return app.state.store.get(str(sid))
        except KeyError:
            raise HTTPException(404,'会话不存在') from None

    def check_session_model(current,engine):
        if current['model_id']!=getattr(engine,'model_id','Qwen/Qwen3-0.6B'):
            raise HTTPException(422,'这段历史属于另一模型，已保留供查看。请点击“新对话”使用当前模型。')

    def check_session_character(current):
        if current['character_id'] not in runtime_characters():
            raise HTTPException(410,'这段历史绑定的角色卡已删除；记录仍保留，但无法继续生成。')

    @contextmanager
    def exclusive():
        # 包括读取历史、生成、写回；防止同时清空或两个请求覆盖同一历史。
        if not app.state.gate.acquire(blocking=False):
            raise HTTPException(409,'当前有请求正在生成，请稍后重试。')
        try:
            yield
        finally:
            app.state.gate.release()

    @app.get('/health')
    def health():
        ready=app.state.engine is not None
        return JSONResponse(status_code=200 if ready else 503,content=dict(
            status='ready' if ready else 'model_unavailable',
            model_modes=['base','lora'] if ready and app.state.engine.adapter_available else ['base'] if ready else [],
            storage='sqlite',generation_busy=app.state.gate.locked(),
            model_id=getattr(app.state.engine,'model_id','Qwen/Qwen3-0.6B') if ready else None,
            same_chat_memory_enabled=bool(getattr(app.state.engine,'same_chat_memory',False)) if ready else None,
            speech_provider=app.state.speech.label if app.state.speech else None,
            asr_provider=app.state.asr.label if app.state.asr else None))

    def openai_model_mode(model_name,engine):
        normalized=model_name.strip().lower()
        if normalized.endswith(('-lora', ':lora')):
            if not engine.adapter_available:
                raise HTTPException(400,detail='当前服务没有加载 LoRA Adapter。')
            return 'lora'
        model_id=getattr(engine,'model_id','Qwen/Qwen3-0.6B')
        aliases={model_id.lower(),model_id.rsplit('/',1)[-1].lower(),
                 model_id.rsplit('/',1)[-1].replace('Qwen3-','qwen3-').lower(),
                 'qwen3-1.7b','qwen3-0.6b'}
        if normalized not in aliases:
            raise HTTPException(404,detail=f'未知模型：{model_name}')
        return 'base'

    def completion_id():
        return 'chatcmpl-'+uuid4().hex

    @app.get('/v1/models')
    def openai_models():
        engine=require_engine()
        model_id=getattr(engine,'model_id','Qwen/Qwen3-0.6B')
        ids=[model_id]
        if engine.adapter_available:
            ids.append(model_id+'-lora')
        now=int(time.time())
        return {'object':'list','data':[{'id':name,'object':'model','created':now,'owned_by':'shiyu-local'}
                                        for name in ids]}

    @app.post('/v1/chat/completions')
    async def openai_chat_completions(body:CompletionRequest,request:Request):
        engine=require_engine()
        model_mode=openai_model_mode(body.model,engine)
        messages=[message.model_dump() for message in body.messages]
        if not any(message['role']=='user' for message in messages):
            raise HTTPException(422,detail='messages 至少需要一条 user 消息。')
        overrides={key:getattr(body,key) for key in
                   ('temperature','top_p','top_k','repetition_penalty') if getattr(body,key) is not None}
        if not app.state.gate.acquire(blocking=False):
            raise HTTPException(409,detail='模型正在处理另一条消息，请稍后重试。')
        ident=completion_id();created=int(time.time())

        if not body.stream:
            try:
                answer,metrics=engine.reply(messages,model_mode=model_mode,max_new_tokens=body.max_tokens,
                                            generation_overrides=overrides)
                if not isinstance(answer,str) or not answer.strip():
                    raise RuntimeError('模型返回空回答')
                usage={'prompt_tokens':int(metrics.get('input_tokens',0)),
                       'completion_tokens':int(metrics.get('output_tokens',0))}
                usage['total_tokens']=usage['prompt_tokens']+usage['completion_tokens']
                return {'id':ident,'object':'chat.completion','created':created,'model':body.model,
                        'choices':[{'index':0,'message':{'role':'assistant','content':answer},
                                    'finish_reason':'stop'}], 'usage':usage}
            except ValueError as exc:
                raise HTTPException(422,detail=str(exc)) from exc
            except Exception as exc:
                logger.exception('OpenAI 兼容接口生成失败')
                raise HTTPException(500,detail='模型生成失败，请查看后端日志。') from exc
            finally:
                app.state.gate.release()

        chunks=Queue()
        stop_event=Event()
        def emit(text):
            if text and not stop_event.is_set():
                chunks.put(('delta',text))
        def generate():
            try:
                answer,metrics=engine.reply_stream(messages,model_mode,stop_event,emit,
                                                   max_new_tokens=body.max_tokens,
                                                   generation_overrides=overrides)
                if not answer.strip():
                    raise RuntimeError('模型返回空回答')
                chunks.put(('complete',metrics))
            except Exception as exc:
                if not stop_event.is_set():
                    logger.exception('OpenAI 兼容流式接口生成失败')
                    chunks.put(('error',str(exc)))
            finally:
                app.state.gate.release()

        Thread(target=generate,daemon=True).start()
        def sse(payload):
            return 'data: '+json.dumps(payload,ensure_ascii=False)+'\n\n'
        async def events():
            try:
                yield sse({'id':ident,'object':'chat.completion.chunk','created':created,'model':body.model,
                           'choices':[{'index':0,'delta':{'role':'assistant'},'finish_reason':None}]})
                while True:
                    if await request.is_disconnected():
                        stop_event.set();return
                    try:
                        kind,value=await asyncio.to_thread(chunks.get,True,0.25)
                    except QueueEmpty:
                        continue
                    if kind=='delta':
                        payload={'id':ident,'object':'chat.completion.chunk','created':created,'model':body.model,
                                 'choices':[{'index':0,'delta':{'content':value},'finish_reason':None}]}
                    elif kind=='complete':
                        payload={'id':ident,'object':'chat.completion.chunk','created':created,'model':body.model,
                                 'choices':[{'index':0,'delta':{},'finish_reason':'stop'}]}
                        yield sse(payload);yield 'data: [DONE]\n\n';return
                    elif kind=='error':
                        yield sse({'error':{'message':'模型生成失败，请查看后端日志。','type':'server_error'}})
                        yield 'data: [DONE]\n\n';return
                    else:
                        continue
                    yield sse(payload)
            finally:
                stop_event.set()
        return StreamingResponse(events(),media_type='text/event-stream',
                                 headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    @app.post('/transcribe')
    async def transcribe(request:Request):
        from recognition import MAX_AUDIO_BYTES,normalize_audio
        if app.state.asr is None: raise HTTPException(503,'语音输入尚未配置')
        content=bytearray()
        async for chunk in request.stream():
            if len(content)+len(chunk)>MAX_AUDIO_BYTES: raise HTTPException(413,'音频超过8MiB')
            content.extend(chunk)
        def run():
            # 锁由工作线程持有；客户端断开不会提前释放正在识别的服务。
            with exclusive():
                try:
                    raw=normalize_audio(bytes(content))
                    return app.state.asr.transcribe(raw)
                except ValueError as exc: raise HTTPException(422,str(exc)) from exc
                except HTTPException: raise
                except Exception as exc:
                    logger.exception('语音识别失败')
                    raise HTTPException(503,'识别未完成，请重试；聊天内容保留。') from exc
        return await asyncio.to_thread(run)

    @app.post('/sessions',status_code=201)
    def create_session(body:SessionRequest):
        engine=require_engine()
        if body.character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在，请刷新角色列表后重试。')
        if body.model_mode=='lora' and not engine.adapter_available:
            raise HTTPException(400,'服务未加载LoRA Adapter')
        return app.state.store.create(body.model_mode,body.character_id,getattr(engine,'model_id','Qwen/Qwen3-0.6B'))

    @app.get('/sessions')
    def list_sessions(limit:int=Query(50,ge=1,le=100),offset:int=Query(0,ge=0),
                      nonempty:bool=False,q:str=Query('',max_length=200)):
        rows=app.state.store.list(limit+1,offset,nonempty,q.strip())
        return {'sessions':rows[:limit],'has_more':len(rows)>limit}

    @app.get('/sessions/{session_id}')
    def get_session(session_id:UUID):
        return session(session_id)

    @app.delete('/sessions/{session_id}/messages')
    def clear_session(session_id:UUID):
        with exclusive():
            session(session_id)
            app.state.store.clear(str(session_id))
            return session(session_id)

    @app.delete('/sessions/{session_id}')
    def delete_session(session_id:UUID):
        with exclusive():
            session(session_id)
            app.state.store.delete_session(str(session_id))
            return {'deleted':True,'id':str(session_id)}

    @app.get('/sessions/{session_id}/memory')
    def get_session_memory(session_id:UUID):
        session(session_id)
        return app.state.store.get_memory(str(session_id))

    @app.put('/sessions/{session_id}/memory')
    def edit_session_memory(session_id:UUID,body:MemoryEditRequest):
        with exclusive():
            session(session_id)
            try:
                updates=dict(notes=body.notes,auto_enabled=body.auto_enabled)
                if body.kept_items is not None:
                    current=app.state.store.get_memory(str(session_id))
                    if current['revision']!=body.revision:
                        raise ValueError('记忆已更新，请重新打开后再保存。')
                    items=current.get('items',[])
                    if any(i>=len(items) for i in body.kept_items):
                        raise ValueError('记忆条目已变化，请重新打开。')
                    updates['items']=[items[i] for i in body.kept_items]
                    retained={item['message_id'] for item in updates['items']}
                    removed={item['message_id'] for item in items} - retained
                    excluded=set(current.get('excluded_message_ids',[])) | removed
                    updates['excluded_message_ids']=sorted(excluded)
                return app.state.store.save_memory(str(session_id),updates,body.revision)
            except ValueError as exc:
                raise HTTPException(409,str(exc)) from exc

    @app.post('/sessions/{session_id}/memory/refresh')
    def refresh_session_memory(session_id:UUID):
        engine=require_engine()
        with exclusive():
            current=session(session_id)
            try:
                return refresh_memory(app.state.store,current,engine,force=True)
            except ValueError as exc:
                raise HTTPException(422,str(exc)) from exc

    @app.post('/sessions/{session_id}/memory/restore')
    def restore_session_memory(session_id:UUID):
        with exclusive():
            session(session_id)
            try:
                return app.state.store.restore_memory(str(session_id))
            except ValueError as exc:
                raise HTTPException(422,str(exc)) from exc

    @app.get('/sessions/{session_id}/prompt')
    def inspect_prompt(session_id:UUID,message:str=Query(default='继续聊聊',max_length=4096)):
        current=session(session_id)
        memory=app.state.store.get_memory(str(session_id))
        messages,trace=assemble_prompt(current,message,runtime_characters(),memory)
        tokenizer=getattr(app.state.engine,'tokenizer',None)
        trace['input_budget']=getattr(app.state.engine,'input_budget',4096)
        trace['messages']=messages
        trace['input_tokens']=None
        if tokenizer:
            from context_window import prepare_context_details
            try:
                inputs,metrics,selected,_=prepare_context_details(tokenizer,messages,budget=trace['input_budget'],trim_history=True)
                trace.update(metrics)
                trace['messages']=selected
                trace['input_tokens']=int(inputs['input_ids'].shape[-1])
            except ValueError as exc:
                trace['warning']=str(exc)
        return trace

    @app.post('/chat')
    def chat(body:ChatRequest):
        engine=require_engine()
        with exclusive():
            current=session(body.session_id)
            check_session_model(current,engine)
            check_session_character(current)
            registry=runtime_characters()
            try:
                profile_reply=character_profile_reply(current['character_id'],body.message,registry)
                if is_explicit_ai_identity_question(body.message):
                    answer=ai_identity_reply(current['character_id'],registry)
                    metrics=dict(seconds=0.0,output_tokens=0,response_source='identity_policy')
                elif profile_reply is not None:
                    answer=profile_reply
                    metrics=dict(seconds=0.0,output_tokens=0,response_source='persona_policy')
                else:
                    messages,trace=prepare_chat(current,body.message,registry)
                    options={'generation_overrides':trace['generation_preset']} if trace['generation_preset'] else {}
                    answer,metrics=engine.reply(messages,model_mode=current['model_mode'],max_new_tokens=body.max_new_tokens,**options)
                    metrics['memory_warning']=trace['memory_warning']
                    if metrics.get('context_omitted_turns'):
                        metrics['memory_warning']='本轮仍有部分旧对话超过上下文预算，可在会话记忆中继续整理。'
                if not isinstance(answer,str) or not answer.strip():
                    raise RuntimeError('模型返回空回答')
            except ValueError as exc:
                raise HTTPException(422,str(exc)) from exc
            except Exception as exc:
                # A CPU/custom engine failure must not trigger an extra Torch import.
                torch_module=sys.modules.get('torch')
                cuda=getattr(torch_module,'cuda',None)
                oom_type=getattr(cuda,'OutOfMemoryError',None)
                if oom_type is not None and isinstance(exc,oom_type):
                    cuda.empty_cache()
                    raise HTTPException(503,'显存不足，请关闭其他GPU进程或清空会话。') from exc
                logger.exception('模型生成失败')
                raise HTTPException(500,'模型生成失败，历史未写入，请重试并查看终端。') from exc
            # 只有完整成功的一轮问答才写入，两个消息在同一事务提交。
            app.state.store.save_turn(str(body.session_id),body.message,answer)
            return dict(session_id=str(body.session_id),model_mode=current['model_mode'],
                        answer=answer,metrics=metrics,message_count=len(current['messages'])+2)

    @app.post('/sessions/{session_id}/stop')
    def stop_session(session_id:UUID):
        session(session_id)
        job=app.state.job
        stopped=bool(job and job.current['id']==str(session_id) and job.cancel())
        return {'stop_requested':stopped,'generation_busy':app.state.gate.locked()}

    @app.post('/sessions/{session_id}/speech')
    def read_message(session_id:UUID,body:SpeechRequest):
        if app.state.speech is None:
            raise HTTPException(503,'尚未配置语音服务，文字聊天仍可使用。')
        with exclusive():
            current=session(session_id)
            target=next((m for m in current['messages'] if m['id']==body.message_id and m['role']=='assistant'),None)
            if target is None:
                raise HTTPException(404,'该会话中没有这条已保存的助手回答')
            try:
                profile=app.state.store.get_voice_profile(current['character_id'])
                if getattr(app.state.speech,'supports_voice_profiles',False) and profile:
                    if (not profile.get('reference_audio_path') or not profile.get('reference_text')) and not getattr(app.state.speech,'fallback_missing_profile',False):
                        raise HTTPException(409,'请先为该角色配置 GPT-SoVITS 参考音频和逐字稿。')
                    content=app.state.speech.synthesize_for_profile(target['content'],profile)
                else:
                    content=app.state.speech.synthesize(target['content'])
                from speech import validate_wav
                validate_wav(content)
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(422,str(exc)) from exc
            except Exception as exc:
                logger.exception('语音合成失败；文字回答保留')
                raise HTTPException(503,'语音生成失败，文字回答已保留，请查看后端日志。') from exc
            return Response(content,media_type='audio/wav',headers={'Cache-Control':'no-store',
                'X-Message-ID':str(body.message_id)})

    @app.post('/chat/stream')
    async def stream_chat(body:StreamRequest,request:Request):
        engine=require_engine()
        if not app.state.gate.acquire(blocking=False):
            raise HTTPException(409,'正在生成，请稍后重试。')
        try:
            current=session(body.session_id)
            check_session_model(current,engine)
            check_session_character(current)
            explicit_identity=is_explicit_ai_identity_question(body.message)
            policy_source='identity_policy' if explicit_identity else 'persona_policy'
            registry=runtime_characters()
            policy_reply=(ai_identity_reply(current['character_id'],registry) if explicit_identity
                          else character_profile_reply(current['character_id'],body.message,registry))
            job=StreamJob(engine,app.state.store,current,body.message,app.state.gate,body.timeout_seconds,
                          body.max_new_tokens,policy_reply=policy_reply,response_source=policy_source,
                          characters=registry,prompt_builder=lambda stop:prepare_chat(current,body.message,registry,stop))
            app.state.job=job
            job.start()
        except Exception:
            app.state.gate.release()
            raise

        async def events():
            terminal=False
            try:
                yield encode_sse(dict(event='start',data=dict(session_id=current['id'],model_mode=current['model_mode'])))
                while True:
                    if await request.is_disconnected():
                        job.cancel('disconnected');break
                    try:
                        event=job.events.get_nowait()
                    except QueueEmpty:
                        await asyncio.sleep(0.03);continue
                    yield encode_sse(event)
                    if event['event'] in ('done','error'):
                        terminal=True;break
            finally:
                if not terminal:
                    job.cancel('disconnected')
        return StreamingResponse(events(),media_type='text/event-stream',
                                 headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})

    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--db',default='outputs/chat.sqlite3')
    parser.add_argument('--adapter-run',default=None)
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--lan',action='store_true',
                        help='显式启用受限局域网配对入口；服务会绑定 0.0.0.0，其余 API 仍只允许本机访问')
    parser.add_argument('--model-profile',choices=['qwen3-0.6b','qwen3-1.7b'],default='qwen3-0.6b')
    parser.add_argument('--same-chat-memory',action='store_true',
                        help='实验开关：从本会话被裁剪的旧助手回答中检索小满自述；新会话不共享记忆')
    parser.add_argument('--speech-provider',choices=['onnx','gpt-sovits','profiles','off'],default='onnx')
    parser.add_argument('--asr-provider',choices=['whisper','off'],default='whisper')
    parser.add_argument('--asr-model-dir',default='outputs/asr-models/whisper-base')
    parser.add_argument('--tts-model-dir',default='outputs/tts-models/melo-zh-en')
    parser.add_argument('--tts-url',default='http://127.0.0.1:9880')
    parser.add_argument('--tts-reference')
    parser.add_argument('--tts-reference-text',default='')
    args=parser.parse_args()
    import uvicorn
    lan_pair_code=f'{secrets.randbelow(1_000_000):06d}' if args.lan else None
    if lan_pair_code:
        print(f'拾语 LAN 配对码（5分钟内有效，一次使用）：{lan_pair_code}',flush=True)
    # 不启用reload或多个worker，避免重复加载基础模型。
    speech_factory=None
    if args.speech_provider=='onnx':
        from speech import OnnxSpeech
        speech_factory=lambda:OnnxSpeech(args.tts_model_dir)
    elif args.speech_provider=='gpt-sovits':
        from speech import GPTSoVITSSpeech
        speech_factory=lambda:GPTSoVITSSpeech(args.tts_url,args.tts_reference,args.tts_reference_text)
    elif args.speech_provider=='profiles':
        from speech import GPTSoVITSSpeech,OnnxSpeech,ProfiledSpeech
        speech_factory=lambda:ProfiledSpeech(OnnxSpeech(args.tts_model_dir),
                                              GPTSoVITSSpeech(args.tts_url,args.tts_reference,args.tts_reference_text))
    asr_factory=None
    if args.asr_provider=='whisper':
        from recognition import WhisperRecognition
        asr_factory=lambda:WhisperRecognition(args.asr_model_dir)
    uvicorn.run(create_app(args.db,args.adapter_run,speech_factory=speech_factory,asr_factory=asr_factory,
                           model_profile=args.model_profile,same_chat_memory=args.same_chat_memory,
                           lan_enabled=args.lan,lan_pair_code=lan_pair_code),
                host='0.0.0.0' if args.lan else '127.0.0.1',port=args.port,workers=1)
