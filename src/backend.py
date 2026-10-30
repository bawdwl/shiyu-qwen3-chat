"""单进程本地FastAPI后端；与Gradio独立启动，不同时占用两份GPU模型。"""
import argparse
import base64
import hashlib
import io
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
from datetime import datetime, timezone, timedelta
from queue import Queue, Empty as QueueEmpty
from pathlib import Path
from contextlib import asynccontextmanager,contextmanager
from threading import Event, Lock, Thread
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse,StreamingResponse,Response,FileResponse,RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from storage import ChatStore
from stream_job import StreamJob,encode_sse
from secret_store import LocalSecretStore, SecretStoreError
from providers import (DeepSeekChatEngine, DeepSeekProvider, ProviderConfigurationError,
                       OpenCodeProvider, ProviderError, estimate_tokens)
from knowledge_guard import query_boundary_risk,replace_forbidden_mentions
from characters import (CHARACTERS,DEFAULT_CHARACTER_IDS,ai_identity_reply,catalogue,character_profile_reply,conversation_messages,
                        is_explicit_ai_identity_question)
from character_cards import (build_generation_messages,card_to_runtime,normalize_card,
                             parse_generated_card,retrieve_worldbook,build_enrichment_messages,apply_enrichment)
from prompt_pipeline import assemble_prompt
from roleplay_response import tts_text_from_content
from session_memory import refresh_memory,retrieve_covered_history
from memory_records import validate_candidate
from structured_memory import (
    CandidateStore,
    MemoryWorker,
    STRUCTURED_MEMORY_EXTRACTION_PROMPT,
    build_unified_memory_context,
    parse_extraction_candidates,
    parse_extraction_result,
)

logger=logging.getLogger(__name__)


class SessionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model_mode: Literal['base','lora']='base'
    character_id: str=Field(default='interview',min_length=1,max_length=80)
    provider: Literal['local','deepseek','opencode'] | None=None


class CharacterIdeaRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    inspiration: str=Field(strict=True,min_length=1,max_length=1800)
    provider: Literal['local','deepseek','opencode'] | None=None

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


class StructuredMemoryEvidenceRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    message_id: int=Field(strict=True,gt=0)
    role: Literal['user','assistant']
    quote: str=Field(strict=True,min_length=1,max_length=600)


class StructuredMemoryCandidateRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind: Literal['user_report','character_detail','shared_event','belief','plan']
    subject: Literal['user','character','shared'] | None=None
    fact_key: str | None=Field(default=None,max_length=120)
    value: str | None=Field(default=None,max_length=200)
    content: str=Field(strict=True,min_length=1,max_length=600)
    event_date: str | None=None
    operation: Literal['add','merge','supersede','contested','ignore']='add'
    target_item_id: int | None=Field(default=None,strict=True,gt=0)
    evidence: list[StructuredMemoryEvidenceRequest]=Field(min_length=1,max_length=12)


class StructuredMemoryReviewRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    action: Literal['activate','revise','delete']
    item_id: int=Field(strict=True,gt=0)
    old_item_id: int | None=Field(default=None,strict=True,gt=0)


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
    request_id: UUID | None=None
    image_id: UUID | None=None

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


class ApiKeyRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    api_key: SecretStr=Field(min_length=8,max_length=512)


class ProviderSettingsRequest(BaseModel):
    model_config=ConfigDict(extra='forbid')
    request_token_limit: int=Field(default=12000,strict=True,ge=256,le=1000000)
    daily_token_limit: int=Field(default=200000,strict=True,ge=1000,le=100000000)


def create_app(db_path='outputs/chat.sqlite3',adapter_run=None,engine_factory=None,speech_factory=None,asr_factory=None,
               model_profile='qwen3-0.6b',same_chat_memory=False,lan_enabled=False,
               lan_pair_code=None,lan_pair_ttl=300,default_provider='local',secret_path=None,
               deepseek_provider=None,opencode_provider=None,
               structured_memory=False,structured_memory_db_path=None,
               auto_memory_worker_enabled=False,memory_daily_token_limit=30000,
               memory_job_ready_seconds=8.0,memory_embedding_model_path=None):
    if engine_factory is None and default_provider=='local':
        def engine_factory():
            from inference import QwenChat
            return QwenChat(adapter_run=adapter_run,trim_history=True,model_profile=model_profile,
                            same_chat_memory=same_chat_memory,input_budget=4096)
    elif engine_factory is None:
        engine_factory=lambda:None

    @asynccontextmanager
    async def lifespan(app):
        app.state.store=ChatStore(db_path)
        app.state.secret_store=LocalSecretStore(
            secret_path or (Path(db_path).resolve().parent/'secrets.json')
        )
        app.state.gate=Lock()
        app.state.engine=None
        app.state.default_provider=default_provider
        app.state.structured_memory=(
            CandidateStore(
                structured_memory_db_path
                or Path(db_path).resolve().parent/'structured-memory-experiment.sqlite3',
                allowed_root=Path(db_path).resolve().parent,
                embedding_model_path=memory_embedding_model_path,
            )
            if structured_memory
            else None
        )
        app.state.deepseek_provider=deepseek_provider or DeepSeekProvider(
            lambda: app.state.secret_store.get('deepseek_api_key'),
            base_url=os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com'),
            model=os.environ.get('DEEPSEEK_MODEL','deepseek-flash'),
        )
        app.state.deepseek_engine=DeepSeekChatEngine(
            app.state.deepseek_provider,
            input_budget=int(os.environ.get('DEEPSEEK_INPUT_BUDGET','16384')),
        )
        app.state.opencode_provider=opencode_provider or OpenCodeProvider(
            lambda: app.state.secret_store.get('opencode_api_key'),
            base_url=os.environ.get('OPENCODE_BASE_URL','https://opencode.ai/zen/go/v1'),
            model=os.environ.get('OPENCODE_MODEL','deepseek-v4.1-flash'),
        )
        app.state.opencode_engine=DeepSeekChatEngine(
            app.state.opencode_provider,
            input_budget=int(os.environ.get('OPENCODE_INPUT_BUDGET','16384')),
        )
        app.state.memory_daily_token_limit=max(1000,int(memory_daily_token_limit))
        app.state.memory_job_ready_seconds=max(
            0.0, float(memory_job_ready_seconds)
        )
        app.state.memory_worker=None
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
        if app.state.structured_memory is not None and auto_memory_worker_enabled:
            app.state.memory_worker=MemoryWorker(
                app.state.structured_memory,
                process_auto_memory_job,
            )
            app.state.memory_worker.start()
        # Start the pairing window only after model and speech startup, just before
        # the HTTP server begins accepting requests.
        app.state.lan_pair_expires_at=time.monotonic()+max(1,int(lan_pair_ttl)) if lan_pair_code else 0
        try:
            yield
        finally:
            worker_stopped=True
            if app.state.memory_worker is not None:
                worker_stopped=await asyncio.to_thread(
                    app.state.memory_worker.stop
                )
            if app.state.job and not app.state.job.finished.is_set():
                app.state.job.cancel('shutdown')
                await asyncio.to_thread(app.state.job.thread.join)
            app.state.memory_worker=None
        app.state.engine=None
        app.state.deepseek_engine=None
        app.state.opencode_engine=None
        if app.state.structured_memory is not None:
            if worker_stopped:
                app.state.structured_memory.close()
            else:
                logger.warning(
                    '后台记忆请求仍未结束，跳过显式关闭记忆数据库以避免并发关闭'
                )
        app.state.structured_memory=None

    app=FastAPI(title='Qwen3 IM Chat API',version='0.9.3',lifespan=lifespan)

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
        react_dir=web_dir/'react-dist'
        react_index=react_dir/'index.html'
        if react_index.is_file():
            app.mount('/ui/assets',StaticFiles(directory=react_dir/'assets'),name='react-assets')
            if (web_dir/'portraits').is_dir():
                app.mount('/ui/media/portraits',StaticFiles(directory=web_dir/'portraits'),name='portrait-assets')
        else:
            app.mount('/ui/assets',StaticFiles(directory=web_dir),name='app-assets')

        @app.get('/',include_in_schema=False)
        def home(): return RedirectResponse('/ui/')

        @app.get('/ui/',include_in_schema=False)
        def application():
            target=react_index if react_index.is_file() else web_dir/'index.html'
            return FileResponse(target,headers={'Cache-Control':'no-cache'})

        @app.get('/ui/icon.svg',include_in_schema=False)
        def app_icon():
            target=react_dir/'icon.svg' if react_index.is_file() else web_dir/'icon.svg'
            return FileResponse(target,media_type='image/svg+xml')

    @app.get('/license',include_in_schema=False)
    def project_license():
        return FileResponse(Path(__file__).resolve().parent.parent / 'LICENSE',media_type='text/plain',
                            filename='LICENSE-AGPL-3.0.txt')

    @app.get('/characters')
    def characters():
        rows=catalogue(DEFAULT_CHARACTER_IDS)
        for card in app.state.store.list_character_cards():
            avatar=app.state.store.latest_character_asset(card['id'],'avatar')
            rows.append(dict(id=card['id'],name=card['name'],tag='自定义角色',color='peach',
                             description=card['description'],greeting=card['greeting'],starters=[],
                             profile=dict(line=card['tagline'],tags=[],scene='',about=card['description']),
                             custom=True,
                             avatar_url=(f"/characters/{card['id']}/avatar" if avatar else None)))
        return dict(characters=rows)

    def runtime_characters():
        rows=dict(CHARACTERS)
        rows.update({card['id']:card_to_runtime(card) for card in app.state.store.list_character_cards()})
        return rows

    @app.get('/settings/providers')
    def get_provider_settings():
        limits=token_limits()
        return {
            'default_provider':app.state.default_provider,
            'deepseek':{
                'model_id':app.state.deepseek_engine.model_id,
                'configured':deepseek_key_configured(),
                'supports_text':True,
                'supports_images':True,
                'usage':app.state.store.usage_for_day('deepseek'),
            },
            'opencode':{
                'model_id':app.state.opencode_engine.model_id,
                'configured':opencode_key_configured(),
                'supports_text':True,
                'supports_images':True,
                'usage':app.state.store.usage_for_day('opencode'),
            },
            'local':{
                'available':app.state.engine is not None,
                'model_id':getattr(app.state.engine,'model_id',None),
                'supports_text':True,
                'supports_images':False,
            },
            'token_limits':{
                'request_token_limit':limits['request'],
                'daily_token_limit':limits['daily'],
            },
        }

    @app.put('/settings/deepseek-key')
    def set_deepseek_key(body:ApiKeyRequest):
        key=body.api_key.get_secret_value().strip()
        if not key:
            raise HTTPException(422,'DeepSeek API Key 不能为空')
        try:
            app.state.secret_store.set('deepseek_api_key',key)
        except SecretStoreError as exc:
            raise HTTPException(500,'API Key 无法安全保存在本机') from exc
        if app.state.memory_worker is not None:
            app.state.memory_worker.wake()
        return {'saved':True,'configured':True,'api_key':None}

    @app.delete('/settings/deepseek-key')
    def delete_deepseek_key():
        try:
            deleted=app.state.secret_store.delete('deepseek_api_key')
        except SecretStoreError as exc:
            raise HTTPException(500,'API Key 无法从本机删除') from exc
        return {'deleted':deleted,'configured':False}

    @app.put('/settings/opencode-key')
    def set_opencode_key(body:ApiKeyRequest):
        key=body.api_key.get_secret_value().strip()
        if not key.startswith('oc_sk_'):
            raise HTTPException(422,'OpenCode Go API Key 格式不正确')
        try:
            app.state.secret_store.set('opencode_api_key',key)
        except SecretStoreError as exc:
            raise HTTPException(500,'API Key 无法安全保存在本机') from exc
        if app.state.memory_worker is not None:
            app.state.memory_worker.wake()
        return {'saved':True,'configured':True,'api_key':None}

    @app.delete('/settings/opencode-key')
    def delete_opencode_key():
        try:
            deleted=app.state.secret_store.delete('opencode_api_key')
        except SecretStoreError as exc:
            raise HTTPException(500,'API Key 无法从本机删除') from exc
        return {'deleted':deleted,'configured':False}

    @app.put('/settings/token-limits')
    def set_token_limits(body:ProviderSettingsRequest):
        app.state.store.set_setting('token_limits',{
            'request_token_limit':body.request_token_limit,
            'daily_token_limit':body.daily_token_limit,
        })
        return {'saved':True,**body.model_dump()}

    def provider_engine(provider):
        if provider=='deepseek':
            if app.state.deepseek_engine is None:
                raise HTTPException(503,'DeepSeek 服务尚未初始化。')
            return app.state.deepseek_engine
        if provider=='opencode':
            if app.state.opencode_engine is None:
                raise HTTPException(503,'OpenCode Go 服务尚未初始化。')
            return app.state.opencode_engine
        if provider=='local':
            if app.state.engine is None:
                raise HTTPException(503,'本地模型未就绪，请查看启动终端。')
            return app.state.engine
        raise HTTPException(422,'未知模型 Provider')

    def provider_name_for_session(current):
        return current.get('provider') or app.state.default_provider

    def deepseek_key_configured():
        try:
            return bool(app.state.secret_store.get('deepseek_api_key'))
        except SecretStoreError:
            return False

    def opencode_key_configured():
        try:
            return bool(app.state.secret_store.get('opencode_api_key'))
        except SecretStoreError:
            return False

    def provider_key_configured(provider):
        if provider=='deepseek':
            return deepseek_key_configured()
        if provider=='opencode':
            return opencode_key_configured()
        return False

    def preferred_cloud_provider():
        if app.state.default_provider in ('opencode','deepseek') and provider_key_configured(
            app.state.default_provider
        ):
            return app.state.default_provider
        if opencode_key_configured():
            return 'opencode'
        if deepseek_key_configured():
            return 'deepseek'
        return None

    def token_limits():
        stored=app.state.store.get_setting('token_limits',{})
        if not isinstance(stored,dict):
            stored={}
        return {
            'request':int(stored.get('request_token_limit',12000)),
            'daily':int(stored.get('daily_token_limit',200000)),
        }

    def check_token_limits(provider,messages,max_new_tokens):
        limits=token_limits()
        estimated=int((estimate_tokens(messages)+int(max_new_tokens))*1.15)
        if estimated>limits['request']:
            raise HTTPException(422,detail={
                'code':'request_token_limit',
                'message':'本轮预计超过单次 Token 上限，请缩短输入或提高设置中的上限。',
                'estimated_tokens':estimated,
                'limit':limits['request'],
            })
        usage=app.state.store.usage_for_day(provider)
        if usage['total_tokens']+estimated>limits['daily']:
            raise HTTPException(429,detail={
                'code':'daily_token_limit',
                'message':'今天的 Token 用量将超过每日上限。',
                'used_tokens':usage['total_tokens'],
                'estimated_tokens':estimated,
                'limit':limits['daily'],
            })
        return estimated

    def check_stream_token_limits(provider,messages,max_new_tokens):
        try:
            check_token_limits(provider,messages,max_new_tokens)
        except HTTPException as exc:
            detail=exc.detail if isinstance(exc.detail,dict) else {'message':str(exc.detail)}
            raise ValueError(detail.get('message','本轮超过 Token 上限。')) from exc

    def record_provider_usage(provider,session_id,request_id,metrics):
        if not request_id or provider not in ('deepseek','opencode'):
            return
        try:
            app.state.store.record_usage(
                str(request_id),provider,metrics.get('model','deepseek-flash'),
                session_id,int(metrics.get('input_tokens',0)),
                int(metrics.get('output_tokens',0)),
                int(metrics.get('cached_tokens',0)))
        except sqlite3.IntegrityError:
            # Request IDs make retries idempotent at the accounting boundary.
            pass

    def record_stream_usage(session_id,provider,request_id,metrics):
        record_provider_usage(provider,session_id,request_id,metrics)

    def provider_request_overrides(provider,session_id,base=None):
        overrides=dict(base or {})
        if provider=='opencode':
            overrides['_opencode_session_id']=str(session_id)
        return overrides

    def attach_image_to_messages(messages,current,image_id):
        if image_id is None:
            return None
        provider=provider_name_for_session(current)
        engine=provider_engine(provider)
        if not getattr(engine,'supports_vision',False):
            raise ValueError('当前模型不支持图片，请切换到支持视觉的云端模型')
        try:
            image=app.state.store.get_image(str(image_id),session_id=current['id'])
        except KeyError:
            raise ValueError('图片不存在或不属于当前角色对话') from None
        if image.get('message_id') is not None:
            raise ValueError('图片已经随另一条消息保存')
        path=Path(image['file_path'])
        if not path.is_file():
            raise ValueError('图片文件不存在，请重新上传')
        encoded=base64.b64encode(path.read_bytes()).decode('ascii')
        last=next((m for m in reversed(messages) if m.get('role')=='user'),None)
        if last is None:
            raise ValueError('图片消息缺少用户文本')
        text=last.get('content') if isinstance(last.get('content'),str) else ''
        last['content']=[
            {'type':'text','text':text},
            {'type':'image_url','image_url':{
                'url':f"data:{image['mime_type']};base64,{encoded}",
                'detail':'original',
            }},
        ]
        return image

    def role_card(current,registry):
        role=registry.get(current.get('character_id'),{})
        return role.get('custom_card')

    def current_internal_state(current):
        store=app.state.structured_memory
        if store is None:
            return None
        return store.internal_state.snapshot(
            current.get('character_id','assistant')
        )

    def apply_knowledge_guard(answer,current,registry):
        answer,violations=replace_forbidden_mentions(role_card(current,registry),answer)
        return answer,({'knowledge_guard':'fallback','forbidden_mentions':violations}
                        if violations else {})

    def apply_structured_memory(messages,current,question,trace,memory=None):
        store=app.state.structured_memory
        trace['structured_memory_ids']=[]
        trace['structured_memory_chars']=0
        if store is None or not messages or messages[0].get('role')!='system':
            return
        memory=memory or {}
        character_id=current.get('character_id','assistant')
        if '_unified_memory_context' not in memory:
            excerpts=[]
            seen=set()
            for item in memory.get('items',[]):
                key=(item.get('message_id'),item.get('role'),item.get('quote'))
                if key not in seen:
                    excerpts.append(item)
                    seen.add(key)
            covered=memory.get('covered_message_id',0)
            if covered:
                for item in retrieve_covered_history(
                    current.get('messages',[]),
                    covered,
                    question,
                    excluded_message_ids=memory.get('excluded_message_ids',[]),
                ):
                    key=(item.get('message_id'),item.get('role'),item.get('quote'))
                    if key not in seen:
                        excerpts.append(item)
                        seen.add(key)
            context,memory_trace=build_unified_memory_context(
                store,
                character_id,
                question,
                excerpts=excerpts,
                summary_notes=memory.get('notes',''),
                max_facts=8,
                max_excerpts=4,
                max_chars=1600,
            )
            memory['_unified_memory_context']=context
            memory['_unified_memory_trace']=memory_trace
            state_text,internal_state=store.internal_state.render_prompt(character_id)
            memory['_internal_state']=internal_state
            memory['_internal_state_text']=state_text
        context=memory.get('_unified_memory_context','')
        memory_trace=memory.get('_unified_memory_trace',{})
        state=memory.get('_internal_state') or store.internal_state.snapshot(character_id)
        trace['structured_memory_ids']=memory_trace.get('selected_ids',[])
        trace['session_memory_ids']=memory_trace.get('selected_excerpt_ids',[])
        trace['structured_memory_chars']=memory_trace.get('characters',0)
        trace['memory_retrieval_mode']=memory_trace.get('retrieval_mode','lexical')
        trace['memory_retrieval_scores']=memory_trace.get('scores',[])
        state_text=memory.get('_internal_state_text') or store.internal_state.render_prompt(
            character_id,
            now=float(
                state.get('mood',{}).get('mood_updated_at')
                or datetime.now().timestamp()
            ),
        )[0]
        trace['internal_state_description']=state_text
        trace['internal_state_snapshot']=state
        messages[0]['content']+=(
            '\n\n【角色内部状态】\n'+state_text
        )
        if context:
            messages[0]['content']+=(
                '\n\n【统一长期记忆】\n'+context+
                '\n用户在当前消息中明确更正的内容'
                '优先于任何长期记忆。若本轮消息与旧记忆冲突，以本轮消息为准。'
            )

    def unify_session_and_structured_memory(current,memory):
        store=app.state.structured_memory
        if store is None or not memory:
            return memory
        structured_ids=store.covered_evidence_message_ids(
            current.get('character_id','assistant')
        )
        result=dict(memory)
        result['items']=[
            item for item in memory.get('items',[])
            if int(item.get('message_id',0)) not in structured_ids
        ]
        result['excluded_message_ids']=sorted(
            set(int(value) for value in memory.get('excluded_message_ids',[]))
            | structured_ids
        )
        result['unified_memory_managed']=True
        return result

    def schedule_auto_memory(current,saved):
        store=app.state.structured_memory
        if store is None:
            return None
        message_id=saved.get('assistant_message_id') or saved.get('user_message_id')
        if not message_id:
            return None
        job_id=store.enqueue_auto_extraction(
            current.get('character_id','assistant'),
            int(message_id),
            ready_in=app.state.memory_job_ready_seconds,
        )
        worker=getattr(app.state,'memory_worker',None)
        if worker is not None:
            worker.wake()
        return job_id

    def ingest_extracted_candidates(character_id,candidates,sources,
                                    *,job_id=None,processed_through=0):
        validated_items=[]
        rejected=[]
        for candidate in candidates:
            try:
                validated=validate_candidate(candidate,sources)
                validated_items.append(validated)
            except (ValueError,TypeError,KeyError) as exc:
                rejected.append({'reason':str(exc),'candidate':candidate})
        # Extraction batches can contain an old and a corrected value. Apply
        # them chronologically even if the model returned the new value first.
        validated_items.sort(
            key=lambda item:max(
                evidence['message_id'] for evidence in item['evidence']
            )
        )
        applied=[]
        store=structured_memory_store()
        for validated in validated_items:
            source_max=max(
                evidence['message_id'] for evidence in validated['evidence']
            )
            if source_max <= processed_through:
                applied.append({
                    'action':'already_processed',
                    'item_id':None,
                    'reason':'候选只引用了已经整理过的消息',
                })
                continue
            try:
                outcome=store.auto_ingest(
                    character_id,validated,job_id=job_id
                )
                applied.append({
                    'action':outcome['action'],
                    'item_id':outcome['item']['id'] if outcome.get('item') else None,
                    'reason':outcome.get('reason'),
                })
            except (ValueError,TypeError,KeyError) as exc:
                rejected.append({'reason':str(exc),'candidate':validated})
        return applied,rejected

    def process_auto_memory_job(job):
        store=app.state.structured_memory
        if store is None:
            return 'structured memory is disabled'
        if not store.job_is_running(job['id']):
            return 'job was cancelled'
        character_id=job['character_id']
        if character_id not in runtime_characters():
            return 'character no longer exists'
        current=app.state.store.get_role_conversation(character_id)
        if current is None:
            return 'conversation no longer exists'
        all_sources=[
            item for item in current['messages']
            if int(item['id'])<=int(job['message_end'])
        ]
        if not all_sources:
            return 'no source messages'
        processed_through=store.last_processed_message_id(character_id)
        pending=[
            item for item in all_sources
            if int(item['id'])>processed_through
        ]
        if not pending:
            return 'no unprocessed messages'
        new_batch=pending[:24]
        last_assistant=max(
            (
                index for index,item in enumerate(new_batch)
                if item['role']=='assistant'
            ),
            default=len(new_batch)-1,
        )
        new_batch=new_batch[:last_assistant+1]
        batch_through=int(new_batch[-1]['id'])
        context=[
            item for item in all_sources
            if int(item['id'])<int(new_batch[0]['id'])
        ][-4:]
        sources=context+new_batch

        selected=provider_name_for_session(current)
        if selected=='local' or not provider_key_configured(selected):
            selected=preferred_cloud_provider()
        if selected is None:
            return {
                'defer': True,
                'retry_after': 60,
                'reason': 'no cloud provider key is configured',
            }
        engine=provider_engine(selected)
        batch_text='\n'.join(item['content'] for item in new_batch)
        existing_facts=[
            {
                'item_id':item['id'],
                'kind':item['kind'],
                'subject':item['subject'],
                'fact_key':item['fact_key'],
                'value':item['value'],
                'content':item['content'],
                'state':item['state'],
            }
            for item in store.extraction_facts(
                character_id,batch_text,limit=40
            )
        ]
        extraction_messages=[
            {'role':'system','content':STRUCTURED_MEMORY_EXTRACTION_PROMPT},
            {'role':'user','content':json.dumps(
                {
                    'messages':[
                        {'id':item['id'],'role':item['role'],'content':item['content']}
                        for item in sources
                    ],
                    'existing_facts':existing_facts,
                },ensure_ascii=False)},
        ]
        max_tokens=1600
        estimated=int((estimate_tokens(extraction_messages)+max_tokens)*1.15)
        usage=store.memory_usage_for_day(character_id)
        if usage['total_tokens']+estimated>app.state.memory_daily_token_limit:
            now=datetime.now().astimezone()
            tomorrow=(now+timedelta(days=1)).replace(
                hour=0,minute=0,second=0,microsecond=0
            )
            return {
                'defer': True,
                'retry_after':max(60,(tomorrow-now).total_seconds()),
                'reason':'daily background memory budget is exhausted',
            }
        limits=token_limits()
        if estimated>limits['request']:
            return 'skipped: automatic memory request exceeds the configured request limit'
        overrides={
            'temperature':0,
            'response_format':{'type':'json_object'},
            '_opencode_session_id':f'structured-memory-{current["id"]}',
        }
        try:
            raw,metrics=engine.reply(
                extraction_messages,max_new_tokens=max_tokens,
                generation_overrides=overrides,
            )
            extraction=parse_extraction_result(raw)
            candidates=extraction['candidates']
            state_update=extraction.get('state_update')
        except (ProviderConfigurationError,ProviderError) as exc:
            return {
                'defer': True,
                'retry_after':60,
                'reason':str(exc),
            }

        prompt_tokens=int(metrics.get('prompt_tokens',estimate_tokens(extraction_messages)))
        completion_tokens=int(metrics.get('output_tokens',0))
        if prompt_tokens+completion_tokens<=0:
            prompt_tokens=estimated
        store.record_memory_usage(
            character_id,selected,str(metrics.get('model',getattr(engine,'model_id',selected))),
            prompt_tokens,completion_tokens,job_id=job['id'],
        )
        if not store.job_is_running(job['id']):
            return 'job was cancelled before memory commit'

        applied,rejected=ingest_extracted_candidates(
            character_id,candidates,sources,
            job_id=job['id'],processed_through=processed_through,
        )
        try:
            store.warm_candidate_embeddings(character_id)
        except Exception:
            logger.exception('后台记忆向量索引失败，下一次检索将回退或重试')
        state_result=None
        state_error=None
        if state_update is not None:
            try:
                state_result=store.internal_state.apply_update(
                    character_id,
                    state_update,
                    new_batch,
                    job_id=job['id'],
                    message_end=batch_through,
                )
            except (TypeError,ValueError,KeyError) as exc:
                state_error=str(exc)
        store.advance_cursor(character_id,batch_through)
        has_more=int(job['message_end'])>batch_through
        return {
            'provider':selected,
            'source_messages':len(sources),
            'new_messages':len(new_batch),
            'processed_through':processed_through,
            'processed_to':batch_through,
            'applied':applied,
            'rejected':rejected,
            'state_update':state_result,
            **({'state_error':state_error} if state_error else {}),
            'usage':{'prompt_tokens':prompt_tokens,
                     'completion_tokens':completion_tokens},
            'requeue':has_more,
            **({'retry_after':0} if has_more else {}),
        }

    def prepare_chat(current,question,registry,stop=None,image_id=None):
        engine=provider_engine(provider_name_for_session(current))
        memory=unify_session_and_structured_memory(
            current,app.state.store.get_memory(current['id'])
        )
        warning=None
        try:
            draft,draft_trace=assemble_prompt(current,question,registry,memory)
            apply_structured_memory(
                draft,current,question,draft_trace,memory
            )
            attach_image_to_messages(draft,current,image_id)
            tokenizer=getattr(engine,'tokenizer',None)
            budget=getattr(engine,'input_budget',4096)
            should_refresh=False
            if tokenizer:
                encoded=tokenizer.apply_chat_template(draft,tokenize=True,add_generation_prompt=True,
                                                      enable_thinking=False,return_dict=True,return_tensors='pt')
                should_refresh=encoded['input_ids'].shape[-1] >= int(budget*0.75)
            else:
                should_refresh=estimate_tokens(draft)>=int(budget*0.75)
            if should_refresh:
                memory=refresh_memory(app.state.store,current,engine,stop)
                memory=unify_session_and_structured_memory(current,memory)
        except ValueError as exc:
            warning=str(exc)
        except Exception:
            if stop is not None and stop.is_set():
                raise
            logger.exception('会话摘要失败，继续使用原始历史')
            warning='摘要暂未更新，继续使用原始历史。'
        messages,trace=assemble_prompt(current,question,registry,memory)
        apply_structured_memory(messages,current,question,trace,memory)
        preset=provider_request_overrides(
            provider_name_for_session(current),current['id'],trace.get('generation_preset'))
        if preset:
            trace['generation_preset']=preset
        image=attach_image_to_messages(messages,current,image_id)
        trace['image_id']=image['id'] if image else None
        trace['memory_warning']=warning
        return messages,trace

    @app.post('/characters/generate')
    def generate_character(body:CharacterIdeaRequest):
        selected=body.provider or app.state.default_provider
        engine=provider_engine(selected)
        if not app.state.gate.acquire(blocking=False):
            raise HTTPException(409,'模型正在生成回答，请稍后再创建角色。')
        try:
            generation_messages=build_generation_messages(body.inspiration)
            check_token_limits(selected,generation_messages,900)
            session_id=f'character-generation-{uuid4()}'
            generated,generation_metrics=engine.reply(
                generation_messages,model_mode='base',max_new_tokens=900,
                generation_overrides=provider_request_overrides(selected,session_id))
            record_provider_usage(selected,None,uuid4(),generation_metrics)
            card=parse_generated_card(generated,body.inspiration)
            try:
                enrichment_messages=build_enrichment_messages(card)
                check_token_limits(selected,enrichment_messages,1000)
                enrichment,enrichment_metrics=engine.reply(
                    enrichment_messages,model_mode='base',max_new_tokens=1000,
                    generation_overrides=provider_request_overrides(selected,session_id))
                record_provider_usage(selected,None,uuid4(),enrichment_metrics)
                card=apply_enrichment(card,enrichment)
            except (ValueError,TypeError) as exc:
                card['draft_notice']+=' 人物补充未通过格式检查，已保留第一步草稿，请手动补充。'
                logger.warning('角色补充失败：%s',exc)
            return {'card':card,'notice':card['draft_notice']}
        except ProviderConfigurationError as exc:
            raise HTTPException(409,str(exc)) from exc
        except ProviderError as exc:
            raise HTTPException(502,str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        except Exception as exc:
            logger.exception('角色卡生成失败')
            raise HTTPException(500,'模型生成角色卡失败，请查看终端日志。') from exc
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

    @app.get('/characters/{character_id}/avatar',include_in_schema=False)
    def character_avatar(character_id:str):
        asset=app.state.store.latest_character_asset(character_id,'avatar')
        if asset is None:
            raise HTTPException(404,'角色头像尚未生成')
        path=Path(asset['file_path']).resolve()
        try:
            path.relative_to(character_asset_root.resolve())
        except ValueError:
            raise HTTPException(404,'角色头像路径无效') from None
        if not path.is_file():
            raise HTTPException(404,'角色头像文件不存在')
        return FileResponse(path,media_type=asset['mime_type'],
                            headers={'Cache-Control':'private, max-age=3600'})

    @app.get('/characters/{character_id}/assets')
    def character_assets(character_id:str):
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在')
        return {
            'assets':[
                {key:value for key,value in asset.items() if key!='file_path'}
                for asset in app.state.store.list_character_assets(character_id)
            ]
        }

    @app.delete('/characters/{character_id}')
    def delete_character(character_id:str):
        with exclusive():
            if character_id in {'assistant','companion','study','interview'}:
                raise HTTPException(409,'不能删除内置角色。')
            try:
                conversation=app.state.store.get_role_conversation(character_id)
                if conversation:
                    remove_session_images(conversation['id'],delete_records=False)
                app.state.store.delete_character_card(character_id)
                if app.state.structured_memory is not None:
                    app.state.structured_memory.clear_scope(character_id)
            except KeyError:
                raise HTTPException(404,'自定义角色不存在。') from None
            remove_character_asset_files(character_id)
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

    image_root=Path(db_path).resolve().parent/'chat-images'
    character_asset_root=Path(db_path).resolve().parent/'character-assets'

    def remove_character_asset_files(character_id):
        root=(character_asset_root/character_id).resolve()
        if not root.is_dir():
            return
        try:
            root.relative_to(character_asset_root.resolve())
        except ValueError:
            return
        for path in root.rglob('*'):
            if path.is_file():
                path.unlink()
        for path in sorted(root.rglob('*'),reverse=True):
            if path.is_dir():
                path.rmdir()
        root.rmdir()

    def remove_session_images(session_id,delete_records=True):
        images=app.state.store.list_images_for_session(session_id)
        for image in images:
            path=Path(image['file_path'])
            try:
                path.resolve().relative_to(image_root.resolve())
            except (ValueError,OSError):
                continue
            if path.is_file():
                path.unlink()
        if delete_records:
            with app.state.store.connect() as db:
                db.execute('DELETE FROM uploaded_images WHERE session_id=?',(session_id,))

    def normalize_uploaded_image(content):
        from PIL import Image,ImageOps,UnidentifiedImageError
        if not content:
            raise ValueError('图片文件为空')
        try:
            with Image.open(io.BytesIO(content)) as raw:
                raw.verify()
            with Image.open(io.BytesIO(content)) as raw:
                if raw.width*raw.height>40_000_000:
                    raise ValueError('图片像素尺寸过大')
                image=ImageOps.exif_transpose(raw)
                if getattr(image,'is_animated',False):
                    image.seek(0)
                image=image.convert('RGBA' if 'A' in image.getbands() else 'RGB')
                image.thumbnail((2048,2048),Image.Resampling.LANCZOS)
                output=io.BytesIO()
                image.save(output,format='WEBP',quality=88,method=6)
                return output.getvalue(),image.width,image.height
        except (UnidentifiedImageError,OSError) as exc:
            raise ValueError('仅支持有效的 JPEG、PNG、GIF 或 WebP 图片') from exc

    @app.post('/images',status_code=201)
    async def upload_image(request:Request,session_id:UUID):
        current=session(session_id)
        provider=provider_name_for_session(current)
        engine=provider_engine(provider)
        if not getattr(engine,'supports_vision',False):
            raise HTTPException(422,'当前模型 Provider 不支持图片；请切换到支持视觉的云端模型。')
        declared_size=int(request.headers.get('content-length','0') or 0)
        if declared_size>20*1024*1024:
            raise HTTPException(413,'图片不能超过20 MiB')
        content=bytearray()
        async for chunk in request.stream():
            if len(content)+len(chunk)>20*1024*1024:
                raise HTTPException(413,'图片不能超过20 MiB')
            content.extend(chunk)
        try:
            normalized,width,height=await asyncio.to_thread(
                normalize_uploaded_image,bytes(content))
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        image_id=str(uuid4())
        image_root.mkdir(parents=True,exist_ok=True)
        target=image_root/f'{image_id}.webp'
        temporary=image_root/f'{image_id}.uploading'
        await asyncio.to_thread(temporary.write_bytes,normalized)
        temporary.replace(target)
        record=app.state.store.save_image_metadata({
            'id':image_id,
            'session_id':str(session_id),
            'file_path':str(target),
            'mime_type':'image/webp',
            'byte_size':len(normalized),
            'sha256':hashlib.sha256(normalized).hexdigest(),
            'width':width,
            'height':height,
        })
        return {
            'id':record['id'],
            'mime_type':record['mime_type'],
            'byte_size':record['byte_size'],
            'width':record['width'],
            'height':record['height'],
            'url':f'/images/{record["id"]}?session_id={session_id}',
        }

    @app.get('/images/{image_id}',include_in_schema=False)
    def get_image(image_id:UUID,session_id:UUID):
        try:
            image=app.state.store.get_image(str(image_id),session_id=str(session_id))
        except KeyError:
            raise HTTPException(404,'图片不存在') from None
        path=Path(image['file_path']).resolve()
        try:
            path.relative_to(image_root.resolve())
        except ValueError:
            raise HTTPException(404,'图片路径无效') from None
        if not path.is_file():
            raise HTTPException(404,'图片文件不存在')
        return FileResponse(path,media_type=image['mime_type'],
                            headers={'Cache-Control':'private, max-age=3600'})

    @app.delete('/images/{image_id}')
    def delete_image(image_id:UUID,session_id:UUID):
        try:
            image=app.state.store.get_image(str(image_id),session_id=str(session_id))
        except KeyError:
            raise HTTPException(404,'图片不存在') from None
        if image.get('message_id') is not None:
            raise HTTPException(409,'图片已经随消息保存，请重置角色对话后删除。')
        with app.state.store.connect() as db:
            db.execute('DELETE FROM uploaded_images WHERE id=?',(str(image_id),))
        path=Path(image['file_path'])
        if path.is_file():
            path.unlink()
        return {'deleted':True,'id':str(image_id)}

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
            raise HTTPException(422,'这段历史属于另一模型，已保留供查看。请重置该角色对话或切换回原模型。')

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
        default_provider=app.state.default_provider
        deepseek_ready=deepseek_key_configured()
        opencode_ready=opencode_key_configured()
        local_ready=app.state.engine is not None
        ready=(local_ready if default_provider=='local'
               else provider_key_configured(default_provider))
        active_engine={
            'deepseek':app.state.deepseek_engine,
            'opencode':app.state.opencode_engine,
            'local':app.state.engine,
        }.get(default_provider)
        deepseek_usage=app.state.store.usage_for_day('deepseek')
        opencode_usage=app.state.store.usage_for_day('opencode')
        limits=token_limits()
        return JSONResponse(status_code=200 if ready else 503,content=dict(
            status='ready' if ready else 'model_unavailable',
            app_version='0.9.3',
            default_provider=default_provider,
            providers={
                'deepseek':{'available':True,'configured':deepseek_ready,
                            'model_id':app.state.deepseek_engine.model_id,
                            'supports_images':True},
                'opencode':{'available':True,'configured':opencode_ready,
                            'model_id':app.state.opencode_engine.model_id,
                            'supports_images':True},
                'local':{'available':local_ready,
                         'model_id':getattr(app.state.engine,'model_id',None),
                         'supports_images':False}},
            model_modes=['base','lora'] if local_ready and getattr(app.state.engine,'adapter_available',False) else ['base'],
            storage='sqlite',generation_busy=app.state.gate.locked(),
            model_id=getattr(active_engine,'model_id',None) if ready else None,
            same_chat_memory_enabled=bool(getattr(app.state.engine,'same_chat_memory',False)) if local_ready else False,
            token_limits={'request':limits['request'],'daily':limits['daily']},
            deepseek_usage=deepseek_usage,
            opencode_usage=opencode_usage,
            structured_memory_enabled=app.state.structured_memory is not None,
            structured_memory_mode=(
                'automatic'
                if app.state.memory_worker is not None
                else ('manual' if app.state.structured_memory is not None else 'disabled')
            ),
            structured_memory_jobs=(app.state.structured_memory.job_counts()
                                    if app.state.structured_memory is not None else {}),
            memory_daily_token_limit=app.state.memory_daily_token_limit,
            memory_retrieval=(
                {
                    'fts5': bool(getattr(app.state.structured_memory,'fts_enabled',False)),
                    'embedding_model': (
                        getattr(
                            app.state.structured_memory.embedder,
                            'identity',
                            getattr(
                                app.state.structured_memory.embedder,
                                'model_name',
                                None,
                            ),
                        )
                        if app.state.structured_memory is not None
                        and app.state.structured_memory.embedder is not None
                        else None
                    ),
                }
                if app.state.structured_memory is not None else None
            ),
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
        selected=body.provider or app.state.default_provider
        engine=provider_engine(selected)
        if body.character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在，请刷新角色列表后重试。')
        if body.model_mode=='lora' and not getattr(engine,'adapter_available',False):
            raise HTTPException(400,'服务未加载LoRA Adapter')
        return app.state.store.create(
            body.model_mode,body.character_id,
            getattr(engine,'model_id','Qwen/Qwen3-0.6B'),selected)

    def ensure_character_opening(current,character_id,registry):
        greeting=str(
            registry.get(character_id,{}).get('greeting') or ''
        ).strip()
        if not greeting:
            return current
        if app.state.store.ensure_opening_message(current['id'],greeting):
            return app.state.store.get(current['id'])
        return current

    @app.put('/characters/{character_id}/conversation')
    def open_character_conversation(character_id:str,
                                    provider:Literal['local','deepseek','opencode'] | None=None):
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在，请刷新角色列表后重试。')
        selected=provider or app.state.default_provider
        engine=provider_engine(selected)
        with exclusive():
            current=app.state.store.get_or_create_role_conversation(
                character_id,model_id=getattr(engine,'model_id','Qwen/Qwen3-0.6B'),
                provider=selected)
            return ensure_character_opening(
                current,character_id,runtime_characters()
            )

    def structured_memory_store():
        if app.state.structured_memory is None:
            raise HTTPException(404,'结构化长期记忆实验尚未开启。')
        return app.state.structured_memory

    def structured_memory_sources(character_id):
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在。')
        current=app.state.store.get_role_conversation(character_id)
        if current is None:
            raise HTTPException(409,'该角色还没有聊天来源，不能创建可追溯记忆。')
        return current['messages']

    def structured_memory_conversation(character_id):
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在。')
        current=app.state.store.get_role_conversation(character_id)
        if current is None:
            raise HTTPException(409,'该角色还没有聊天来源，不能创建可追溯记忆。')
        return current

    @app.get('/characters/{character_id}/structured-memory')
    def list_structured_memories(character_id:str,include_inactive:bool=False):
        store=structured_memory_store()
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在。')
        records=store.list_items(character_id,include_inactive=include_inactive)
        return {'character_id':character_id,'records':records,
                'counts':{
                    'current':sum(item['state']=='active' for item in records),
                    'candidate':sum(item['state']=='candidate' for item in records),
                      'contested':sum(item['state']=='contested' for item in records),
                  },
                'jobs':store.job_counts(character_id),
                'usage':store.memory_usage_for_day(character_id),
                'processed_through':store.last_processed_message_id(character_id)}

    @app.get('/characters/{character_id}/structured-memory/status')
    def structured_memory_status(character_id:str):
        store=structured_memory_store()
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在。')
        usage=store.memory_usage_for_day(character_id)
        worker=getattr(app.state,'memory_worker',None)
        return {
            'character_id':character_id,
            'worker_running':bool(worker and worker.is_running()),
            'jobs':store.job_counts(character_id),
            'usage_today':usage,
            'daily_token_limit':app.state.memory_daily_token_limit,
            'automatic':bool(worker and worker.is_running()),
            'retrieval':{
                'fts5':bool(getattr(store,'fts_enabled',False)),
                'embedding_model':(
                    getattr(
                        store.embedder,
                        'identity',
                        getattr(store.embedder,'model_name',None),
                    )
                    if store.embedder is not None else None
                ),
            },
            'state_transition_count':int(store.db.execute(
                'SELECT COUNT(*) FROM state_transitions WHERE character_id=?',
                (character_id,),
            ).fetchone()[0]),
        }

    @app.post('/characters/{character_id}/structured-memory/candidates',status_code=201)
    def create_structured_memory_candidate(character_id:str,
                                           body:StructuredMemoryCandidateRequest):
        store=structured_memory_store()
        messages=structured_memory_sources(character_id)
        try:
            candidate=validate_candidate(body.model_dump(),messages)
            result=store.auto_ingest(character_id,candidate)
        except ValueError as exc:
            raise HTTPException(422,str(exc)) from exc
        return result

    @app.post('/characters/{character_id}/structured-memory/extract')
    def extract_structured_memories(character_id:str,
                                    message_limit:int=Query(24,ge=4,le=40)):
        store=structured_memory_store()
        current=structured_memory_conversation(character_id)
        sources=current['messages']
        selected=provider_name_for_session(current)
        if selected=='local':
            selected=('opencode' if opencode_key_configured() else
                      'deepseek' if deepseek_key_configured() else 'local')
        if not provider_key_configured(selected):
            raise HTTPException(503,'提取候选记忆需要先配置 OpenCode Go 或 DeepSeek API Key。')
        engine=provider_engine(selected)
        batch=sources[-message_limit:]
        existing_facts=[
            {
                'item_id':item['id'],
                'kind':item['kind'],
                'subject':item['subject'],
                'fact_key':item['fact_key'],
                'value':item['value'],
                'content':item['content'],
                'state':item['state'],
            }
            for item in store.list_items(character_id,include_inactive=True)
            if item['state']!='deleted'
        ]
        extraction_messages=[
            {'role':'system','content':STRUCTURED_MEMORY_EXTRACTION_PROMPT},
            {'role':'user','content':json.dumps(
                {
                    'messages':[
                        {'id':item['id'],'role':item['role'],'content':item['content']}
                        for item in batch
                    ],
                    'existing_facts':existing_facts,
                },ensure_ascii=False)},
        ]
        with exclusive():
            check_token_limits(selected,extraction_messages,1536)
            overrides={
                'temperature':0,
                '_opencode_session_id':f'structured-memory-{current["id"]}',
            }
            try:
                raw,metrics=engine.reply(
                    extraction_messages,max_new_tokens=1536,
                    generation_overrides=overrides,
                )
                candidates=parse_extraction_candidates(raw)
            except (ProviderConfigurationError,ProviderError) as exc:
                raise HTTPException(503,str(exc)) from exc
            except Exception as exc:
                logger.exception('结构化记忆候选提取失败')
                raise HTTPException(502,'候选记忆提取未返回有效结果。') from exc
            record_provider_usage(selected,current['id'],uuid4(),metrics)
        actions,rejected=ingest_extracted_candidates(
            character_id,candidates,sources
        )
        created=[]
        duplicates=[]
        superseded=[]
        contested=[]
        ignored=[]
        for outcome in actions:
            item_id=outcome.get('item_id')
            item=store.get(character_id,item_id) if item_id else None
            if outcome['action']=='added' and item is not None:
                created.append(item)
            elif outcome['action']=='merged' and item is not None:
                duplicates.append(item['id'])
            elif outcome['action']=='superseded' and item is not None:
                superseded.append(item)
            elif outcome['action']=='contested' and item is not None:
                contested.append(item)
            else:
                ignored.append(outcome)
        return {'created':created,'duplicates':duplicates,'superseded':superseded,
                'contested':contested,'ignored':ignored,'rejected':rejected,
                'source_messages':len(batch),'usage':metrics}

    @app.post('/characters/{character_id}/structured-memory/review')
    def review_structured_memory(character_id:str,body:StructuredMemoryReviewRequest):
        store=structured_memory_store()
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在。')
        try:
            if body.action=='activate':
                store.confirm_initial(character_id,body.item_id)
            elif body.action=='revise':
                if body.old_item_id is None:
                    raise ValueError('确认更正需要提供 old_item_id')
                store.confirm_revision(character_id,body.old_item_id,body.item_id)
            else:
                store.delete(character_id,body.item_id)
            item=store.get(character_id,body.item_id)
        except ValueError as exc:
            raise HTTPException(409,str(exc)) from exc
        return {'action':body.action,'item':item}

    @app.delete('/characters/{character_id}/structured-memory')
    def clear_structured_memories(character_id:str):
        store=structured_memory_store()
        if character_id not in runtime_characters():
            raise HTTPException(404,'角色不存在。')
        return {'deleted':store.clear_scope(character_id),'character_id':character_id}

    @app.delete('/characters/{character_id}/conversation/messages')
    def reset_character_conversation(character_id:str):
        with exclusive():
            if character_id not in runtime_characters():
                raise HTTPException(404,'角色不存在。')
            existing=app.state.store.get_role_conversation(character_id)
            if existing:
                remove_session_images(existing['id'],delete_records=False)
            try:
                result=app.state.store.clear_role_conversation(character_id)
            except KeyError:
                engine=provider_engine(app.state.default_provider)
                result=app.state.store.get_or_create_role_conversation(
                    character_id,model_id=getattr(engine,'model_id','Qwen/Qwen3-0.6B'),
                    provider=app.state.default_provider)
            if app.state.structured_memory is not None:
                app.state.structured_memory.clear_scope(character_id)
            remove_character_asset_files(character_id)
            return ensure_character_opening(
                result,character_id,runtime_characters()
            )

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
            remove_session_images(str(session_id),delete_records=False)
            app.state.store.clear(str(session_id))
            return session(session_id)

    @app.delete('/sessions/{session_id}')
    def delete_session(session_id:UUID):
        with exclusive():
            session(session_id)
            remove_session_images(str(session_id),delete_records=False)
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
            except HTTPException:
                raise
            except HTTPException:
                raise
            except ProviderConfigurationError as exc:
                raise HTTPException(409,str(exc)) from exc
            except ProviderError as exc:
                raise HTTPException(502,str(exc)) from exc
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
        memory=unify_session_and_structured_memory(
            current,app.state.store.get_memory(str(session_id))
        )
        messages,trace=assemble_prompt(current,message,runtime_characters(),memory)
        apply_structured_memory(messages,current,message,trace,memory)
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
        with exclusive():
            current=session(body.session_id)
            selected=provider_name_for_session(current)
            engine=provider_engine(selected)
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
                    messages,trace=prepare_chat(current,body.message,registry,image_id=body.image_id)
                    check_token_limits(selected,messages,body.max_new_tokens)
                    options={'generation_overrides':trace['generation_preset']} if trace['generation_preset'] else {}
                    answer,metrics=engine.reply(messages,model_mode=current['model_mode'],max_new_tokens=body.max_new_tokens,**options)
                    answer,guard_metrics=apply_knowledge_guard(answer,current,registry)
                    metrics.update(guard_metrics)
                    metrics['memory_warning']=trace['memory_warning']
                    if metrics.get('context_omitted_turns'):
                        metrics['memory_warning']='本轮仍有部分旧对话超过上下文预算，可在会话记忆中继续整理。'
                if not isinstance(answer,str) or not answer.strip():
                    raise RuntimeError('模型返回空回答')
            except HTTPException:
                raise
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
            saved=app.state.store.save_turn(
                str(body.session_id),body.message,answer,
                image_id=str(body.image_id) if body.image_id else None,
            )
            record_provider_usage(selected,current['id'],body.request_id,metrics)
            try:
                schedule_auto_memory(current,saved)
            except Exception:
                logger.exception('聊天已保存，但自动记忆任务入队失败')
            return dict(session_id=str(body.session_id),model_mode=current['model_mode'],
                        provider=selected,answer=answer,display=None,metrics=metrics,
                        message_count=len(current['messages'])+2)

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
                    content=app.state.speech.synthesize_for_profile(
                        tts_text_from_content(target['content']),profile
                    )
                else:
                    content=app.state.speech.synthesize(
                        tts_text_from_content(target['content'])
                    )
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
        if not app.state.gate.acquire(blocking=False):
            raise HTTPException(409,'正在生成，请稍后重试。')
        try:
            current=session(body.session_id)
            selected=provider_name_for_session(current)
            engine=provider_engine(selected)
            check_session_model(current,engine)
            check_session_character(current)
            explicit_identity=is_explicit_ai_identity_question(body.message)
            policy_source='identity_policy' if explicit_identity else 'persona_policy'
            registry=runtime_characters()
            policy_reply=(ai_identity_reply(current['character_id'],registry) if explicit_identity
                          else character_profile_reply(current['character_id'],body.message,registry))
            boundary_card=role_card(current,registry)
            strict_boundary=bool(
                policy_reply is None
                and (
                    query_boundary_risk(boundary_card,body.message)['risky']
                    or (
                        body.image_id is not None
                        and (boundary_card or {}).get('knowledge_boundary',{}).get('character_kind')=='historical'
                    )
                )
            )
            job=StreamJob(engine,app.state.store,current,body.message,app.state.gate,body.timeout_seconds,
                          body.max_new_tokens,policy_reply=policy_reply,response_source=policy_source,
                          characters=registry,prompt_builder=lambda stop:prepare_chat(
                              current,body.message,registry,stop,body.image_id),
                          provider_name=selected,request_id=body.request_id,
                          record_usage=record_stream_usage,
                          after_save=lambda current,saved,message,answer:schedule_auto_memory(
                              current,saved),
                          validate_messages=lambda messages: check_stream_token_limits(
                              selected,messages,body.max_new_tokens),
                          image_id=str(body.image_id) if body.image_id else None,
                          buffer_until_validated=strict_boundary,
                          transform_answer=lambda answer:apply_knowledge_guard(
                              answer,current,registry))
            app.state.job=job
            job.start()
        except Exception:
            app.state.gate.release()
            raise

        async def events():
            terminal=False
            try:
                yield encode_sse(dict(event='start',data=dict(
                    session_id=current['id'],model_mode=current['model_mode'],
                    provider=selected)))
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
    parser.add_argument('--provider',choices=['opencode','deepseek','local'],default='opencode')
    parser.add_argument('--same-chat-memory',action='store_true',
                        help='实验开关：从本会话被裁剪的旧助手回答中检索小满自述；新会话不共享记忆')
    structured_memory=parser.add_mutually_exclusive_group()
    structured_memory.add_argument(
        '--structured-memory',dest='structured_memory',action='store_true',
        default=True,
        help='启用后台自动整理的结构化长期记忆（默认开启）')
    structured_memory.add_argument(
        '--no-structured-memory',dest='structured_memory',action='store_false',
        help='关闭结构化长期记忆')
    parser.add_argument('--structured-memory-db',default=None,
                        help='结构化记忆独立SQLite路径；默认位于聊天数据库同目录')
    parser.add_argument('--memory-daily-token-limit',type=int,default=30000,
                        help='每个角色每天用于后台记忆分析的独立Token上限')
    parser.add_argument(
        '--memory-embedding-model',
        default='outputs/models/bge-small-zh-v1.5',
        help='本地中文BGE模型目录；不存在时退回FTS5和词法检索',
    )
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
                           lan_enabled=args.lan,lan_pair_code=lan_pair_code,
                           default_provider=args.provider,
                           structured_memory=args.structured_memory,
                           structured_memory_db_path=args.structured_memory_db,
                           auto_memory_worker_enabled=args.structured_memory,
                           memory_daily_token_limit=args.memory_daily_token_limit,
                           memory_embedding_model_path=args.memory_embedding_model,
                           ),
                host='0.0.0.0' if args.lan else '127.0.0.1',port=args.port,workers=1)
