"""单入口启动应用：自动选择装有项目依赖的 qwen3-im-chat Python。"""
import argparse
import os
import shutilimport subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.parse import urlparse
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))
from runtime_models import get_profile
def backend_python(root):    candidates=[]    configured=os.environ.get('QWEN3_IM_CHAT_PYTHON')    if configured:        candidates.append(Path(configured))    conda=shutil.which('conda')    if conda:        try:            base=subprocess.run([conda,'info','--base'],capture_output=True,text=True,                                timeout=10,check=True).stdout.strip()            if base:                candidates.append(Path(base)/'envs'/'qwen3-im-chat'/'python.exe')        except (OSError,subprocess.SubprocessError):            pass    candidates.extend([        Path('D:/Others/Study/Anaconda3/envs/qwen3-im-chat/python.exe'),        root/'.venv/Scripts/python.exe',        Path(sys.executable),    ])    for candidate in candidates:        if not candidate.is_file():            continue        try:            probe=subprocess.run([str(candidate),'-c','import fastapi,httpx,pydantic'],                                 capture_output=True,text=True,timeout=15)        except (OSError,subprocess.SubprocessError):            continue        if probe.returncode==0:            return str(candidate)    raise RuntimeError('未找到已安装项目依赖的 qwen3-im-chat Python。'                       '请激活正确环境或设置 QWEN3_IM_CHAT_PYTHON 环境变量。')
def running_health():
    try:
        response=httpx.get('http://127.0.0.1:8000/health',timeout=2,trust_env=False)
    except (httpx.ConnectError,httpx.ConnectTimeout):
        return None
    except httpx.HTTPError as error:
        raise RuntimeError('本机8000端口没有正常响应，请检查已有服务；不会重复加载模型。') from error
    try:
        health=response.json()
    except ValueError as error:
        raise RuntimeError('8000端口不是可识别的聊天服务，请检查端口占用。') from error
    if not isinstance(health,dict) or response.status_code not in (200,503) or health.get('storage')!='sqlite' or not isinstance(health.get('model_modes'),list):
        raise RuntimeError('8000端口不是可识别的聊天服务，请检查端口占用。')
    return health


def check_reusable(health,provider='deepseek',profile=None,same_chat_memory=False,                   speech_provider=None,structured_memory=False,                   memory_daily_token_limit=30000):    actual_provider=health.get('default_provider','local')    if actual_provider!=provider:        raise RuntimeError(f'已有服务使用 {actual_provider}，与本次请求的 {provider} 不一致。请先停止已有服务。')    if provider=='local':        if health.get('status')!='ready':            raise RuntimeError('已有本地模型服务未就绪，请查看其启动日志；不会重复启动。')        if profile and health.get('model_id')!=get_profile(profile)['model_id']:            raise RuntimeError('已有服务使用 '+str(health.get('model_id'))+'，与指定模型不一致。请先停止已有服务，再切换模型。')        if 'same_chat_memory_enabled' not in health:            raise RuntimeError('已有服务版本不支持本轮记忆对照，请先停止旧服务再启动。')        if bool(health['same_chat_memory_enabled'])!=bool(same_chat_memory):            expected='开启' if same_chat_memory else '关闭'            actual='开启' if health['same_chat_memory_enabled'] else '关闭'            raise RuntimeError(f'已有服务的同会话记忆为{actual}，与本次要求{expected}不一致。请先停止已有服务再启动。')
    expected_speech={'onnx':'本地中文语音（MeloTTS，CPU；非GPT-SoVITS）',
                     'gpt-sovits':'GPT-SoVITS（独立本地服务）',
                     'profiles':'按角色音色（GPT-SoVITS；未配置时 MeloTTS）',
                     'off':None}.get(speech_provider)
    if speech_provider and health.get('speech_provider')!=expected_speech:
        raise RuntimeError('已有服务语音提供方与本次要求不一致；请先停止已有服务，再使用所选语音模式启动。')


    if 'structured_memory_enabled' in health and bool(health['structured_memory_enabled'])!=bool(structured_memory):        expected='开启' if structured_memory else '关闭'        actual='开启' if health['structured_memory_enabled'] else '关闭'        raise RuntimeError(f'已有服务的结构化长期记忆为{actual}，与本次要求{expected}不一致。请先停止已有服务再启动。')    if structured_memory and 'structured_memory_enabled' not in health:        raise RuntimeError('已有服务版本不支持结构化长期记忆，请先停止旧服务再启动。')    if structured_memory:        if health.get('structured_memory_mode')!='automatic':            raise RuntimeError('已有服务仍使用人工确认版记忆流程，请先停止旧服务再启动。')        if int(health.get('memory_daily_token_limit',-1))!=int(memory_daily_token_limit):            raise RuntimeError('已有服务的后台记忆预算与本次启动参数不一致，请先停止旧服务。')        if not isinstance(health.get('memory_retrieval'),dict):            raise RuntimeError('已有服务尚未加载FTS5/BGE检索能力，请先停止旧服务再启动。')def start_gpt_sovits(root,url):
    target=urlparse(url)
    if target.scheme!='http' or target.hostname not in ('127.0.0.1','localhost','::1'):
        raise RuntimeError('GPT-SoVITS 只允许绑定本机 HTTP 地址。')
    try:
        response=httpx.get(url.rstrip('/')+'/docs',timeout=2,trust_env=False)
        if response.status_code==200:
            return None
        raise RuntimeError('GPT-SoVITS 端口已被占用，但不是可用的本机 API。')
    except (httpx.ConnectError,httpx.ConnectTimeout):
        pass
    env_python=root/'outputs/gpt-sovits-env/python.exe'
    api=root/'third_party/GPT-SoVITS/api_v2.py'
    config=root/'outputs/gpt-sovits-tts-cpu.yaml'
    if not all(path.is_file() for path in (env_python,api,config)):
        raise RuntimeError('GPT-SoVITS 本地环境或模型配置缺失，请先完成专用环境安装。')
    if target.port!=9881:
        raise RuntimeError('内置 GPT-SoVITS 启动配置目前使用本机端口 9881。')
    logs=root/'outputs';logs.mkdir(exist_ok=True)
    log=(logs/'gpt-sovits-api.log').open('a',encoding='utf-8')
    process=subprocess.Popen([str(env_python),'-u',str(api),'-c',str(config),
                              '-a','127.0.0.1','-p','9881'],cwd=root/'third_party/GPT-SoVITS',
                             stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    deadline=time.monotonic()+150
    try:
        while time.monotonic()<deadline:
            if process.poll() is not None:
                raise RuntimeError('GPT-SoVITS 启动失败，请查看 outputs/gpt-sovits-api.log。')
            try:
                response=httpx.get(url.rstrip('/')+'/docs',timeout=2,trust_env=False)
                if response.status_code==200:
                    log.close()
                    return process
            except httpx.HTTPError:
                pass
            time.sleep(.5)
        raise RuntimeError('GPT-SoVITS 启动超时，请查看 outputs/gpt-sovits-api.log。')
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        log.close()
        raise

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--no-open',action='store_true')
    parser.add_argument('--provider',choices=['opencode','deepseek','local'],default='opencode')    parser.add_argument('--model-profile',choices=['qwen3-0.6b','qwen3-1.7b'])
    parser.add_argument('--same-chat-memory',action='store_true',
                        help='实验性开启小满同一聊天的裁剪后记忆检索；默认关闭')
    structured_memory=parser.add_mutually_exclusive_group()    structured_memory.add_argument(        '--structured-memory',dest='structured_memory',action='store_true',        default=True,        help='启用后台自动整理的结构化长期记忆（默认开启）')    structured_memory.add_argument(        '--no-structured-memory',dest='structured_memory',action='store_false',        help='临时关闭结构化长期记忆，便于排错或做对照实验')    parser.add_argument('--memory-daily-token-limit',type=int,default=30000,                        help='每个角色每天用于后台记忆分析的独立Token上限')    parser.add_argument('--speech-provider',choices=['onnx','gpt-sovits','profiles','off'])
    parser.add_argument('--asr-provider',choices=['whisper','off'],default='whisper',
                        help='语音识别需另行准备模型；可用 off 关闭')
    parser.add_argument('--tts-url',default='http://127.0.0.1:9881')
    args=parser.parse_args();root=Path(__file__).resolve().parent;url='http://127.0.0.1:8000/ui/'
    existing=running_health()
    if existing is not None:
        check_reusable(existing,args.provider,args.model_profile,args.same_chat_memory,                       args.speech_provider,args.structured_memory,                       args.memory_daily_token_limit)
        print('本机服务正在运行：'+str(existing.get('model_id'))+' '+url,flush=True)
        if not args.no_open: webbrowser.open(url)
        return
    command=[backend_python(root),'-u',str(root/'src/backend.py')]
    command+=['--provider',args.provider]
    profile=None    if args.provider=='local':        profile=args.model_profile or ('qwen3-1.7b' if (root/'outputs/models/qwen3-1.7b/source.json').exists() else 'qwen3-0.6b')        command+=['--model-profile',profile]
    speech_provider=args.speech_provider or 'profiles'
    tts_process=None
    if speech_provider in ('gpt-sovits','profiles'):
        tts_process=start_gpt_sovits(root,args.tts_url)
    command+=['--speech-provider',speech_provider]
    command+=['--asr-provider',args.asr_provider]
    if speech_provider in ('gpt-sovits','profiles'):
        command+=['--tts-url',args.tts_url]
    if args.same_chat_memory: command.append('--same-chat-memory')    command.append('--structured-memory' if args.structured_memory                   else '--no-structured-memory')    command+=['--memory-daily-token-limit',str(args.memory_daily_token_limit)]
    if args.provider=='local' and profile=='qwen3-0.6b' and (root/'outputs/lora-data-v2/run-config.json').exists(): command+=['--adapter-run','outputs/lora-data-v2']
    logs=root/'outputs';logs.mkdir(exist_ok=True)
    with (logs/'app-service.log').open('a',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=root,stdout=log,stderr=log,
                                 env={**os.environ,'PYTHONIOENCODING':'utf-8'},
                                 creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            print('正在启动本机模型，详细日志：outputs/app-service.log',flush=True)
            deadline=time.monotonic()+180
            while True:
                if process.poll() is not None: raise RuntimeError('启动失败，请查看outputs/app-service.log')
                health=running_health()
                if health is not None:
                    check_reusable(health,args.provider,profile,args.same_chat_memory,                                   speech_provider,args.structured_memory,                                   args.memory_daily_token_limit)
                    break
                if time.monotonic()>deadline: raise RuntimeError('启动超时，请查看outputs/app-service.log')
                time.sleep(.5)
            label='DeepSeek deepseek-flash' if args.provider=='deepseek' else str(profile)            print(label+' · 打开 '+url+'；保持此窗口运行，按Ctrl+C停止。',flush=True)
            if not args.no_open: webbrowser.open(url)
            process.wait()
        finally:
            if process.poll() is None: process.terminate();process.wait(timeout=15)
            if tts_process is not None and tts_process.poll() is None:
                tts_process.terminate();tts_process.wait(timeout=15)

if __name__=='__main__': main()
