"""真实本地WAV/WebM识别验收；合成音频不等于真人麦克风准确率测试。"""
import json
import site
import subprocess
import time
from pathlib import Path
from api_client import BackendClient

def main():
    client=BackendClient();before=client.request('GET','/sessions')
    folder=Path('outputs/asr-demo');folder.mkdir(parents=True,exist_ok=True)
    source=Path('outputs/speech-demo/chat-answer.wav')
    if not source.exists(): raise RuntimeError('请先运行smoke_speech.py生成本项目的合成语音样本')
    site.addsitedir(str(Path('outputs/media-runtime').resolve()))
    import imageio_ffmpeg
    webm=folder/'input.webm'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-y','-loglevel','error','-i',str(source),
                    '-c:a','libopus',str(webm)],check=True,timeout=30)
    results=[]
    for path in (source,webm):
        start=time.perf_counter();result=client.transcribe(path)
        assert result['text'].strip()
        results.append(dict(format=path.suffix,seconds=round(time.perf_counter()-start,3),**result))
    assert client.request('GET','/sessions')==before
    assert not client.request('GET','/health')['generation_busy']
    report=dict(fixture='本项目MeloTTS合成：你好，我可以帮助你学习。',
                limitation='仅验证端到端链路，未测真人口音、噪声与中文字错率。',results=results,history_unchanged=True)
    Path('docs/asr-smoke-result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True,indent=2))

if __name__=='__main__': main()
