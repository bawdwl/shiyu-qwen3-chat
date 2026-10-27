"""独立CPU识别进程；只接受有界PCM16 WAV，不写聊天历史。"""
import io
import json
import subprocess
import sys
import tempfile
import wave
from array import array
from pathlib import Path

MAX_AUDIO_BYTES=8*1024*1024

def normalize_audio(content):
    """兼容浏览器WebM录音；只允许音频容器和本地文件协议。"""
    if not content or len(content)>MAX_AUDIO_BYTES: raise ValueError('音频为空或超过8MiB')
    if content[:4]==b'RIFF':
        validate_input(content)
        return content
    import site
    site.addsitedir(str(Path(__file__).resolve().parent/'outputs/media-runtime'))
    import imageio_ffmpeg
    with tempfile.TemporaryDirectory() as folder:
        source=Path(folder)/'input.audio';dest=Path(folder)/'decoded.wav';source.write_bytes(content)
        process=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-nostdin','-hide_banner','-loglevel','error',
            '-protocol_whitelist','file,pipe','-format_whitelist','matroska,webm,ogg,mp3,flac,mov',
            '-i',str(source),'-t','31','-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(dest)],
            capture_output=True,timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        if process.returncode or not dest.exists(): raise ValueError('音频无法解码，请上传PCM WAV或重新录音')
        result=dest.read_bytes();validate_input(result)
        return result

def validate_input(content):
    if not content or len(content)>MAX_AUDIO_BYTES: raise ValueError('音频为空或超过8MiB')
    try:
        with wave.open(io.BytesIO(content),'rb') as audio:
            channels=audio.getnchannels();rate=audio.getframerate();frames=audio.getnframes()
            if audio.getsampwidth()!=2 or channels not in (1,2) or not 8000<=rate<=48000:
                raise ValueError('请使用8–48kHz、16位PCM WAV音频')
            if not 0<frames/rate<=30: raise ValueError('请录制或上传30秒以内的音频')
            samples=audio.readframes(frames)
            if len(samples)!=frames*channels*2: raise ValueError('音频内容不完整')
            values=array('h',samples)
            if not values or max(abs(x) for x in values)<32: raise ValueError('音频接近静音，请检查麦克风')
            return frames/rate
    except (wave.Error,EOFError) as exc:
        raise ValueError('无法读取音频，请使用16位PCM WAV文件') from exc

class WhisperRecognition:
    label='本地语音输入（Whisper base，CPU）'
    def __init__(self,model_dir='outputs/asr-models/whisper-base'):
        self.model_dir=Path(model_dir).resolve()
        for name in ('base-encoder.int8.onnx','base-decoder.int8.onnx','base-tokens.txt'):
            if not (self.model_dir/name).is_file(): raise ValueError('请先运行fetch_asr_model.py下载识别模型')
    def transcribe(self,content):
        content=normalize_audio(content)
        duration=validate_input(content)
        with tempfile.TemporaryDirectory() as folder:
            source=Path(folder)/'input.wav';result=Path(folder)/'result.json';source.write_bytes(content)
            worker=Path(__file__).with_name('onnx_asr_worker.py')
            process=subprocess.run([sys.executable,str(worker),str(self.model_dir),str(source),str(result)],
                capture_output=True,timeout=120,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if process.returncode: raise RuntimeError('本地识别进程失败，请检查模型和运行环境')
            text=json.loads(result.read_text(encoding='utf-8'))['text'].strip()
            if not text: raise ValueError('没有识别出文字，请重新录音')
            return dict(text=text,audio_seconds=duration,provider=self.label)
