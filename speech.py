"""可替换的本地朗读提供商；系统语音与GPT-SoVITS明确区分。"""
import io
import os
import re
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from urllib.parse import urlparse

import httpx


def speech_text(text):
    text = re.sub(r'```.*?```', ' 这里有一段代码，请查看文字内容。 ', text, flags=re.S)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'(?m)^\s{0,3}#{1,6}\s+', '', text)
    text = re.sub(r'[*`_]', '', text)
    text = ' '.join(text.split())
    if not text:
        raise ValueError('没有可朗读的内容')
    if len(text) > 2000:
        raise ValueError('回答超过2000字符，请先让模型简短总结后再朗读。')
    return text


def validate_wav(content):
    if not content or len(content) > 40 * 1024 * 1024:
        raise ValueError('语音服务返回空音频或音频过大')
    try:
        with wave.open(io.BytesIO(content), 'rb') as audio:
            if audio.getnframes() == 0 or audio.getnchannels() not in (1, 2):
                raise ValueError('语音服务返回无效音频')
            if not 8000 <= audio.getframerate() <= 192000:
                raise ValueError('语音采样率不受支持')
    except (wave.Error, EOFError) as exc:
        raise ValueError('语音服务没有返回可播放的PCM WAV') from exc
    return content


class OnnxSpeech:
    label = '本地中文语音（MeloTTS，CPU；非GPT-SoVITS）'

    def __init__(self, model_dir='outputs/tts-models/melo-zh-en'):
        self.model_dir=Path(model_dir).resolve()
        if not (self.model_dir/'model.int8.onnx').is_file():
            raise ValueError('中文语音模型未下载，请先运行fetch_tts_model.py')
        self.worker = Path(__file__).with_name('onnx_speech_worker.py')

    def synthesize(self, text):
        text = speech_text(text)
        with tempfile.TemporaryDirectory(prefix='qwen-speech-') as folder:
            root = Path(folder)
            source, output = root / 'text.txt', root / 'speech.wav'
            source.write_text(text, encoding='utf-8')
            # Text is a data file, never interpolated into executable code.
            subprocess.run([sys.executable,str(self.worker),'--text-file',str(source),'--output',str(output),
                            '--model-dir',str(self.model_dir)],
                           check=True, capture_output=True, timeout=120,
                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            return validate_wav(output.read_bytes())


class GPTSoVITSSpeech:
    label = 'GPT-SoVITS（独立本地服务）'
    supports_voice_profiles = True

    def __init__(self, url, reference_audio, reference_text):
        target = urlparse(url)
        if target.scheme != 'http' or target.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('当前只接入本机GPT-SoVITS服务')
        self.url = url.rstrip('/')
        self.reference_audio = reference_audio
        self.reference_text = reference_text

    def synthesize(self, text):
        if not self.reference_audio or not self.reference_text.strip():
            raise ValueError('请为角色配置参考音频及其逐字稿。')
        return self._synthesize(text,self.reference_audio,self.reference_text,'zh')

    def synthesize_for_profile(self,text,profile):
        audio=profile.get('reference_audio_path')
        prompt=profile.get('reference_text','').strip()
        language=profile.get('language','zh')
        if not audio or not Path(audio).is_file() or not prompt:
            raise ValueError('该角色的 GPT-SoVITS 音色档案尚未配置完整参考音频和逐字稿。')
        return self._synthesize(text,audio,prompt,language)

    def _synthesize(self,text,reference_audio,reference_text,language):
        body = dict(text=speech_text(text), text_lang=language,
                    ref_audio_path=reference_audio, prompt_text=reference_text,
                    prompt_lang=language, media_type='wav', streaming_mode=False)
        with httpx.Client(timeout=120) as client:
            response = client.post(self.url + '/tts', json=body)
            response.raise_for_status()
            return validate_wav(response.content)


class ProfiledSpeech:
    """按角色音色合成；没有 GPT-SoVITS 音色档案时沿用 MeloTTS。"""
    label = '按角色音色（GPT-SoVITS；未配置时 MeloTTS）'
    supports_voice_profiles = True
    fallback_missing_profile = True

    def __init__(self, fallback, profile_provider):
        self.fallback = fallback
        self.profile_provider = profile_provider

    def synthesize(self, text):
        return self.fallback.synthesize(text)

    def synthesize_for_profile(self, text, profile):
        audio = profile.get('reference_audio_path') if profile else None
        prompt = profile.get('reference_text', '').strip() if profile else ''
        if audio and Path(audio).is_file() and prompt:
            return self.profile_provider.synthesize_for_profile(text, profile)
        return self.fallback.synthesize(text)
