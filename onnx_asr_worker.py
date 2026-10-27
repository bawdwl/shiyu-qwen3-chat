"""使用隔离的sherpa-onnx依赖执行中文Whisper识别。"""
import json
import site
import sys
import wave
from pathlib import Path
site.addsitedir(str(Path(__file__).resolve().parent/'outputs/onnx-tts-runtime'))
import numpy as np
import sherpa_onnx

def main():
    model,source,dest=map(Path,sys.argv[1:])
    recognizer=sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=str(model/'base-encoder.int8.onnx'),decoder=str(model/'base-decoder.int8.onnx'),
        tokens=str(model/'base-tokens.txt'),language='zh',task='transcribe',num_threads=2,provider='cpu')
    with wave.open(str(source),'rb') as audio:
        rate=audio.getframerate();channels=audio.getnchannels()
        samples=np.frombuffer(audio.readframes(audio.getnframes()),dtype='<i2').astype(np.float32)/32768
        samples=samples.reshape(-1,channels).mean(axis=1)
    stream=recognizer.create_stream();stream.accept_waveform(rate,samples)
    recognizer.decode_stream(stream)
    dest.write_text(json.dumps(dict(text=stream.result.text),ensure_ascii=False),encoding='utf-8')

if __name__=='__main__': main()
