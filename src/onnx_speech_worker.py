"""独立CPU进程合成WAV；额外依赖只加入本进程路径。"""
import argparse
import array
import site
import sys
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text-file', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()
    dependencies = Path(__file__).resolve().parent.parent / 'outputs/onnx-tts-runtime'
    if dependencies.is_dir():
        site.addsitedir(str(dependencies))
    import sherpa_onnx
    root = Path(args.model_dir).resolve()
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(model=str(root/'model.int8.onnx'),
        lexicon=str(root/'lexicon.txt'), tokens=str(root/'tokens.txt'), dict_dir=str(root/'dict'))
    model = sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=2, provider='cpu')
    config = sherpa_onnx.OfflineTtsConfig(model=model,
        rule_fsts=','.join(str(root/name) for name in ('date.fst', 'number.fst')), max_num_sentences=1)
    if not config.validate():
        raise ValueError('Invalid TTS model configuration')
    engine = sherpa_onnx.OfflineTts(config)
    audio = engine.generate(Path(args.text_file).read_text(encoding='utf-8'), sid=0, speed=1.0)
    pcm = array.array('h', (int(max(-1, min(1, value))*32767) for value in audio.samples))
    if sys.byteorder != 'little':
        pcm.byteswap()
    with wave.open(args.output, 'wb') as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(audio.sample_rate)
        target.writeframes(pcm.tobytes())


if __name__ == '__main__':
    main()
