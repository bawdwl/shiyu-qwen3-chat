"""下载固定版本的MeloTTS ONNX模型，保留上游许可与文件指纹。"""
import hashlib
import json
import urllib.request
from pathlib import Path
from download_utils import fetch

REPO = 'csukuangfj/vits-melo-tts-zh_en'
REVISION = 'a0d5c6a264c0ef92d70d8661d8cc502d79627cd6'


def main():
    root = Path('outputs/tts-models/melo-zh-en')
    root.mkdir(parents=True, exist_ok=True)
    api = f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true'
    with urllib.request.urlopen(api, timeout=30) as response:
        entries = json.load(response)
    allowed = {'LICENSE', 'README.md', 'model.int8.onnx', 'lexicon.txt', 'tokens.txt', 'date.fst', 'number.fst'}
    records = []
    for entry in entries:
        name = entry['path']
        if entry['type'] != 'file' or (name not in allowed and not name.startswith('dict/')):
            continue
        target = root / name
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('Invalid upstream path')
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = entry.get('lfs', {}).get('oid')
        digest = fetch(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}', target, expected)
        content = target.read_bytes()
        if len(content) != entry['size']:
            raise ValueError('Unexpected size: ' + name)
        if not expected and hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() != entry['oid']:
            raise ValueError('Git blob mismatch: ' + name)
        records.append(dict(path=name, bytes=len(content), sha256=digest))
        print('Verified ' + name, flush=True)
    if not (root/'model.int8.onnx').is_file():
        raise ValueError('Missing model')
    (root/'source-manifest.json').write_text(json.dumps(dict(repo=REPO, revision=REVISION,
        model_origin='myshell-ai/MeloTTS-Chinese', license_file='LICENSE',
        note='Upstream prequantized ONNX model, not our own quantization and not GPT-SoVITS.',
        files=records), ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
