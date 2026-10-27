"""下载固定版本Whisper base的上游INT8 ONNX文件并核验。"""
import hashlib
import json
import urllib.request
from pathlib import Path
from fetch_oasst2_source import fetch

REPO='csukuangfj/sherpa-onnx-whisper-base'
REVISION='bb53ee204431c90d314c1cc08d28d23e5b7927cc'

def main():
    root=Path('outputs/asr-models/whisper-base');root.mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen(f'https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true',timeout=30) as response:
        entries=json.load(response)
    names={'base-encoder.int8.onnx','base-decoder.int8.onnx','base-tokens.txt'}
    records=[]
    for entry in entries:
        name=entry['path']
        if name not in names: continue
        target=root/name
        digest=fetch(f'https://huggingface.co/{REPO}/resolve/{REVISION}/{name}',target,entry.get('lfs',{}).get('oid'))
        content=target.read_bytes()
        if len(content)!=entry['size']: raise ValueError('Size mismatch')
        if not entry.get('lfs') and hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()!=entry['oid']:
            raise ValueError('Git blob mismatch')
        records.append(dict(path=name,bytes=len(content),sha256=digest));print('Verified '+name,flush=True)
    if len(records)!=3: raise ValueError('Missing model files')
    license_url='https://raw.githubusercontent.com/openai/whisper/v20250625/LICENSE'
    license_hash=fetch(license_url,root/'LICENSE')
    (root/'source-manifest.json').write_text(json.dumps(dict(repo=REPO,revision=REVISION,
        origin='openai/whisper',license='MIT',license_url=license_url,license_sha256=license_hash,
        note='Upstream converted and quantized model; not our own quantization.',files=records),indent=2),encoding='utf-8')

if __name__=='__main__': main()
