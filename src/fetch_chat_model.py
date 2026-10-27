"""下载固定版本1.7B候选；不覆盖0.6B权重或Adapter。"""
import json
import hashlib
from pathlib import Path
from huggingface_hub import snapshot_download,HfApi

MODEL_ID='Qwen/Qwen3-1.7B'
REVISION='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
MODEL_DIR=Path('outputs/models/qwen3-1.7b')

def main():
    snapshot_download(MODEL_ID,revision=REVISION,local_dir=MODEL_DIR,max_workers=3,
        allow_patterns=['*.json','*.safetensors','tokenizer*','vocab*','merges*','LICENSE','README.md'])
    records=[]
    metadata=HfApi().model_info(MODEL_ID,revision=REVISION,files_metadata=True)
    for entry in metadata.siblings:
        file=MODEL_DIR/entry.rfilename
        if not file.is_file(): continue
        digest=hashlib.sha256()
        with file.open('rb') as source:
            for chunk in iter(lambda:source.read(8*1024*1024),b''): digest.update(chunk)
        if entry.lfs and digest.hexdigest()!=entry.lfs.sha256: raise ValueError('Model hash mismatch')
        records.append(dict(path=entry.rfilename,bytes=file.stat().st_size,sha256=digest.hexdigest()))
    source_record=dict(model_id=MODEL_ID,revision=REVISION,license='Apache-2.0',files=records,
        note='Original upstream model; no LoRA loaded or newly trained.')
    (MODEL_DIR/'source.json').write_text(json.dumps(source_record,indent=2),encoding='utf-8')
    Path('docs/qwen3-1.7b-source.json').write_text(json.dumps(source_record,indent=2),encoding='utf-8')
    print('Downloaded fixed Qwen3-1.7B candidate',flush=True)

if __name__=='__main__': main()
