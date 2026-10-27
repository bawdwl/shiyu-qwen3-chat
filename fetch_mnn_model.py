"""Download the upstream 4-bit MNN export; this does not convert or train weights."""
import hashlib
import json
import time
from pathlib import Path

import httpx
from huggingface_hub import HfApi

MODEL_ID = 'taobao-mnn/Qwen3-1.7B-MNN'
REVISION = '1b12fe7da4acff8593fd20cc6007f0fa85c02682'
DEST = Path('outputs/models/qwen3-1.7b-mnn')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    metadata = HfApi().model_info(MODEL_ID, revision=REVISION, files_metadata=True)
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        for entry in metadata.siblings:
            name = entry.rfilename
            if name.startswith('.') or '/' in name:
                continue
            target = DEST / name
            expected = entry.lfs.sha256 if entry.lfs else None
            valid = target.exists() and target.stat().st_size == entry.size
            if valid and expected:
                valid = sha256(target) == expected
            if not valid:
                part = target.with_name(name + '.part')
                for attempt in range(3):
                    try:
                        offset = part.stat().st_size if part.exists() else 0
                        if offset == entry.size:
                            if not expected or sha256(part) == expected:
                                break
                            part.unlink()
                            offset = 0
                        headers = {'Range': f'bytes={offset}-'} if offset else {}
                        url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}?download=true&attempt={time.time_ns()}'
                        with client.stream('GET', url, headers=headers) as response:
                            response.raise_for_status()
                            if response.status_code == 206:
                                if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                                    raise ValueError('Unexpected range response')
                            else:
                                offset = 0
                            with part.open('ab' if offset else 'wb') as output:
                                for block in response.iter_bytes(1024 * 1024):
                                    output.write(block)
                        if part.stat().st_size != entry.size:
                            raise ValueError('Incomplete download: ' + name)
                        if expected and sha256(part) != expected:
                            part.unlink()
                            raise ValueError('Upstream hash mismatch: ' + name)
                        break
                    except (httpx.HTTPError, ValueError):
                        if attempt == 2:
                            raise
                part.replace(target)
            records.append(dict(path=name, bytes=target.stat().st_size, sha256=sha256(target)))
            print('Verified ' + name, flush=True)
    source = dict(model_id=MODEL_ID, revision=REVISION, license='Apache-2.0',
                  files=records, note='Upstream 4-bit export; not converted or trained by this project.')
    content = json.dumps(source, ensure_ascii=False, indent=2)
    (DEST / 'source.json').write_text(content, encoding='utf-8')
    Path('docs/mnn-model-source.json').write_text(content, encoding='utf-8')


if __name__ == '__main__':
    main()
