"""Download the local Chinese BGE embedding model used by structured memory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ID = 'BAAI/bge-small-zh-v1.5'
DEFAULT_REVISION = '7999e1d3359715c523056ef9478215996d62a620'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--output-dir',
        default='outputs/models/bge-small-zh-v1.5',
        help='local model directory',
    )
    parser.add_argument(
        '--revision',
        default=DEFAULT_REVISION,
        help='optional Hugging Face revision/commit',
    )
    args = parser.parse_args()
    destination = Path(args.output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=REPO_ID,
        repo_type='model',
        revision=args.revision,
        local_dir=destination,
        ignore_patterns=['pytorch_model.bin'],
    )
    files = []
    for path in sorted(destination.rglob('*')):
        if (
            not path.is_file()
            or path.name == 'model_source.json'
            or '.cache' in path.relative_to(destination).parts
        ):
            continue
        files.append({
            'path': path.relative_to(destination).as_posix(),
            'size': path.stat().st_size,
            'sha256': sha256(path),
        })
    manifest = {
        'repo_id': REPO_ID,
        'revision': args.revision,
        'license': 'MIT per model card; recheck the upstream repository before redistribution',
        'files': files,
    }
    (destination / 'model_source.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'Downloaded {REPO_ID} to {destination}')


if __name__ == '__main__':
    main()
