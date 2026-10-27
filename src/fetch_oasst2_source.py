"""下载固定版本公开数据到outputs；不安装依赖，不写入现有训练集。"""
import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = 'OpenAssistant/oasst2'
REVISION = '179dd21fc55192153d94adb0e0ce8f69e222bf75'
FILENAME = '2023-11-05_oasst2_ready.messages.jsonl.gz'
FILE_SHA256 = 'a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751'


def fetch(url, target, expected_sha256=None):
    if target.exists():
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError('Existing source file has unexpected SHA256: ' + str(target))
        return digest
    partial = target.with_suffix(target.suffix + '.part')
    digest = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=60) as response, partial.open('wb') as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ValueError('Download hash mismatch; partial file retained for diagnosis')
    partial.replace(target)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='outputs/dataset-sources/oasst2')
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
    print('Downloading pinned OpenAssistant source (about 54MB)...', flush=True)
    raw_hash = fetch(prefix + FILENAME, root / FILENAME, FILE_SHA256)
    card_hash = fetch(prefix + 'README.md', root / 'UPSTREAM-README.md')
    record = dict(repo=REPO, revision=REVISION, license_declared_by_dataset_card='apache-2.0',
                  downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
                  source_url=f'https://huggingface.co/datasets/{REPO}',
                  filename=FILENAME, sha256=raw_hash, dataset_card_sha256=card_hash,
                  note='Public third-party data. Human/contributor annotations do not guarantee correctness. Raw author IDs stay local; derivative export requires filtering and review.')
    (root / 'source-manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
