"""Download the OpenRAIL-M Stable Diffusion 1.5 bundle for MediaPipe Android."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


SOURCE_URL = (
    "https://huggingface.co/na5h13/stable-diffusion-v1-5-mediapipe/"
    "resolve/main/sd15-mediapipe.zip"
)
SOURCE_PAGE = "https://huggingface.co/na5h13/stable-diffusion-v1-5-mediapipe"
LICENSE = "CreativeML OpenRAIL-M"
EXPECTED_VOCAB = "bpe_simple_vocab_16e6.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, output: Path) -> None:
    partial = output.with_suffix(output.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=60) as response:
        if offset and response.status != 206:
            offset = 0
            partial.unlink(missing_ok=True)
        total_header = response.headers.get("Content-Length")
        total = int(total_header) + offset if total_header else 0
        mode = "ab" if offset else "wb"
        written = offset
        with partial.open(mode) as target:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                written += len(chunk)
                if total:
                    percent = written * 100 / total
                    print(
                        f"\r下载中 {percent:5.1f}%  {written / 2**30:.2f}/{total / 2**30:.2f} GiB",
                        end="",
                        flush=True,
                    )
    print()
    partial.replace(output)


def inspect_bundle(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
    names = [item.filename for item in infos]
    bin_names = [name for name in names if name.lower().endswith(".bin")]
    vocab_names = [name for name in names if name.endswith(EXPECTED_VOCAB)]
    if not bin_names:
        raise RuntimeError("模型压缩包中未找到 .bin 权重")
    if not vocab_names:
        raise RuntimeError(f"模型压缩包中未找到 {EXPECTED_VOCAB}")
    return {
        "bin_files": len(bin_names),
        "vocab_file": vocab_names[0],
        "largest_bin_bytes": max(
            item.file_size for item in infos if item.filename.lower().endswith(".bin")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/models/mediapipe-sd15/sd15-mediapipe.zip"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or args.force:
        download(SOURCE_URL, output)
    else:
        print(f"复用已有文件：{output}")

    report = inspect_bundle(output)
    report.update(
        source_url=SOURCE_URL,
        source_page=SOURCE_PAGE,
        license=LICENSE,
        sha256=sha256(output),
        byte_size=output.stat().st_size,
        downloaded_at=datetime.now(timezone.utc).isoformat(),
    )
    output.with_suffix(".source.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
