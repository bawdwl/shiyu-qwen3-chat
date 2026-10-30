"""Isolated BGE embedding worker using a line-delimited JSON protocol."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

# Keep the Windows OpenMP workaround inside this isolated process. The web
# process may already load ASR/TTS runtimes that ship another OpenMP copy.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='strict')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    from memory_embeddings import BgeSmallZhEmbedder

    embedder = BgeSmallZhEmbedder(
        args.model,
        device=args.device,
        max_length=args.max_length,
    )
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            texts = request.get("texts")
            if not isinstance(texts, list):
                raise ValueError("texts must be a list")
            response = {"vectors": embedder.encode(texts)}
        except Exception as exc:
            if os.environ.get("SHIYU_MEMORY_EMBEDDING_DEBUG") == "1":
                traceback.print_exc()
            response = {"error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
