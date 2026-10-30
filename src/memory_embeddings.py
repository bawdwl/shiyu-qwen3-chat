"""Local semantic embeddings for the PC memory-retrieval pipeline.

The project deliberately keeps this optional. Chat and structured memory must
continue to work from FTS and deterministic lexical matching when the local
BGE model has not been downloaded or cannot be loaded.
"""
from __future__ import annotations

import logging
import json
import os
import subprocess
import sys
from pathlib import Path
from threading import RLock


LOGGER = logging.getLogger(__name__)
DEFAULT_BGE_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_BGE_MODEL_PATH = Path("outputs/models/bge-small-zh-v1.5")
BGE_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BgeSmallZhEmbedder:
    """Lazy CPU/GPU embedder using the BGE model's CLS-pooled representation."""

    provider_name = "bge"
    model_name = DEFAULT_BGE_MODEL_NAME

    def __init__(self, model_path, *, device=None, max_length=256):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"BGE model directory not found: {self.model_path}")
        self.device_preference = device
        self.max_length = max(32, min(512, int(max_length)))
        self._lock = RLock()
        self._tokenizer = None
        self._model = None
        self._device = None
        self._dimension = None
        self.revision = self._read_revision()

    def _read_revision(self):
        manifest = self.model_path / "model_source.json"
        if not manifest.is_file():
            return None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        revision = payload.get("revision")
        return revision if isinstance(revision, str) and revision.strip() else None

    @property
    def identity(self):
        return self.model_name + (
            f"@{self.revision}" if self.revision else ""
        )

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer
        import torch

        with self._lock:
            if self._model is not None:
                return
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self._model = AutoModel.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            self._model.eval()
            requested = self.device_preference
            if requested in (None, "auto"):
                requested = "cuda" if torch.cuda.is_available() else "cpu"
            if requested == "cuda" and not torch.cuda.is_available():
                requested = "cpu"
            self._device = torch.device(requested)
            self._model.to(self._device)

    def encode(self, texts):
        import torch
        import torch.nn.functional as functional

        normalized = [str(text or "").strip() or " " for text in texts]
        if not normalized:
            return []
        self._load()
        with self._lock, torch.inference_mode():
            encoded = self._tokenizer(
                normalized,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: value.to(self._device) for name, value in encoded.items()}
            output = self._model(**encoded)
            # BGE-small-zh-v1.5 uses the CLS representation and cosine distance.
            vectors = functional.normalize(
                output.last_hidden_state[:, 0, :],
                p=2,
                dim=1,
            )
            self._dimension = int(vectors.shape[1])
            return vectors.detach().cpu().tolist()

    def encode_query(self, text):
        return self.encode([BGE_QUERY_INSTRUCTION + str(text)])[0]

    @property
    def dimension(self):
        self._load()
        if self._dimension is None:
            self.encode(["初始化"])
        return self._dimension


class BgeWorkerEmbedder:
    """Persistent subprocess bridge that isolates BGE's native runtime."""

    provider_name = "bge-worker"
    model_name = DEFAULT_BGE_MODEL_NAME

    def __init__(self, model_path, *, device=None, max_length=256):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"BGE model directory not found: {self.model_path}")
        self.device = device or "cpu"
        self.max_length = max(32, min(512, int(max_length)))
        self._process = None
        self._lock = RLock()
        self.revision = self._read_revision()

    def _read_revision(self):
        manifest = self.model_path / "model_source.json"
        if not manifest.is_file():
            return None
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        revision = payload.get("revision")
        return revision if isinstance(revision, str) and revision.strip() else None

    @property
    def identity(self):
        return self.model_name + (
            f"@{self.revision}" if self.revision else ""
        )

    def _start(self):
        if self._process is not None and self._process.poll() is None:
            return
        worker = Path(__file__).with_name("memory_embedding_worker.py")
        command = [
            sys.executable,
            "-u",
            str(worker),
            "--model",
            str(self.model_path),
            "--device",
            str(self.device),
            "--max-length",
            str(self.max_length),
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        environment = {
            **os.environ,
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
        cache_root = Path(__file__).resolve().parents[1] / "outputs" / ".cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        environment.setdefault("HF_HOME", str(cache_root / "huggingface"))
        environment.setdefault(
            "TRANSFORMERS_CACHE",
            str(cache_root / "transformers"),
        )
        self._process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=creationflags,
            env=environment,
        )

    def encode(self, texts):
        normalized = [str(text or "") for text in texts]
        if not normalized:
            return []
        with self._lock:
            try:
                self._start()
                self._process.stdin.write(
                    json.dumps({"texts": normalized}, ensure_ascii=False) + "\n"
                )
                self._process.stdin.flush()
                line = self._process.stdout.readline()
                if not line:
                    raise RuntimeError("BGE worker exited without a response")
                response = json.loads(line)
            except Exception:
                self.close()
                raise
        if response.get("error"):
            raise RuntimeError(response["error"])
        vectors = response.get("vectors")
        if not isinstance(vectors, list) or len(vectors) != len(normalized):
            raise RuntimeError("BGE worker returned an invalid vector count")
        return vectors

    def encode_query(self, text):
        return self.encode([BGE_QUERY_INSTRUCTION + str(text)])[0]

    def close(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass


def load_default_embedder(model_path=None, *, device=None):
    """Return a BGE embedder when a complete local model directory exists."""
    path = Path(model_path) if model_path else DEFAULT_BGE_MODEL_PATH
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    required = (
        path / "config.json",
        path / "vocab.txt",
    )
    has_weights = any(
        (path / name).is_file()
        for name in ("model.safetensors", "pytorch_model.bin")
    )
    if not all(item.is_file() for item in required) or not has_weights:
        LOGGER.info("local BGE model not available at %s; using lexical retrieval", path)
        return None
    try:
        return BgeWorkerEmbedder(path, device=device)
    except Exception:
        LOGGER.exception("failed to initialize local BGE embedder at %s", path)
        return None
