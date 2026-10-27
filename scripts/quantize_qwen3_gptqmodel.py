"""Run the project's isolated Qwen3-1.7B GPTQModel 4-bit calibration trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENV_ROOT = ROOT / "outputs" / "gptqmodel-env"
HF_HOME = ENV_ROOT / "hf-cache"
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HOME / "hub")
os.environ["HF_MODULES_CACHE"] = str(HF_HOME / "modules")
os.environ.pop("TRANSFORMERS_CACHE", None)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def load_calibration(path: Path, tokenizer) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"Calibration row {line_no} has no messages list")
            texts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=False,
                )
            )
    if not texts:
        raise ValueError("Calibration file is empty")
    return texts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "outputs" / "models" / "qwen3-1.7b",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=ROOT / "data" / "gptq-calibration-v1" / "zh-calibration.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "models" / "qwen3-1.7b-gptqmodel-4bit-g128",
    )
    parser.add_argument("--min-free-vram-gb", type=float, default=4.7)
    args = parser.parse_args()

    source = args.source.resolve()
    calibration_path = args.calibration.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Source model directory not found: {source}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    import torch
    import transformers
    from transformers import AutoTokenizer
    import gptqmodel
    from gptqmodel import BACKEND, GPTQModel, QuantizeConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this trial is configured for the RTX 4060")
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_gb = free_bytes / 1024**3
    if free_gb < args.min_free_vram_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GiB VRAM is free; required preflight is "
            f"{args.min_free_vram_gb:.2f} GiB. Close no processes automatically."
        )

    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    calibration_texts = load_calibration(calibration_path, tokenizer)
    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        sym=True,
        desc_act=True,
        true_sequential=True,
    )

    print(f"Source: {source}", flush=True)
    print(f"Output: {output}", flush=True)
    print(f"Calibration rows: {len(calibration_texts)}", flush=True)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}; free before load: "
        f"{free_gb:.2f}/{total_bytes / 1024**3:.2f} GiB",
        flush=True,
    )
    print("Loading source weights in FP16 for GPTQ calibration...", flush=True)

    model = GPTQModel.from_pretrained(
        str(source),
        quant_config,
        trust_remote_code=False,
        torch_dtype=torch.float16,
        device="cuda:0",
    )
    print("Running 4-bit GPTQ calibration (batch size 1)...", flush=True)
    quant_log = model.quantize(
        calibration_texts,
        batch_size=1,
        calibration_enable_gpu_cache=False,
        tokenizer=tokenizer,
        backend=BACKEND.TORCH,
        buffered_fwd=True,
        auto_gc=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output), max_shard_size="2GB")
    elapsed_seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(0) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(0) / 1024**3

    source_bytes = directory_bytes(source)
    output_bytes = directory_bytes(output)
    report = {
        "status": "quantized",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_model": str(source),
        "source_model_bytes": source_bytes,
        "output_model": str(output),
        "output_model_bytes": output_bytes,
        "size_ratio_output_to_source": round(output_bytes / source_bytes, 4),
        "size_reduction_percent": round((1 - output_bytes / source_bytes) * 100, 2),
        "quantization": {
            "method": "GPTQ",
            "tool": "GPTQModel",
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "desc_act": True,
            "true_sequential": True,
            "backend": "torch",
            "source_dtype": "float16",
        },
        "calibration": {
            "file": str(calibration_path),
            "sha256": sha256_file(calibration_path),
            "samples": len(calibration_texts),
            "kind": "project-authored synthetic text, calibration only; not SFT data or evaluation",
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gptqmodel": getattr(gptqmodel, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(0),
            "vram_total_gib": round(total_bytes / 1024**3, 2),
            "vram_free_before_gib": round(free_gb, 2),
            "peak_torch_allocated_gib": round(peak_allocated, 2),
            "peak_torch_reserved_gib": round(peak_reserved, 2),
        },
        "duration_seconds": round(elapsed_seconds, 1),
        "gptq_layer_log_entries": len(quant_log) if isinstance(quant_log, dict) else None,
    }
    report_path = ROOT / "outputs" / "reports" / "gptqmodel-qwen3-1.7b-4bit-v1.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved quantized model: {output}", flush=True)
    print(f"Saved run report: {report_path}", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
