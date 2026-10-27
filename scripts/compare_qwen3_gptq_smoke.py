"""Compare the base and GPTQModel checkpoint on a small diagnostic-only prompt set."""

from __future__ import annotations

import gc
import json
import os
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
# GPTQModel's TorchQuantLinear normally wraps dequantization in torch.compile;
# this Windows setup has no Triton wheel, so use the eager Torch path for the trial.
os.environ["TORCHDYNAMO_DISABLE"] = "1"

SOURCE = ROOT / "outputs" / "models" / "qwen3-1.7b"
QUANTIZED = ROOT / "outputs" / "models" / "qwen3-1.7b-gptqmodel-4bit-g128"
PROMPTS = [
    "我第一次去看现场音乐演出，座位离舞台比较远会不会影响体验？简单回答两句就好。",
    "新买的白衬衫第一次下水，我应该注意什么才不容易洗坏？简单说重点就行。",
    "家里的台灯换了新灯泡还是不亮，我能先排查哪些地方？请给我两三个检查步骤。",
]


def make_prompt(tokenizer, user_text: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int) -> dict:
    encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = encoded.to(device)
    input_length = encoded["input_ids"].shape[-1]
    torch = __import__("torch")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=50,
            use_cache=True,
            eos_token_id=[151645, 151643],
            pad_token_id=151643,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    new_tokens = max(0, int(output.shape[-1]) - input_length)
    text = tokenizer.decode(output[0, input_length:], skip_special_tokens=True).strip()
    return {
        "text": text,
        "input_tokens": input_length,
        "output_tokens": new_tokens,
        "generation_seconds": round(elapsed, 3),
        "output_tokens_per_second": round(new_tokens / elapsed, 2) if elapsed > 0 else None,
    }


def run_base(tokenizer):
    import torch
    from transformers import AutoModelForCausalLM

    print("Loading original FP16 base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        SOURCE,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Base device map: {getattr(model, 'hf_device_map', 'n/a')}", flush=True)
    results = []
    for index, question in enumerate(PROMPTS, start=1):
        result = generate_one(model, tokenizer, make_prompt(tokenizer, question), 128)
        print(f"Base {index}/{len(PROMPTS)}: {result['output_tokens']} tokens", flush=True)
        results.append(result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run_quantized(tokenizer):
    import torch
    from gptqmodel import BACKEND, GPTQModel

    print("Loading GPTQModel 4-bit checkpoint...", flush=True)
    quantized = GPTQModel.from_quantized(
        str(QUANTIZED),
        device="cuda:0",
        backend=BACKEND.TORCH,
        trust_remote_code=False,
    )
    quantized.eval()
    model = quantized.model
    print("GPTQ runtime backend: TorchQuantLinear", flush=True)
    results = []
    for index, question in enumerate(PROMPTS, start=1):
        result = generate_one(model, tokenizer, make_prompt(tokenizer, question), 128)
        print(f"GPTQ {index}/{len(PROMPTS)}: {result['output_tokens']} tokens", flush=True)
        results.append(result)
    del model
    del quantized
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main() -> None:
    import torch
    import transformers
    import gptqmodel
    from transformers import AutoTokenizer

    if not SOURCE.is_dir() or not QUANTIZED.is_dir():
        raise FileNotFoundError("Both source and GPTQ model directories must exist")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    free, total = torch.cuda.mem_get_info(0)
    print(
        f"GPU {torch.cuda.get_device_name(0)}; free {free / 1024**3:.2f}/"
        f"{total / 1024**3:.2f} GiB",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)

    base_results = run_base(tokenizer)
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    quantized_results = run_quantized(tokenizer)

    report = {
        "status": "smoke_completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "diagnostic smoke only; not the frozen evaluation or author rating",
        "settings": {
            "prompt_style": "Qwen3 chat template, enable_thinking=False",
            "decoding": "greedy, max_new_tokens=128",
            "runtime": "sequential loads; original uses Transformers, GPTQ uses GPTQModel TorchQuantLinear",
        },
        "environment": {
            "python": __import__("platform").python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gptqmodel": getattr(gptqmodel, "__version__", "unknown"),
            "gpu": torch.cuda.get_device_name(0),
        },
        "items": [
            {
                "id": f"smoke-{index:02d}",
                "prompt": question,
                "base": base,
                "gptq": quant,
            }
            for index, (question, base, quant) in enumerate(
                zip(PROMPTS, base_results, quantized_results), start=1
            )
        ],
    }
    report_path = ROOT / "outputs" / "reports" / "gptqmodel-qwen3-1.7b-smoke-v1.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qwen3-1.7B GPTQModel 冒烟对照 v1",
        "",
        "> 这是加载与回答差异的诊断试跑，不是冻结评测、盲评或正式质量结论。",
        "> GPTQ 校准集只有 24 条，工具提示样本数/长度不足；量化结果需谨慎解释。",
        "",
        f"- 日期：{report['created_at_utc']} UTC",
        f"- 解码：Qwen3 对话模板，关闭 thinking，贪心解码，最多 128 个新 token。",
        "- 两个模型依次加载，避免同时占用显存。原始模型用 Transformers；量化模型用 GPTQModel TorchQuantLinear。",
        "",
    ]
    for item in report["items"]:
        lines.extend(
            [
                f"## {item['id']}",
                "",
                f"**用户：** {item['prompt']}",
                "",
                f"**原始模型：** {item['base']['text']}",
                "",
                f"*{item['base']['output_tokens']} tokens，{item['base']['generation_seconds']} 秒，"
                f"{item['base']['output_tokens_per_second']} token/s*",
                "",
                f"**GPTQ：** {item['gptq']['text']}",
                "",
                f"*{item['gptq']['output_tokens']} tokens，{item['gptq']['generation_seconds']} 秒，"
                f"{item['gptq']['output_tokens_per_second']} token/s*",
                "",
            ]
        )
    markdown_path = ROOT / "outputs" / "reports" / "gptqmodel-qwen3-1.7b-smoke-v1.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved JSON: {report_path}", flush=True)
    print(f"Saved comparison: {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
