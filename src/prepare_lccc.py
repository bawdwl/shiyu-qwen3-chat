"""Freeze a small, privacy-filtered LCCC-base LoRA pilot.

The source is limited by its upstream notice to research use. Keep source and
derived data local and do not distribute the corpus or this pilot dataset.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from inspect_lccc import CONTACT_PATTERNS, normalize_chinese_spacing, parse_dialog, sha256


STRATA = ("two_utterances", "three_utterances", "four_plus")
WEIGHTS = {"two_utterances": 0.30, "three_utterances": 0.20, "four_plus": 0.50}
CONTACT_REQUEST = re.compile(
    r"(?:加我(?:微信|微.?信|QQ)|私信我|发我(?:微信|QQ)|(?:微信|QQ)号|手机号|身份证号|联系方式)"
)


def classify(dialog: list[str]) -> str:
    if len(dialog) == 2:
        return "two_utterances"
    if len(dialog) == 3:
        return "three_utterances"
    return "four_plus"


def normalize_dialog(dialog: list[str]) -> list[str]:
    return [normalize_chinese_spacing(turn) for turn in dialog]


def contact_hits(dialog: list[str]) -> list[str]:
    text = "\n".join(dialog)
    hits = [name for name, pattern in CONTACT_PATTERNS.items() if pattern.search(text)]
    if CONTACT_REQUEST.search(text):
        hits.append("contact_request")
    return hits


def acceptable(dialog: list[str]) -> tuple[list[str] | None, str | None]:
    if any("\ufffd" in turn or "\x00" in turn for turn in dialog):
        return None, "invalid_encoding"
    normalized = normalize_dialog(dialog)
    if any(not turn or len(turn) > 300 for turn in normalized):
        return None, "utterance_length"
    if sum(map(len, normalized)) > 1800:
        return None, "dialogue_length"
    hits = contact_hits(dialog) + contact_hits(normalized)
    if hits:
        return None, "contact_pattern"
    return normalized, None


def dialog_hash(dialog: list[str]) -> str:
    return hashlib.sha256("\n".join(dialog).encode("utf-8")).hexdigest()


def allocate(total: int) -> dict[str, int]:
    base = {key: int(total * WEIGHTS[key]) for key in STRATA}
    remaining = total - sum(base.values())
    for key in STRATA[:remaining]:
        base[key] += 1
    return base


def reservoir_add(pool: list, item, count: int, capacity: int, rng: random.Random) -> None:
    if len(pool) < capacity:
        pool.append(item)
        return
    slot = rng.randrange(count)
    if slot < capacity:
        pool[slot] = item


def read_train_pool(path: Path, total_rows: int, candidate_rows: int,
                    pool_sizes: dict[str, int], seed: int) -> tuple[dict, dict]:
    rng = random.Random(seed)
    wanted = set(rng.sample(range(total_rows), min(candidate_rows, total_rows)))
    pools = {key: [] for key in STRATA}
    seen = Counter()
    dropped = Counter()
    parsed = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        for row_id, line in enumerate(stream):
            if row_id not in wanted:
                continue
            parsed += 1
            try:
                raw = parse_dialog(line)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                dropped["malformed"] += 1
                continue
            normalized, reason = acceptable(raw)
            if reason:
                dropped[reason] += 1
                continue
            category = classify(normalized)
            seen[category] += 1
            reservoir_add(pools[category], (row_id, normalized), seen[category], pool_sizes[category], rng)
    if parsed != len(wanted):
        raise ValueError(f"train 扫描只找到 {parsed}/{len(wanted)} 个抽样行号")
    return pools, {"candidate_row_ids": len(wanted), "parsed_candidate_rows": parsed,
                   "eligible_candidates_by_stratum": dict(seen), "dropped_candidates": dict(dropped)}


def read_holdout_pool(path: Path, pool_sizes: dict[str, int], seed: int) -> tuple[dict, dict]:
    rng = random.Random(seed)
    pools = {key: [] for key in STRATA}
    seen = Counter()
    dropped = Counter()
    total = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        for row_id, line in enumerate(stream):
            total += 1
            try:
                raw = parse_dialog(line)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                dropped["malformed"] += 1
                continue
            normalized, reason = acceptable(raw)
            if reason:
                dropped[reason] += 1
                continue
            category = classify(normalized)
            seen[category] += 1
            reservoir_add(pools[category], (row_id, normalized), seen[category], pool_sizes[category], rng)
    return pools, {"source_rows": total, "eligible_candidates_by_stratum": dict(seen),
                   "dropped_candidates": dict(dropped)}


def choose_dialogs(pools: dict, quotas: dict[str, int], seen_hashes: set[str],
                   rng: random.Random, split: str) -> tuple[list[dict], dict]:
    chosen = []
    counts = {}
    for category in STRATA:
        candidates = list(pools[category])
        rng.shuffle(candidates)
        selected = []
        local = set()
        for row_id, dialog in candidates:
            digest = dialog_hash(dialog)
            if digest in seen_hashes or digest in local:
                continue
            local.add(digest)
            selected.append({"row_id": row_id, "dialog": dialog, "hash": digest,
                             "stratum": category, "split": split})
            if len(selected) == quotas[category]:
                break
        if len(selected) != quotas[category]:
            raise ValueError(
                f"{split}/{category} 去重后只有 {len(selected)} 条，需 {quotas[category]}；"
                "可提高 train 候选行数或降低样本量"
            )
        chosen.extend(selected)
        seen_hashes.update(row["hash"] for row in selected)
        counts[category] = len(selected)
    rng.shuffle(chosen)
    return chosen, counts


def messages_for(dialog: list[str], end_index: int) -> list[dict[str, str]]:
    return [dict(role="user" if index % 2 == 0 else "assistant", content=text)
            for index, text in enumerate(dialog[:end_index + 1])]


def make_sft_rows(dialogs: list[dict], split: str) -> list[dict]:
    rows = []
    for item in dialogs:
        dialog = item["dialog"]
        for target_index in range(1, len(dialog), 2):
            if split == "eval":
                context = messages_for(dialog, target_index - 1)
                if not context or context[-1]["role"] != "user":
                    raise AssertionError("测试上下文必须以用户消息结束")
                row = dict(
                    id=f"lccc_test_{item['row_id']:07d}_t{target_index:02d}",
                    source="LCCC-base",
                    source_split="test",
                    source_dialog_id=str(item["row_id"]),
                    source_dialog_hash=item["hash"],
                    stratum=item["stratum"],
                    target_turn_index=target_index,
                    messages=context,
                    prompts=[context[-1]["content"]],
                    reference=dialog[target_index],
                )
            else:
                row = dict(
                    id=f"lccc_{split}_{item['row_id']:07d}_t{target_index:02d}",
                    source="LCCC-base",
                    source_split=split,
                    source_dialog_id=str(item["row_id"]),
                    source_dialog_hash=item["hash"],
                    stratum=item["stratum"],
                    target_turn_index=target_index,
                    messages=messages_for(dialog, target_index),
                )
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> str:
    content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                      for row in rows).encode("utf-8")
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def prepare(source_dir: Path, audit_path: Path, output_dir: Path, train_dialogues: int,
            validation_dialogues: int, test_dialogues: int, candidate_rows: int, seed: int) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已存在的冻结数据集：{output_dir}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("version") != "lccc-base-source-audit-v1":
        raise ValueError("LCCC-base 审计报告版本不匹配")
    paths = {
        "train": source_dir / "lccc_base_train.jsonl.gz",
        "validation": source_dir / "lccc_base_valid.jsonl.gz",
        "test": source_dir / "lccc_base_test.jsonl.gz",
    }
    for split, path in paths.items():
        expected = audit["splits"][split]["sha256"]
        if sha256(path) != expected:
            raise ValueError(f"源文件指纹不匹配：{path}")

    quotas = {"train": allocate(train_dialogues),
              "validation": allocate(validation_dialogues), "test": allocate(test_dialogues)}
    pool_sizes = {split: {key: max(value * 3, value + 20) for key, value in split_quotas.items()}
                  for split, split_quotas in quotas.items()}
    pools, train_scan = read_train_pool(paths["train"], audit["splits"]["train"]["rows"],
                                        candidate_rows, pool_sizes["train"], seed)
    holdout_pools, validation_scan = read_holdout_pool(paths["validation"], pool_sizes["validation"], seed + 1)
    test_pools, test_scan = read_holdout_pool(paths["test"], pool_sizes["test"], seed + 2)

    seen_hashes: set[str] = set()
    selected = {}
    selections = {}
    for index, (split, available) in enumerate((
        ("train", pools), ("validation", holdout_pools), ("test", test_pools),
    )):
        selected[split], selections[split] = choose_dialogs(
            available, quotas[split], seen_hashes, random.Random(seed + 10 + index), split,
        )

    output_rows = {
        "train": make_sft_rows(selected["train"], "train"),
        "validation": make_sft_rows(selected["validation"], "validation"),
        "eval": make_sft_rows(selected["test"], "eval"),
    }
    if any(not rows for rows in output_rows.values()):
        raise ValueError("训练、验证、测试输出都必须非空")

    output_dir.mkdir(parents=True, exist_ok=False)
    output_hashes = {split: write_jsonl(output_dir / f"{split}.jsonl", rows)
                     for split, rows in output_rows.items()}
    source_file_hashes = {split: audit["splits"][split]["sha256"] for split in paths}
    manifest = {
        "version": "lccc-base-pilot-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": audit["source"],
        "source_dir": str(source_dir.resolve().relative_to(Path.cwd().resolve())),
        "source_files": source_file_hashes,
        "license": "LCCC upstream research-use-only notice; HF dataset card separately lists MIT",
        "use_scope": "local research use only; do not distribute source or derived corpus",
        "sampling_seed": seed,
        "candidate_rows_scanned": candidate_rows,
        "strata_weights": WEIGHTS,
        "selected_dialogues_by_split": selections,
        "selected_dialogue_counts_by_output": {
            "train": len(selected["train"]),
            "validation": len(selected["validation"]),
            "eval": len(selected["test"]),
        },
        "selected_source_rows_by_split": {
            split: [item["row_id"] for item in rows] for split, rows in selected.items()
        },
        "selected_dialogue_hashes_by_split": {
            split: [item["hash"] for item in rows] for split, rows in selected.items()
        },
        "contact_filter_scan": {"train_candidate_sample": train_scan,
                                "validation": validation_scan, "test": test_scan},
        "normalization": "Remove word-token spaces between Chinese, adjacent Chinese/Latin/digits and Chinese punctuation; preserve spaces between English words.",
        "supervision": "Every assistant turn is a target with all preceding context; test records retain the next assistant reply as a local reference.",
        "outputs": {
            split: {"count": len(rows), "sha256": output_hashes[split]}
            for split, rows in output_rows.items()
        },
        "limitations": [
            "This is a small stratified pilot, not full-dataset training.",
            "LCCC is human-human social-media dialogue and does not provide character identity or app companionship behavior.",
            "Pattern filters catch common contact information only and cannot guarantee de-identification.",
            "Author GitHub and Hugging Face license descriptions differ; this pilot follows the stricter research-only scope.",
        ],
    }
    (output_dir / "freeze-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/LCCC-base/source")
    parser.add_argument("--audit", default="data/LCCC-base/audit/report.json")
    parser.add_argument("--output-dir", default="data/lccc-v1-pilot")
    parser.add_argument("--train-dialogues", type=int, default=800)
    parser.add_argument("--validation-dialogues", type=int, default=120)
    parser.add_argument("--test-dialogues", type=int, default=120)
    parser.add_argument("--candidate-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    for name in ("train_dialogues", "validation_dialogues", "test_dialogues", "candidate_rows"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} 必须大于0")
    manifest = prepare(Path(args.source_dir), Path(args.audit), Path(args.output_dir),
                       args.train_dialogues, args.validation_dialogues, args.test_dialogues,
                       args.candidate_rows, args.seed)
    print(json.dumps({
        "status": "prepared",
        "output_dir": args.output_dir,
        "selected_dialogues_by_split": manifest["selected_dialogues_by_split"],
        "outputs": manifest["outputs"],
        "contact_filter_scan": manifest["contact_filter_scan"],
        "use_scope": manifest["use_scope"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
