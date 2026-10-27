"""Create a local-only human review packet for LCCC training candidates.

This packet samples only from the already frozen LCCC train split. It is a
corpus-suitability check, not a model evaluation set and not training data.
LCCC source and derived material stay local under the project's stricter
research-only interpretation of the upstream terms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

from inspect_lccc import CONTACT_PATTERNS
from training_data import audit_lccc, file_hash, read_records


BUCKETS = ((8, 15), (16, 30), (31, 60))
VISIBLE_PII = re.compile(r"\d{5,}|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
CONTACT_REQUEST = re.compile(
    r"(?:加我(?:微信|微.?信|QQ)|私信我|发我(?:微信|QQ)|(?:微信|QQ)号|手机号|身份证号|联系方式)"
)


def candidate_reason(row: dict, low: int, high: int) -> str | None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 4 or len(messages) % 2:
        return "not_multi_turn"
    if messages[-1].get("role") != "assistant":
        return "missing_assistant_target"
    answer = messages[-1].get("content", "")
    if not isinstance(answer, str) or not low <= len(answer) <= high:
        return "target_length_outside_bucket"
    text = "\n".join(str(message.get("content", "")) for message in messages)
    if any(pattern.search(text) for pattern in CONTACT_PATTERNS.values()) or CONTACT_REQUEST.search(text):
        return "contact_pattern"
    if VISIBLE_PII.search(text):
        return "digit_or_email_pattern"
    if any("\ufffd" in message.get("content", "") or "\x00" in message.get("content", "")
           for message in messages):
        return "invalid_character"
    return None


def render_packet(selected: list[tuple[str, dict]], data_hash: str, seed: int) -> str:
    lines = [
        "# LCCC 训练样本适用性小审阅",
        "",
        "这不是在给模型打分，也不是让你判断原始聊天双方谁对谁错。每题只看最后一条助手回复：",
        "**如果要训练‘小满’这样的日常聊天角色，你愿意把这条回复作为正面示范吗？**",
        "",
        "- `可直接用`：自然接住当前上下文，没有明显身份混淆、凭空编造或不合适表达。简短回复只要符合上下文，也可以直接用。",
        "- `不适合`：明显生硬、答非所问、只用没有信息的短词敷衍结束，或把真人经历错误地教给 AI 角色。短本身不是缺点。",
        "- `需改写/不确定`：方向可能有用，但原样训练不放心。",
        "- 每题可选一个最主要问题：`上下文`、`不自然`、`敷衍短答`、`冒充真人经历`、`事实不稳`、`不适合/不安全`、`其他`。理由可留空。",
        "",
        "这些例子按回复长度分层抽样，只来自冻结 LCCC 训练拆分；因此不能据此估计整个数据集的总体质量。样本用于决定是否值得继续人工筛选，不会自动进入训练。数据和本审阅包仅留本机。",
        f"",
        f"抽样种子：`{seed}`；训练文件 SHA-256：`{data_hash}`。",
        "",
    ]
    for case_id, row in selected:
        lines.extend([f"## {case_id}", ""])
        for message in row["messages"][-6:]:
            speaker = "用户" if message["role"] == "user" else "对话中的另一方"
            lines.append(f"- **{speaker}：** {message['content']}")
        lines.extend([
            "",
            "**你的判断：** `可直接用 / 不适合 / 需改写/不确定`",
            "",
            "**主要问题（可空）：**",
            "",
            "---",
            "",
        ])
    lines.extend([
        "## 结果怎么用",
        "",
        "先看哪些类型被你判为可用、哪些问题反复出现。即使多数通过，也只表示可以继续扩大人工筛选；它不代表模型回答已经改善，也不会把这 24 条直接拿去微调。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/lccc-v1-pilot")
    parser.add_argument("--output-dir", default="outputs/lccc-data-curation-review-v1")
    parser.add_argument("--per-bucket", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if args.per_bucket < 1:
        parser.error("--per-bucket 必须大于0")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"拒绝覆盖已有审阅包：{output_dir}")
    audit = audit_lccc(data_dir)
    train_path = data_dir / "train.jsonl"
    rows = read_records(train_path)
    by_bucket: dict[str, list[dict]] = {}
    dropped: dict[str, int] = {}
    for low, high in BUCKETS:
        label = f"{low}-{high}"
        candidates = []
        for row in rows:
            reason = candidate_reason(row, low, high)
            if reason is None:
                candidates.append(row)
            else:
                dropped[reason] = dropped.get(reason, 0) + 1
        by_bucket[label] = candidates
        if len({row["source_dialog_id"] for row in candidates}) < args.per_bucket:
            raise ValueError(f"长度组 {label} 去重后候选不足：{len(candidates)}")

    rng = random.Random(args.seed)
    selected: list[tuple[str, dict]] = []
    used_dialog_ids: set[str] = set()
    selection_manifest = []
    for low, high in BUCKETS:
        label = f"{low}-{high}"
        candidates = [row for row in by_bucket[label]
                      if row["source_dialog_id"] not in used_dialog_ids]
        rng.shuffle(candidates)
        chosen = candidates[:args.per_bucket]
        if len(chosen) != args.per_bucket:
            raise ValueError(f"长度组 {label} 不能满足不同源对话的分层抽样")
        for row in chosen:
            used_dialog_ids.add(row["source_dialog_id"])
            case_id = f"C{len(selected) + 1:02d}"
            selected.append((case_id, row))
            selection_manifest.append({
                "case_id": case_id,
                "source_record_id": row["id"],
                "source_dialog_id": row["source_dialog_id"],
                "target_chars": len(row["messages"][-1]["content"]),
                "bucket": label,
                "content_sha256": hashlib.sha256(
                    json.dumps(row["messages"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
            })

    output_dir.mkdir(parents=True, exist_ok=False)
    train_hash = file_hash(train_path)
    (output_dir / "score-sheet.md").write_text(
        render_packet(selected, train_hash, args.seed), encoding="utf-8",
    )
    manifest = {
        "version": "lccc-data-curation-review-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "manual corpus suitability check; not model evaluation or training data",
        "data_dir": str(data_dir),
        "source_split": "train only",
        "source_train_sha256": train_hash,
        "audit_counts": audit["counts"],
        "seed": args.seed,
        "sampling": {
            "per_bucket": args.per_bucket,
            "target_character_buckets": [list(bucket) for bucket in BUCKETS],
            "requires_at_least_two_dialogue_turns": True,
            "one_candidate_per_source_dialogue": True,
            "visible_contact_phone_email_patterns_excluded": True,
        },
        "candidate_counts_by_bucket": {key: len(value) for key, value in by_bucket.items()},
        "selected": selection_manifest,
        "limitations": [
            "LCCC is human-human social-media conversation, not companion-app dialogue.",
            "Length-stratified sampling deliberately over-represents longer replies; it cannot estimate full-corpus quality.",
            "Pattern filtering cannot guarantee complete privacy screening.",
            "The source and review packet are local-only under this project’s stricter research-use interpretation.",
        ],
    }
    (output_dir / "review-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({
        "status": "prepared",
        "output_dir": str(output_dir),
        "selected_count": len(selected),
        "candidate_counts_by_bucket": manifest["candidate_counts_by_bucket"],
        "source_train_sha256": train_hash,
        "source_split": "train only",
        "audit_counts": audit["counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
