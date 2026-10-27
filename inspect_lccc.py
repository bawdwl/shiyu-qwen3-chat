"""Audit LCCC-base compressed JSONL without modifying source files.

The output report contains aggregate counts only. The preview is sampled from
the corpus and redacts common contact identifiers; keep both outputs local.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PHONE = re.compile(r"(?<!\d)(?:(?:\+?86)[ -]?)?1[3-9]\d{9}(?!\d)")
EMAIL = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
ID_CARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
QQ = re.compile(r"(?:QQ|qq|扣扣)(?:号|号码)?\s*[:：]?\s*[1-9]\d{4,10}")
WECHAT = re.compile(r"(?:微信|微.信|v信|vx)(?:号|ID|id)?\s*[:：]?\s*[A-Za-z][-_A-Za-z0-9]{4,}")
CONTACT_PATTERNS = {
    "phone": PHONE,
    "email": EMAIL,
    "url": URL,
    "id_card": ID_CARD,
    "qq": QQ,
    "wechat": WECHAT,
}
HAN = r"\u3400-\u9fff"
PUNCTUATION = "，。！？；：、…,.!?;:"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_dialog(line: str) -> list[str]:
    row = json.loads(line)
    dialog = row.get("dialog") if isinstance(row, dict) else row
    if not isinstance(dialog, list) or len(dialog) < 2 or any(
        not isinstance(turn, str) or not turn.strip() for turn in dialog
    ):
        raise ValueError("invalid dialog structure")
    return dialog


def normalize_chinese_spacing(text: str) -> str:
    """Undo word-level spaces while preserving spaces between Latin words."""
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(rf"(?<=[{HAN}]) +(?=[{HAN}])", "", text)
    text = re.sub(rf"(?<=[{HAN}]) +(?=[A-Za-z0-9])", "", text)
    text = re.sub(rf"(?<=[A-Za-z0-9]) +(?=[{HAN}])", "", text)
    text = re.sub(r"(?<=\d) +(?=\d)", "", text)
    text = re.sub(rf"(?<=[{HAN}]) +(?=[{re.escape(PUNCTUATION)}])", "", text)
    text = re.sub(rf"(?<=[{re.escape(PUNCTUATION)}]) +(?=[{HAN}])", "", text)
    text = re.sub(rf" *([{re.escape(PUNCTUATION)}]) *", r"\1", text)
    text = re.sub(r"(?<![A-Za-z])(?:[A-Za-z] ){2,}[A-Za-z](?![A-Za-z])", lambda m: m.group(0).replace(" ", ""), text)
    return text.strip()


def redact(text: str) -> str:
    text = URL.sub("[链接]", text)
    text = EMAIL.sub("[邮箱]", text)
    text = ID_CARD.sub("[证件号]", text)
    text = PHONE.sub("[手机号]", text)
    text = QQ.sub("[账号]", text)
    text = WECHAT.sub("[账号]", text)
    return text


def percentile_from_hist(hist: Counter[int], p: float, total: int) -> int | None:
    if total == 0:
        return None
    target = max(1, int(total * p + 0.999999))
    cumulative = 0
    for value in sorted(hist):
        cumulative += hist[value]
        if cumulative >= target:
            return value
    return max(hist)


def inspect_split(path: Path, sample_size: int, seed: int) -> dict:
    rng = random.Random(seed)
    turn_counts: Counter[int] = Counter()
    contact_counts: Counter[str] = Counter()
    reservoir: list[tuple[int, list[str]]] = []
    total = malformed = empty_turns = 0
    utterances = characters = 0
    min_turns: int | None = None
    max_turns = 0

    with gzip.open(path, "rt", encoding="utf-8", errors="strict") as stream:
        for line_number, line in enumerate(stream, start=1):
            total += 1
            try:
                dialog = parse_dialog(line)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                malformed += 1
                continue
            turns = len(dialog)
            turn_counts[turns] += 1
            min_turns = turns if min_turns is None else min(min_turns, turns)
            max_turns = max(max_turns, turns)
            utterances += turns
            characters += sum(len(turn) for turn in dialog)
            if any(not turn.strip() for turn in dialog):
                empty_turns += 1
            joined = "\n".join(dialog)
            for name, pattern in CONTACT_PATTERNS.items():
                if pattern.search(joined):
                    contact_counts[name] += 1

            item = (line_number - 1, dialog)
            if len(reservoir) < sample_size:
                reservoir.append(item)
            else:
                slot = rng.randrange(total)
                if slot < sample_size:
                    reservoir[slot] = item

    dialog_total = total - malformed
    return {
        "file": path.name,
        "sha256": sha256(path),
        "rows": total,
        "valid_dialogs": dialog_total,
        "malformed_rows": malformed,
        "single_turn_dialogs": turn_counts[2],
        "dialogs_with_more_than_two_utterances": dialog_total - turn_counts[2],
        "dialogs_with_four_or_more_utterances": sum(
            count for turns, count in turn_counts.items() if turns >= 4
        ),
        "mean_utterances_per_dialog": round(utterances / dialog_total, 3) if dialog_total else None,
        "median_utterances_per_dialog": percentile_from_hist(turn_counts, 0.5, dialog_total),
        "p95_utterances_per_dialog": percentile_from_hist(turn_counts, 0.95, dialog_total),
        "min_utterances_per_dialog": min_turns,
        "max_utterances_per_dialog": max_turns,
        "utterance_count": utterances,
        "mean_characters_per_utterance": round(characters / utterances, 2) if utterances else None,
        "dialogs_with_contact_pattern": dict(contact_counts),
        "sampled_dialog_ids": [row_id for row_id, _ in reservoir],
        "sampled_dialogs": [
            [redact(normalize_chinese_spacing(turn)) for turn in dialog]
            for _, dialog in reservoir
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="data/LCCC-base/source")
    parser.add_argument("--output-dir", default="data/LCCC-base/audit")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有审计结果目录：{output}")
    if args.sample_size < 1:
        parser.error("--sample-size 必须大于0")

    split_files = {
        "train": source / "lccc_base_train.jsonl.gz",
        "validation": source / "lccc_base_valid.jsonl.gz",
        "test": source / "lccc_base_test.jsonl.gz",
    }
    missing = [str(path) for path in split_files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("缺少 LCCC-base 文件：" + ", ".join(missing))

    report = {
        "version": "lccc-base-source-audit-v1",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "thu-coai/lccc official loader URLs (the loader points to silver/lccc files)",
        "use_scope": "research use only; local project; do not redistribute source or derived corpus",
        "seed": args.seed,
        "sample_size_per_split": args.sample_size,
        "splits": {
            name: inspect_split(path, args.sample_size, args.seed + index)
            for index, (name, path) in enumerate(split_files.items())
        },
        "limitations": [
            "Contact checks cover common patterns only and do not prove all personal information has been removed.",
            "The corpus is social-media conversation; an automated audit cannot establish naturalness or rights for every source item.",
            "Sample texts are redacted previews and should remain local.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    preview = []
    for split, stats in report["splits"].items():
        for dialog_id, dialog in zip(stats["sampled_dialog_ids"], stats["sampled_dialogs"]):
            preview.append(f"[{split} row={dialog_id}]\n" + "\n".join(
                f"{'用户' if index % 2 == 0 else '对话者'}：{turn}"
                for index, turn in enumerate(dialog)
            ))
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "redacted-preview.txt").write_text("\n\n".join(preview) + "\n", encoding="utf-8")
    summary = {
        name: {key: value for key, value in stats.items() if key != "sampled_dialogs"}
        for name, stats in report["splits"].items()
    }
    print(json.dumps({"status": "completed", "output_dir": str(output), "splits": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
