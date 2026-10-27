"""Generate paired base/LoRA replies for a small, blinded LCCC pilot review."""
import argparse
import hashlib
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import transformers
from transformers import set_seed

from characters import conversation_messages
from inference import GENERATION_SETTINGS, QwenChat
from training_data import audit_lccc, read_records


STRATA = ("two_utterances", "three_utterances", "four_plus")


def select_cases(records, case_count=15, seed=20261023, strata=STRATA):
    if not strata or len(set(strata)) != len(strata) or any(item not in STRATA for item in strata):
        raise ValueError("评估分层配置无效")
    if case_count < len(strata):
        raise ValueError("至少需要每种对话长度各选一题")
    by_dialog = {}
    for row in records:
        current = by_dialog.get(row["source_dialog_id"])
        if current is None or row["target_turn_index"] > current["target_turn_index"]:
            by_dialog[row["source_dialog_id"]] = row
    groups = {category: [] for category in STRATA}
    for row in by_dialog.values():
        groups[row["stratum"]].append(row)
    quotas = {category: case_count // len(strata) for category in strata}
    for category in strata[:case_count % len(strata)]:
        quotas[category] += 1
    rng = random.Random(seed)
    chosen = []
    for category in strata:
        candidates = sorted(groups[category], key=lambda row: (int(row["source_dialog_id"]), row["id"]))
        if len(candidates) < quotas[category]:
            raise ValueError(f"{category} 只有 {len(candidates)} 个对话，无法选 {quotas[category]} 题")
        chosen.extend(rng.sample(candidates, quotas[category]))
    rng.shuffle(chosen)
    return chosen


def select_cases_by_ids(records, case_ids):
    if not case_ids or len(set(case_ids)) != len(case_ids):
        raise ValueError("人工筛选的题目编号为空或重复")
    by_id = {row["id"]: row for row in records}
    missing = [case_id for case_id in case_ids if case_id not in by_id]
    if missing:
        raise ValueError(f"测试拆分中找不到题目编号：{', '.join(missing)}")
    cases = [by_id[case_id] for case_id in case_ids]
    if any(row["stratum"] != "four_plus" or len(row["messages"]) < 3 for row in cases):
        raise ValueError("人工筛选的题目必须来自多轮测试对话")
    if len({row["source_dialog_id"] for row in cases}) != len(cases):
        raise ValueError("人工筛选的题目必须来自不同对话")
    return cases


def render_review(packet, multi_turn_only=False):
    title = "# 中文多轮聊天回复盲评" if multi_turn_only else "# 中文聊天回复盲评"
    context_note = (
        "每题包含至少三条上下文消息，适合判断模型是否接住了前面的对话。"
        if multi_turn_only else "每题展示一段真实人类对话上下文。"
    )
    lines = [
        title,
        "",
        f"这些题取自 LCCC-base 官方 test 拆分的本地小样本。{context_note}每题有两份模型回复，A/B 来源逐题随机。",
        "两种模型都使用 App 中“小满（日常搭子）”的同一角色设定。",
        "请先看上下文，再判断哪份回复更像自然聊天、是否接住了对方的话。语料参考回复不展示，也不要求模型照抄。",
        "",
        "每题只需勾选一个结果：A 更自然 / B 更自然 / 差不多 / 两个都不自然。可选写一句原因。",
        "",
    ]
    for index, case in enumerate(packet, 1):
        lines += [f"## 第 {index} 题", "", "对话上下文：", ""]
        for message in case["messages"]:
            speaker = "对话者1" if message["role"] == "user" else "对话者2"
            lines.append(f"- **{speaker}：** {message['content']}")
        lines += ["", f"**回答 A：** {case['answers']['A']}", "", f"**回答 B：** {case['answers']['B']}", "",
                  "你的选择：", "", "- [ ] A 更自然", "- [ ] B 更自然", "- [ ] 差不多",
                  "- [ ] 两个都不自然", "", "原因（可不填）：", "", "---", ""]
    return "\n".join(lines)


def randomized_mode_mapping(rng):
    modes = ["base", "lora"]
    rng.shuffle(modes)
    return dict(zip(("A", "B"), modes))


def companion_review_messages(messages):
    """Use the same Xiao Man system prompt as the app's companion chat."""
    return conversation_messages(dict(character_id="companion", messages=messages))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/lccc-v1-pilot")
    parser.add_argument("--adapter-run", default="outputs/lora-lccc-base-v1-pilot")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--case-count", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--multi-turn-only", action="store_true",
                        help="只从至少三条上下文消息的四轮及以上对话中抽题")
    parser.add_argument("--case-ids-file", default=None,
                        help="按文件中的测试样本 ID 和顺序选题；每行一个 ID，可用 # 开头写注释")
    args = parser.parse_args()
    data_root = Path(args.data_dir)
    audit = audit_lccc(data_root)
    adapter_root = Path(args.adapter_run)
    run_config = json.loads((adapter_root / "run-config.json").read_text(encoding="utf-8"))
    profile = run_config["model_profile"]
    if profile != "qwen3-1.7b":
        raise ValueError("评估要求 Qwen3-1.7B 基座")
    if run_config["data"]["sha256"]["eval"] != audit["sha256"]["eval"]:
        raise ValueError("Adapter 训练与 LCCC test 评估数据指纹不一致")
    manifest = json.loads((data_root / "freeze-manifest.json").read_text(encoding="utf-8"))
    records = read_records(data_root / "eval.jsonl")
    if args.case_ids_file:
        case_ids = [line.strip() for line in Path(args.case_ids_file).read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        cases = select_cases_by_ids(records, case_ids)
        if len(cases) != args.case_count:
            raise ValueError(f"人工筛选了 {len(cases)} 题，但 --case-count 是 {args.case_count}")
        selected_strata = sorted({case["stratum"] for case in cases})
    else:
        selected_strata = ("four_plus",) if args.multi_turn_only else STRATA
        cases = select_cases(records, args.case_count, args.seed, strata=selected_strata)
        if args.multi_turn_only and any(len(case["messages"]) < 3 for case in cases):
            raise ValueError("多轮盲评样本必须至少包含三条上下文消息")

    root = Path(args.output_dir or ("outputs/lccc-base-blind-review-" + datetime.now().strftime("%Y%m%d-%H%M%S")))
    root.mkdir(parents=True, exist_ok=False)
    packet_dir = root / "reviewer-packet"
    packet_dir.mkdir()
    engine = QwenChat(adapter_run=adapter_root, model_profile=profile, trim_history=True)

    rng = random.SystemRandom()
    packet, key, private_records = [], {}, []
    for case in cases:
        answers = {}
        mapping = randomized_mode_mapping(rng)
        messages = companion_review_messages(case["messages"])
        for label in ("A", "B"):
            mode = mapping[label]
            set_seed(args.seed)
            answer, metrics = engine.reply(messages, model_mode=mode)
            answers[label] = answer
            private_records.append(dict(case_id=case["id"], label=label, mode=mode,
                                        answer=answer, metrics=metrics))
        packet.append(dict(case_id=case["id"], stratum=case["stratum"],
                           messages=case["messages"], answers=answers))
        key[case["id"]] = mapping
        print(f"Generated paired replies for {case['id']}", flush=True)

    (packet_dir / "score-sheet.md").write_text(
        render_review(packet, multi_turn_only=args.multi_turn_only or bool(args.case_ids_file)),
        encoding="utf-8",
    )
    score_rows = [dict(case_id=case["case_id"], preference=None, reason=None) for case in packet]
    (packet_dir / "scores.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in score_rows), encoding="utf-8",
    )
    (root / "organizer-key.json").write_text(json.dumps({
        "mapping": key,
        "note": "评审者请不要打开此文件；它揭示了A/B对应的模型。",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "private-results.json").write_text(json.dumps(private_records, ensure_ascii=False, indent=2), encoding="utf-8")
    config = dict(
        timestamp=datetime.now().astimezone().isoformat(),
        model=run_config["base_model"], revision=run_config["revision"],
        adapter_sha256=hashlib.sha256((adapter_root / "adapter" / "adapter_model.safetensors").read_bytes()).hexdigest(),
        dataset_sha256=audit["sha256"]["eval"],
        source_dialogues=[case["source_dialog_id"] for case in cases],
        case_ids=[case["id"] for case in cases], case_count=len(cases),
        selection_seed=args.seed, selected_strata=list(selected_strata),
        selection_method="prompt-curated-heldout-ids" if args.case_ids_file else "random-by-stratum",
        generation_settings=GENERATION_SETTINGS,
        enable_thinking=False, dtype="bfloat16", torch=torch.__version__,
        transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
        character_id="companion",
        dataset_role="LCCC-base author test split, locally filtered and frozen",
        source_scope=manifest["use_scope"],
        note=("Prompts were selected before inspecting generated answers; 15 cases are a small pilot, not evidence of general quality improvement."
              if args.case_ids_file else "Blind preference pilot only; 15 cases do not establish general quality improvement."),
    )
    (root / "run-config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "review-pending", "output_dir": str(root),
                      "case_count": len(packet), "score_sheet": str(packet_dir / "score-sheet.md")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
