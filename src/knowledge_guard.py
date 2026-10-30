"""Deterministic helpers for character knowledge-boundary checks."""
from __future__ import annotations

import re
import unicodedata


MODERN_TERMS = (
    "python",
    "javascript",
    "人工智能",
    "大模型",
    "计算机",
    "电脑",
    "手机",
    "互联网",
    "高铁",
    "飞机",
    "电影",
    "电视",
    "二维码",
    "微信",
    "网络",
    "数据库",
    "编程",
    "算法",
    "smartphone",
    "computer",
    "internet",
    "software",
    "artificial intelligence",
)


def _norm(value: str) -> str:
    return unicodedata.normalize("NFKC", value or "").casefold()


def term_matches(text: str, terms: list[str] | tuple[str, ...]) -> list[str]:
    normalized = _norm(text)
    hits = []
    for term in terms:
        clean = _norm(term).strip()
        if clean and clean in normalized:
            hits.append(term)
    return list(dict.fromkeys(hits))


def query_boundary_risk(card: dict | None, query: str) -> dict:
    """Return a conservative boundary signal for a custom character.

    This is a guard, not a semantic classifier. It intentionally errs toward
    the strict path for historical characters and explicit forbidden terms.
    """
    if not card:
        return {"risky": False, "matched_terms": [], "reason": None}
    boundary = card.get("knowledge_boundary") or {}
    terms = list(boundary.get("forbidden_terms") or [])
    if boundary.get("character_kind") == "historical":
        terms.extend(term for term in MODERN_TERMS if term not in terms)
    hits = term_matches(query, terms)
    return {
        "risky": bool(hits),
        "matched_terms": hits,
        "reason": "time_or_universe_boundary" if hits else None,
    }


def response_boundary_violations(card: dict | None, answer: str) -> list[str]:
    if not card:
        return []
    boundary = card.get("knowledge_boundary") or {}
    terms = list(boundary.get("forbidden_terms") or [])
    if boundary.get("character_kind") == "historical":
        terms.extend(term for term in MODERN_TERMS if term not in terms)
    return term_matches(answer, terms)


def strict_prompt(card: dict | None, query: str) -> str | None:
    """Create a short strict-mode instruction for a suspected boundary query."""
    risk = query_boundary_risk(card, query)
    if not risk["risky"]:
        return None
    boundary = card.get("knowledge_boundary") or {}
    terms = "、".join(risk["matched_terms"])
    learning = (
        "可以先表示自己不熟悉；如果用户愿意解释，再基于用户刚提供的信息讨论，"
        "但必须明确这是听用户说的，不能声称自己原本掌握。"
        if boundary.get("can_learn_from_user", True)
        else "不要接受或利用该范围外知识，也不要解释其内容。"
    )
    return (
        "【严格角色知识边界】本轮问题触及角色不应知道的概念："
        f"{terms}。角色所处时代：{boundary.get('era') or '未指定'}；"
        f"知识截止：{boundary.get('knowledge_cutoff') or '未指定'}。{learning}"
        "禁止输出现代百科式解释，禁止假装亲历，禁止泄露后世或跨世界信息。"
    )


def safe_fallback(card: dict | None) -> str:
    boundary = (card or {}).get("knowledge_boundary") or {}
    if boundary.get("character_kind") == "historical":
        return "此事我未曾听闻。若你愿意，不妨讲给我听听。"
    return "这件事超出我现在能知道的范围了。你若愿意，可以告诉我你知道的部分。"


def replace_forbidden_mentions(card: dict | None, answer: str) -> tuple[str, list[str]]:
    """Fallback for deterministic forbidden terms after generation."""
    violations = response_boundary_violations(card, answer)
    if not violations:
        return answer, []
    # The fallback is intentionally conservative. A later strict-model rewrite
    # can provide a more nuanced response, but unsupported output is not shown.
    return safe_fallback(card), violations

