"""Compatibility helpers for natural role-play messages.

The four-part tagged response experiment has been removed. New chat messages
are stored as one natural string. These helpers only flatten old stored display
metadata and keep parenthetical stage directions out of speech synthesis.
"""
from __future__ import annotations

import re


_STAGE_DIRECTION = re.compile(r"[（(][^（）()\n]{1,120}[）)]")


def tts_text_from_content(content: object) -> str:
    """Remove short parenthetical stage directions before synthesis."""
    original = str(content or "").strip()
    spoken = _STAGE_DIRECTION.sub(" ", original)
    spoken = re.sub(r"\s+", " ", spoken)
    spoken = re.sub(r"\s*([，。！？；：、])\s*", r"\1", spoken).strip()
    return spoken or original


def legacy_display_as_message(display: dict | None, fallback: object = "") -> str:
    """Flatten a rejected four-part message into one natural chat bubble."""
    text = str(fallback or "").strip()
    if not isinstance(display, dict):
        return text
    speech = str(display.get("speech") or text).strip()
    action = str(display.get("action") or "").strip().strip("（）()")
    if (
        action
        and action != "没有明显的外部动作变化。"
        and not speech.startswith(("（", "("))
    ):
        return f"（{action}）{speech}"
    return speech
