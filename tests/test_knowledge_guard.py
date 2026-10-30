import unittest

from knowledge_guard import (
    query_boundary_risk,
    replace_forbidden_mentions,
    strict_prompt,
)
from prompt_pipeline import assemble_prompt


LIBAI_CARD = {
    "id": "custom-libai",
    "name": "李白",
    "knowledge_boundary": {
        "character_kind": "historical",
        "era": "唐朝",
        "knowledge_cutoff": "唐代",
        "forbidden_terms": ["python", "电脑"],
        "can_learn_from_user": True,
    },
}


class KnowledgeGuardTests(unittest.TestCase):
    def test_historical_character_detects_modern_topic(self):
        risk = query_boundary_risk(LIBAI_CARD, "李白，Python 是什么？")
        self.assertTrue(risk["risky"])
        self.assertIn("python", [item.casefold() for item in risk["matched_terms"]])
        prompt = strict_prompt(LIBAI_CARD, "李白，Python 是什么？")
        self.assertIn("严格角色知识边界", prompt)
        self.assertIn("用户愿意解释", prompt)

    def test_forbidden_output_is_replaced_with_safe_character_response(self):
        answer, violations = replace_forbidden_mentions(
            LIBAI_CARD, "Python 是一种现代编程语言。"
        )
        self.assertIn("python", [item.casefold() for item in violations])
        self.assertIn("未曾听闻", answer)
        self.assertNotIn("编程语言", answer)

    def test_prompt_pipeline_adds_strict_boundary_section(self):
        registry = {
            "assistant": {"id": "assistant", "name": "助手", "prompt": "你是助手。"},
            "custom-libai": {
                "id": "custom-libai",
                "name": "李白",
                "prompt": "你是李白。",
                "custom_card": LIBAI_CARD,
            }
        }
        current = {
            "id": "session",
            "character_id": "custom-libai",
            "messages": [],
        }
        messages, trace = assemble_prompt(
            current, "请解释 Python。", registry, {"items": [], "notes": ""}
        )
        self.assertTrue(trace["strict_knowledge_mode"])
        self.assertIn("严格角色知识边界", messages[0]["content"])


if __name__ == "__main__":
    unittest.main()
