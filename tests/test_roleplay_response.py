import unittest

from characters import NATURAL_ROLEPLAY_INSTRUCTION
from roleplay_response import legacy_display_as_message, tts_text_from_content


class NaturalRoleplayTests(unittest.TestCase):
    def test_natural_instruction_forbids_section_headers(self):
        self.assertIn("不要添加“角色说出口”", NATURAL_ROLEPLAY_INSTRUCTION)
        self.assertIn("简短的中文括号", NATURAL_ROLEPLAY_INSTRUCTION)
        self.assertIn("不默认展示好感度", NATURAL_ROLEPLAY_INSTRUCTION)

    def test_legacy_four_part_message_flattens_to_action_and_speech(self):
        text = legacy_display_as_message(
            {
                "speech": "我才没有等你。",
                "inner_thought": "其实很开心。",
                "action": "别过脸，把杯子握紧。",
                "state": {"affection": 80},
            },
            "fallback",
        )
        self.assertEqual(text, "（别过脸，把杯子握紧。）我才没有等你。")
        self.assertNotIn("其实很开心", text)
        self.assertNotIn("80", text)

    def test_tts_removes_parenthetical_stage_directions(self):
        self.assertEqual(
            tts_text_from_content("（噗地笑出声）你可算来了。（拍拍椅子）坐吧。"),
            "你可算来了。坐吧。",
        )
        self.assertEqual(
            tts_text_from_content("（只写动作）"),
            "（只写动作）",
        )


if __name__ == "__main__":
    unittest.main()
