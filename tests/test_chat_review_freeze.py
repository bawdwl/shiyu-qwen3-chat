import unittest

from prepare_lccc import dialog_hash
from prepare_lccc_chat_review import choose_utterance


class ChatReviewFreezeTests(unittest.TestCase):
    def setUp(self):
        self.dialog = ['今天学了一点吉他，很开心。', '慢慢来就好。', '我打算明天接着练。', '好呀。']

    def test_keeps_original_utterance_without_rewriting_or_reference(self):
        before = list(self.dialog)
        text, digest = choose_utterance(self.dialog, 2, set(), set())
        self.assertEqual(text, before[2])
        self.assertEqual(digest, dialog_hash(before))
        self.assertEqual(self.dialog, before)

    def test_rejects_assistant_turn_out_of_bounds_and_boolean_index(self):
        for turn in (1, 3, 4, -1, True):
            with self.assertRaisesRegex(ValueError, 'user-side'):
                choose_utterance(self.dialog, turn, set(), set())

    def test_rejects_dialogue_overlap_even_when_target_text_is_new(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            choose_utterance(self.dialog, 0, {dialog_hash(self.dialog)}, set())

    def test_rejects_identical_utterance_from_different_dialogue(self):
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            choose_utterance(self.dialog, 0, set(), {self.dialog[0]})


if __name__ == '__main__':
    unittest.main()
