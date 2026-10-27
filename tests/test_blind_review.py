import copy
import random
import unittest
from prepare_blind_review import make_packet, summarize_reviews, DIMENSIONS


class BlindReviewTests(unittest.TestCase):
    def setUp(self):
        self.cases = [dict(id='one', category='demo', prompts=['question'],
                           content_reference='reference', content_rubric=['point'])]
        row = dict(id='one', error=None, turns=[dict(prompt='question', answer='text',
                   ended_with_eos=True, seconds=1, output_tokens=2)])
        self.packet, self.mapping, self.rows = make_packet(self.cases, [row], [row], random.Random(1))

    def test_packet_excludes_source_and_measurement_metadata(self):
        sample = self.packet[0]['samples']['A']
        self.assertEqual(set(sample), {'failed', 'turns'})
        self.assertEqual(set(sample['turns'][0]), {'prompt', 'answer', 'truncated'})
        self.assertEqual(set(self.mapping['one'].values()), {'base', 'lora'})

    def test_pending_is_not_zero_and_has_no_pair(self):
        report = summarize_reviews(self.rows, self.mapping)
        self.assertIsNone(report['models']['base']['correctness']['mean'])
        self.assertEqual(report['models']['base']['correctness']['pending'], 1)
        self.assertEqual(report['paired']['correctness']['paired'], 0)

    def test_paired_direction_and_partial_dimensions(self):
        for row in self.rows:
            mode = self.mapping['one'][row['sample_label']]
            row.update(reason='actual review example', reviewer='tester', reviewed_at='2026-10-22')
            row['scores']['correctness'] = 2 if mode == 'base' else 0
        report = summarize_reviews(self.rows, self.mapping)
        self.assertEqual(report['paired']['correctness']['regressed'], 1)
        self.assertEqual(report['paired']['correctness']['mean_delta'], -2)
        self.assertEqual(report['paired']['completeness']['paired'], 0)

    def test_invalid_scores_and_missing_identity_rejected(self):
        for invalid in [True, 3, -1, 1.0, '2']:
            rows = copy.deepcopy(self.rows)
            rows[0]['scores']['correctness'] = invalid
            with self.assertRaises(ValueError):
                summarize_reviews(rows, self.mapping)
        self.rows[0]['scores']['correctness'] = 2
        with self.assertRaises(ValueError):
            summarize_reviews(self.rows, self.mapping)

    def test_duplicate_missing_and_extra_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            summarize_reviews([self.rows[0], self.rows[0]], self.mapping)
        with self.assertRaises(ValueError):
            summarize_reviews(self.rows[:1], self.mapping)
        self.rows[0]['scores']['extra'] = None
        with self.assertRaises(ValueError):
            summarize_reviews(self.rows, self.mapping)

    def test_mismatched_batch_and_invalid_mapping_rejected(self):
        with self.assertRaises(ValueError):
            summarize_reviews(self.rows, self.mapping, 'a-different-batch')
        with self.assertRaises(ValueError):
            summarize_reviews(self.rows, {'one': {'A': 'base', 'B': 'base'}})


if __name__ == '__main__':
    unittest.main()
