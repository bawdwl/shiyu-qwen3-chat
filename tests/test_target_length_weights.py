import unittest

from training_data import equal_bucket_sample_weights


class TargetLengthWeightTests(unittest.TestCase):
    def test_balances_total_weight_without_dropping_examples(self):
        rows = [
            {'messages': [{'role': 'assistant', 'content': '好'}]},
            {'messages': [{'role': 'assistant', 'content': '知道啦'}]},
            {'messages': [{'role': 'assistant', 'content': '那你今天准备先做哪一件事呢？'}]},
        ]
        weights, audit = equal_bucket_sample_weights(rows, threshold_chars=3)
        self.assertEqual(len(weights), len(rows))
        self.assertEqual(audit['counts'], {'short_le_threshold': 2, 'long_gt_threshold': 1})
        self.assertAlmostEqual(audit['total_weight_by_bucket']['short_le_threshold'], 1.5)
        self.assertAlmostEqual(audit['total_weight_by_bucket']['long_gt_threshold'], 1.5)
        self.assertEqual(weights[0], weights[1])
        self.assertGreater(weights[2], weights[0])

    def test_rejects_empty_or_single_bucket_data(self):
        with self.assertRaises(ValueError):
            equal_bucket_sample_weights([])
        with self.assertRaises(ValueError):
            equal_bucket_sample_weights([{'messages': [{'content': '好的'}]}])


if __name__ == '__main__':
    unittest.main()
