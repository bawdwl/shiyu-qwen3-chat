import unittest

from training_recipe import accumulation_factors, lr_multiplier


class TrainingRecipeTests(unittest.TestCase):
    def test_token_accumulation_matches_concatenated_token_loss(self):
        # One 2-token sample has loss 4, an 8-token sample has loss 1.
        # Concatenation must produce (2*4 + 8*1)/10, rather than (4+1)/2.
        factors = accumulation_factors([2, 8], 'tokens')
        self.assertAlmostEqual(sum(w * loss for w, loss in zip(factors, [4.0, 1.0])), 1.6)
        self.assertAlmostEqual(sum(factors), 1.0)

    def test_short_final_group_is_not_scaled_by_nominal_batch_size(self):
        self.assertEqual(accumulation_factors([7], 'tokens'), [1.0])
        self.assertEqual(accumulation_factors([2, 8]), [0.5, 0.5])

    def test_zero_target_sample_fails_instead_of_dividing_by_zero(self):
        for lengths in ([], [0], [2, -1]):
            with self.assertRaises(ValueError):
                accumulation_factors(lengths, 'tokens')

    def test_legacy_constant_and_cosine_warmup_boundaries(self):
        self.assertEqual([lr_multiplier(s, 10) for s in range(1, 11)], [1.0] * 10)
        values = [lr_multiplier(s, 20, 'cosine', 0.1) for s in range(1, 21)]
        self.assertEqual(values[:2], [0.5, 1.0])
        self.assertAlmostEqual(values[-1], 0.0)
        self.assertTrue(all(a >= b for a, b in zip(values[2:], values[3:])))

    def test_single_update_is_nonzero_and_invalid_schedule_is_rejected(self):
        self.assertEqual(lr_multiplier(1, 1, 'cosine', 0.05), 1.0)
        for args in ((0, 3), (4, 3), (1, 3, 'unknown'), (1, 3, 'cosine', 1)):
            with self.assertRaises(ValueError):
                lr_multiplier(*args)


if __name__ == '__main__':
    unittest.main()
