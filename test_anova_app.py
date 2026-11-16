"""Focused tests for the manually calculated one-way ANOVA."""

import math
import unittest

from anova_app import calculate_one_way_anova, parse_values


class OneWayAnovaTest(unittest.TestCase):
    def test_required_worked_example(self):
        result = calculate_one_way_anova(
            {
                "Group A": [4, 5, 6],
                "Group B": [7, 8, 9],
                "Group C": [10, 11, 12],
            },
            alpha=0.05,
        )

        self.assertEqual(result["ssb"], 54)
        self.assertEqual(result["ssw"], 6)
        self.assertEqual(result["sst"], 60)
        self.assertEqual(result["df_between"], 2)
        self.assertEqual(result["df_within"], 6)
        self.assertEqual(result["df_total"], 8)
        self.assertEqual(result["msb"], 27)
        self.assertEqual(result["msw"], 1)
        self.assertEqual(result["f_statistic"], 27)
        self.assertAlmostEqual(result["f_critical"], 5.1432528498, places=8)
        self.assertAlmostEqual(result["p_value"], 0.001, places=12)
        self.assertTrue(result["reject_null"])
        self.assertTrue(result["verification_matches"])
        self.assertTrue(result["sums_of_squares_match"])
        self.assertAlmostEqual(result["eta_squared"], 0.9)
        self.assertTrue(math.isclose(result["scipy_f"], 27.0))

    def test_input_parser_rejects_non_numeric_values(self):
        with self.assertRaisesRegex(ValueError, "not a valid number"):
            parse_values("1, nope, 3", "Group A")

    def test_two_identical_constant_groups_have_undefined_f(self):
        result = calculate_one_way_anova(
            {"A": [2, 2], "B": [2, 2]}, alpha=0.05
        )
        self.assertTrue(math.isnan(result["f_statistic"]))
        self.assertFalse(result["decision_defined"])


if __name__ == "__main__":
    unittest.main()
