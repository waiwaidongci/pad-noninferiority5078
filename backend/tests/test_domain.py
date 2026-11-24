"""Exactness checks for the calibration-batch withdrawal analysis.

Run from the backend directory with ``python -I -m unittest discover -s tests``.
The domain module only uses the standard library, so these tests need no
database or third-party package.
"""

import itertools
import math
import unittest
from decimal import Decimal
from statistics import mean, stdev

from src.services import domain


def _pair(index, difference, baseline=8.0):
    return {
        "protocol_id": domain.PROTOCOL_ID,
        "pair_id": f"PAD-{index:03d}",
        "station_id": f"HX-{index:02d}",
        "window_start": "2026-11-12 08:00",
        "baseline_loss_kw": baseline,
        "trial_loss_kw": baseline + difference,
    }


def _correspondence(count, per_batch):
    return [
        {"station_id": f"HX-{index:02d}", "calibration_batch": f"CAL-{index // per_batch + 1:02d}"}
        for index in range(count)
    ]


def _datasets(differences, per_batch):
    rows = [_pair(index, value) for index, value in enumerate(differences)]
    return rows, _correspondence(len(differences), per_batch)


def _reference_upper(values):
    if len(values) < 2:
        return None
    spread = stdev(values)
    if spread == 0:
        return None
    return mean(values) + domain._critical_t(len(values) - 1) * spread / math.sqrt(len(values))


class CriticalTTest(unittest.TestCase):
    def test_matches_reference_table(self):
        # Two-sided/two-sided-adjacent published one-sided 95% quantiles.
        expected = {1: 6.313751515, 2: 2.919985580, 5: 2.015048373, 11: 1.795884819, 35: 1.689572458}
        for df, value in expected.items():
            self.assertAlmostEqual(domain._critical_t(df), value, places=8)

    def test_monotone_in_degrees_of_freedom(self):
        values = [domain._critical_t(df) for df in range(1, 600)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_does_not_stall_on_float_plateau(self):
        # At df=1140 the tail function flattens into a ~1e-15 plateau. A
        # solution taken from a stale bisection midpoint lands near 2.47;
        # the true quantile is ~1.64619 (verified with 50-digit Decimal).
        self.assertAlmostEqual(domain._critical_t(1140), 1.6461913600, places=9)


class CorrespondenceTest(unittest.TestCase):
    def test_missing_station_is_rejected(self):
        rows, batches = _datasets([0.0] * 12, 3)
        with self.assertRaisesRegex(ValueError, "少配"):
            domain.compute(rows, {"min_pairs": 12}, batches[:-1])

    def test_extra_station_is_rejected(self):
        rows, batches = _datasets([0.0] * 12, 3)
        batches = batches + [{"station_id": "HX-99", "calibration_batch": "CAL-05"}]
        with self.assertRaisesRegex(ValueError, "多配"):
            domain.compute(rows, {"min_pairs": 12}, batches)

    def test_duplicate_station_is_rejected(self):
        rows, batches = _datasets([0.0] * 12, 3)
        batches[1] = dict(batches[0])
        with self.assertRaisesRegex(ValueError, "重复"):
            domain.compute(rows, {"min_pairs": 12}, batches)

    def test_station_case_is_normalized(self):
        rows, batches = _datasets([0.05, 0.1, 0.0, 0.2, 0.05, 0.1, 0.0, 0.15, 0.05, 0.1, 0.0, 0.2], 3)
        batches = [{"station_id": row["station_id"].lower(), "calibration_batch": row["calibration_batch"].lower()}
                   for row in batches]
        result = domain.compute(rows, {"min_pairs": 12}, batches)
        self.assertEqual(result["metrics"]["有效配对数"], 12)


class WithdrawalTest(unittest.TestCase):
    def test_base_upper_is_unchanged(self):
        rows = domain.sample_rows()
        differences = [float(Decimal(str(row["trial_loss_kw"])) - Decimal(str(row["baseline_loss_kw"])))
                       for row in rows]
        result = domain.compute(rows, {"min_pairs": 12}, domain.sample_batch_rows())
        expected = 0.0675 + domain._critical_t(35) * stdev(differences) / math.sqrt(36)
        self.assertEqual(result["metrics"]["单侧95%置信上界（kW）"], expected)
        self.assertAlmostEqual(expected, 0.14928707, places=8)

    def test_highest_bound_matches_brute_force(self):
        differences = [(-18, 12, 48, -33, 25, 7, -42, 31, -12, 58, -5, 23,
                        9, -27, 40, -16, 35, -8, 14, 51, -36, 17, -22, 4)]
        differences = [value / 100 for value in differences[0]]
        rows, batches = _datasets(differences, 3)
        result = domain.compute(rows, {"min_pairs": 12}, batches)
        groups = {f"CAL-{index // 3 + 1:02d}": differences[index:index + 3]
                  for index in range(0, len(differences), 3)}
        best = None
        best_combo = None
        for size in range(5):
            for combo in itertools.combinations(sorted(groups), size):
                kept = [value for label in groups if label not in combo for value in groups[label]]
                upper = _reference_upper(kept)
                if upper is not None and (best is None or upper > best):
                    best, best_combo = upper, combo
        self.assertAlmostEqual(result["metrics"]["最高单侧95%置信上界（kW）"], best, places=15)
        self.assertEqual(result["metrics"]["最高上界对应撤回批次"], "、".join(best_combo) or "不撤回")

    def test_exact_subset_count(self):
        rows, batches = _datasets([0.01 * index for index in range(36)], 6)
        result = domain.compute(rows, {"min_pairs": 12}, batches)
        self.assertEqual(result["metrics"]["撤回组合总数"], 1 + 6 + 15 + 20 + 15)

    def test_sixty_batches_enumerates_every_subset(self):
        differences = [0.01 * ((index * 37) % 23) - 0.1 for index in range(120)]
        rows, batches = _datasets(differences, 2)
        result = domain.compute(rows, {"min_pairs": 12}, batches)
        self.assertEqual(result["metrics"]["撤回组合总数"], 1 + 60 + 1770 + 34220 + 487635)

    def test_insufficient_stations_is_flagged(self):
        rows, batches = _datasets([0.01 * index for index in range(18)], 3)
        result = domain.compute(rows, {"min_pairs": 12}, batches)
        self.assertFalse(result["can_publish"])
        self.assertGreater(result["withdrawal"]["flagged_by_reason"][domain.REASON_STATIONS], 0)

    def test_zero_spread_cannot_compute(self):
        rows, batches = _datasets([0.1] * 12, 3)
        result = domain.compute(rows, {"min_pairs": 12}, batches)
        self.assertFalse(result["can_publish"])
        self.assertIsNone(result["metrics"].get("单侧95%置信上界（kW）"))
        self.assertGreater(result["withdrawal"]["flagged_by_reason"][domain.REASON_BOUND], 0)

    def test_margin_failure_is_flagged(self):
        differences = [0.6] * 6 + [0.55, 0.5, 0.6, 0.55, 0.5, 0.6] + [0.4] * 6
        rows, batches = _datasets(differences, 6)
        result = domain.compute(rows, {"min_pairs": 12}, batches)
        self.assertFalse(result["can_publish"])
        self.assertGreater(result["withdrawal"]["flagged_by_reason"][domain.REASON_MARGIN], 0)

    def test_flagged_list_is_worst_first(self):
        differences = [0.1] * 12 + [0.45] * 3 + [0.2] * 3
        rows, batches = _datasets(differences, 3)
        result = domain.compute(rows, {"min_pairs": 12}, batches)
        flagged = result["withdrawal"]["flagged"]
        self.assertLessEqual(len(flagged), domain.MAX_FLAGGED_COMBINATIONS)
        keys = [(item["upper"] is not None, -(item["upper"] or 0.0)) for item in flagged]
        self.assertEqual(keys, sorted(keys))

    def test_too_many_batches_is_rejected(self):
        differences = [0.01 * index for index in range(101)]
        rows, batches = _datasets(differences, 1)
        with self.assertRaisesRegex(ValueError, "超过"):
            domain.compute(rows, {"min_pairs": 12}, batches)


if __name__ == "__main__":
    unittest.main()
