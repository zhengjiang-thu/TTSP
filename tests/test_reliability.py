import math
import unittest

from TTSP.config import compute_fresh_count
from TTSP.reliability import (
    calculate_token_entropies,
    filter_traces,
    retained_trace_count,
    score_trace,
)


class ReliabilityTests(unittest.TestCase):
    def test_truncated_entropy_is_renormalized(self):
        entropies = calculate_token_entropies(
            [{0: math.log(0.75), 1: math.log(0.25)}],
            k_top=2,
        )
        expected = -(0.75 * math.log(0.75) + 0.25 * math.log(0.25))
        self.assertAlmostEqual(entropies[0], expected)

    def test_missing_logprobs_are_not_treated_as_confident(self):
        trace = {"token_entropies": [None, None]}
        self.assertEqual(score_trace(trace), -math.inf)

    def test_gating_keeps_exact_count_with_ties(self):
        traces = [
            {"trace_id": index, "token_entropies": [0.5]}
            for index in range(8)
        ]
        retained, _ = filter_traces(traces, filtering_ratio=0.4)
        self.assertEqual(retained_trace_count(8, 0.4), 5)
        self.assertEqual([trace["trace_id"] for trace in retained], list(range(5)))

    def test_first_round_is_all_fresh_and_later_round_uses_ceil(self):
        self.assertEqual(compute_fresh_count(0, 8, 0.4, False), 8)
        self.assertEqual(compute_fresh_count(1, 8, 0.4, True), 4)
        self.assertEqual(compute_fresh_count(2, 8, 0.4, False), 4)

    def test_gate_can_disable_every_trace_at_rho_one(self):
        traces = [{"trace_id": 0, "token_entropies": [0.5]}]
        retained, threshold = filter_traces(traces, filtering_ratio=1.0)
        self.assertEqual(retained, [])
        self.assertEqual(threshold, math.inf)


if __name__ == "__main__":
    unittest.main()
