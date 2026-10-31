import math

import pytest

from grover.analysis import max_gates_for_advantage
from grover.stats import classical_expected_queries, quantum_expected_queries, wilson_interval


def test_wilson_contains_point_estimate():
    lo, hi = wilson_interval(400, 1000)
    assert lo < 0.4 < hi
    assert wilson_interval(0, 100)[0] == 0.0


def test_query_costs():
    assert classical_expected_queries(8, 1) == pytest.approx(4.5)
    assert quantum_expected_queries(0, 1 / 8) == pytest.approx(8)  # random guessing with replacement
    assert quantum_expected_queries(2, 0) == math.inf


def test_noiseless_grover_always_has_budget():
    assert max_gates_for_advantage(5, rate=0.0) == math.inf
    assert max_gates_for_advantage(5, rate=0.01) > 0
