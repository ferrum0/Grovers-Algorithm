"""Statistics for success probabilities and query-cost comparisons."""

from __future__ import annotations

import math
from typing import Tuple


def wilson_interval(successes: int, shots: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (well-behaved near 0 and 1)."""
    if shots <= 0:
        return (0.0, 1.0)
    p = successes / shots
    denom = 1 + z * z / shots
    centre = (p + z * z / (2 * shots)) / denom
    half = z * math.sqrt(p * (1 - p) / shots + z * z / (4 * shots * shots)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def classical_expected_queries(num_states: int, num_marked: int) -> float:
    """Expected oracle queries for classical brute force (sampling without replacement).

    Checking items in random order until a marked one is found takes (N+1)/(M+1)
    queries on average.
    """
    return (num_states + 1) / (num_marked + 1)


def quantum_expected_queries(iterations: int, success_prob: float) -> float:
    """Expected oracle queries for repeat-until-success Grover.

    Each attempt spends ``iterations`` coherent oracle calls plus one classical
    evaluation of f to verify the measured candidate; attempts repeat until the
    verification passes, so the count is geometric with mean (k+1)/p.
    """
    if success_prob <= 0:
        return math.inf
    return (iterations + 1) / success_prob
