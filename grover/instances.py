"""Fixed benchmark instances.

Generated once with ``planted_3sat(n, M, seed)`` and frozen here so that results are
comparable across runs and library versions. ``tests/test_sat.py`` re-checks each
instance's solution count.
"""

from __future__ import annotations

from typing import Dict, List

from .sat import CNF

INSTANCES: Dict[str, CNF] = {
    "sat3_m1": CNF(
        3,
        ((1, -2, -3), (-1, -2, -3), (1, 2, 3), (-1, 2, 3), (1, 2, -3), (-1, 2, -3), (-1, -2, 3)),
        name="sat3_m1",
        description="3 variables, unique solution x=(0,1,0)",
    ),
    "sat4_m1": CNF(
        4,
        (
            (-1, -2, -4), (-1, -2, 3), (-2, -3, -4), (-2, 3, -4), (-2, -3, 4), (-1, 2, -4),
            (2, -3, -4), (-1, 3, 4), (1, 2, 4), (-2, 3, 4), (2, -3, 4),
        ),
        name="sat4_m1",
        description="4 variables, unique solution x=(0,0,0,1)",
    ),
    "sat5_m1": CNF(
        5,
        (
            (-2, 4, -5), (1, 2, -5), (3, -4, 5), (-1, -2, 5), (-3, 4, -5), (-1, 4, 5),
            (2, -3, 4), (3, -4, -5), (-1, -3, -5), (1, 3, 5), (-1, -4, 5), (1, -3, 5),
            (3, 4, -5),
        ),
        name="sat5_m1",
        description="5 variables, unique solution x=(0,1,1,1,1)",
    ),
    "sat4_m2": CNF(
        4,
        (
            (1, 2, 3), (-1, 3, -4), (-1, 2, -4), (1, 2, -3), (-1, -2, 3), (-1, -2, -4),
            (-1, -2, -3), (2, -3, 4), (-1, 3, 4), (-2, 3, -4), (1, -3, 4),
        ),
        name="sat4_m2",
        description="4 variables, two solutions",
    ),
    "sat5_m2": CNF(
        5,
        (
            (1, 2, -5), (1, -2, 4), (-2, -3, -5), (1, 2, 3), (3, 4, -5), (-1, -2, -5),
            (1, -3, 5), (-1, 3, -4), (-1, -2, -4), (-1, -2, 3), (-2, -3, 4), (1, -4, -5),
            (2, 4, -5), (-1, -3, -5), (2, -3, 4), (1, 3, -4),
        ),
        name="sat5_m2",
        description="5 variables, two solutions",
    ),
}

# The qubit-scaling series (one marked state each) used for the "success vs. qubits" plot.
SCALING_SERIES: List[str] = ["sat3_m1", "sat4_m1", "sat5_m1"]


def get_instance(name: str) -> CNF:
    try:
        return INSTANCES[name]
    except KeyError:
        raise KeyError(f"unknown instance {name!r}; choose from {sorted(INSTANCES)}") from None


def instances_for_qubits(qubits: List[int]) -> List[CNF]:
    """The unique-solution instance for each requested qubit count."""
    by_n = {INSTANCES[k].num_vars: INSTANCES[k] for k in SCALING_SERIES}
    missing = [n for n in qubits if n not in by_n]
    if missing:
        raise ValueError(f"no scaling instance for n={missing}; available: {sorted(by_n)}")
    return [by_n[n] for n in qubits]
