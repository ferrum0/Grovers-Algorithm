import itertools

import pytest

from grover.instances import INSTANCES
from grover.sat import CNF, assignment_to_bitstring, bitstring_to_assignment, planted_3sat


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_instance_solution_counts(name):
    cnf = INSTANCES[name]
    expected_m = int(name.split("_m")[1])
    assert len(cnf.solutions()) == expected_m
    assert all(len(c) == 3 for c in cnf.clauses)


def test_evaluate_and_negation():
    cnf = CNF(3, ((1, -2, 3),))
    assert cnf.evaluate((0, 0, 0))  # ¬x2 true
    assert not cnf.evaluate((0, 1, 0))


def test_bitstring_roundtrip_is_little_endian():
    assert assignment_to_bitstring((1, 0, 0)) == "001"  # x1 lives on qubit 0, rightmost
    for a in itertools.product((0, 1), repeat=4):
        assert bitstring_to_assignment(assignment_to_bitstring(a)) == a


@pytest.mark.parametrize("n,m", [(3, 1), (4, 2), (6, 1)])
def test_planted_generator(n, m):
    assert len(planted_3sat(n, m, seed=7).solutions()) == m
