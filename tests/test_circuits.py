import math

import pytest
from qiskit.quantum_info import Statevector

from grover.circuits import (
    best_iterations, grover_circuit, ideal_success_probability, optimal_iterations,
    phase_oracle_from_solutions, sat_oracle,
)
from grover.instances import INSTANCES
from grover.sat import assignment_to_bitstring


def test_optimal_iterations_matches_spec_formula():
    assert optimal_iterations(8, 1) == 2
    assert optimal_iterations(16, 1) == 3
    assert optimal_iterations(32, 1) == 4
    assert optimal_iterations(16, 2) == 2
    for n in range(2, 10):
        assert optimal_iterations(2 ** n, 1) == math.floor(math.pi / 4 * math.sqrt(2 ** n))


def test_best_iterations_is_argmax():
    for N, M in [(8, 1), (16, 1), (32, 2), (64, 3)]:
        k = best_iterations(N, M)
        assert all(ideal_success_probability(N, M, k) >= ideal_success_probability(N, M, j) - 1e-12
                   for j in range(0, optimal_iterations(N, M) + 3))  # first lobe only; sin² is periodic


def test_oracle_flips_only_marked_phases():
    sols = [(1, 0, 1), (0, 1, 1)]
    op = Statevector.from_label("+++").evolve(phase_oracle_from_solutions(3, sols))
    marked = {int(assignment_to_bitstring(s), 2) for s in sols}
    for i, amp in enumerate(op.data):
        assert amp.real == pytest.approx((-1 if i in marked else 1) / math.sqrt(8))


@pytest.mark.parametrize("name", sorted(INSTANCES))
def test_statevector_matches_theory(name):
    cnf = INSTANCES[name]
    N, M = 2 ** cnf.num_vars, len(cnf.solutions())
    marked = {assignment_to_bitstring(s) for s in cnf.solutions()}
    for k in range(0, optimal_iterations(N, M) + 2):
        probs = Statevector(grover_circuit(sat_oracle(cnf), k, measure=False)).probabilities_dict()
        p = sum(v for b, v in probs.items() if b in marked)
        assert p == pytest.approx(ideal_success_probability(N, M, k), abs=1e-9)
