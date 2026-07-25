"""Grover oracle, diffusion operator, full circuits and closed-form theory."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate

from .sat import CNF


def _mcz(qc: QuantumCircuit, qubits: Sequence[int]) -> None:
    """Multi-controlled Z on ``qubits`` (symmetric, so control/target choice is arbitrary)."""
    qubits = list(qubits)
    if len(qubits) == 1:
        qc.z(qubits[0])
    elif len(qubits) == 2:
        qc.cz(qubits[0], qubits[1])
    else:
        qc.append(ZGate().control(len(qubits) - 1), qubits)


def phase_oracle_from_solutions(num_qubits: int, solutions: Iterable[Sequence[int]]) -> QuantumCircuit:
    """U_f|x> = (-1)^f(x)|x>, built as one X-conjugated MCZ per marked bit pattern.

    Qubit i holds variable x_{i+1}. Zeros in the pattern are flipped with X so the
    MCZ fires exactly on that basis state.
    """
    qc = QuantumCircuit(num_qubits, name="Oracle")
    for sol in solutions:
        if len(sol) != num_qubits:
            raise ValueError("solution length does not match qubit count")
        zeros = [i for i, b in enumerate(sol) if b == 0]
        if zeros:
            qc.x(zeros)
        _mcz(qc, range(num_qubits))
        if zeros:
            qc.x(zeros)
    return qc


def sat_oracle(cnf: CNF) -> QuantumCircuit:
    return phase_oracle_from_solutions(cnf.num_vars, cnf.solutions())


def diffuser(num_qubits: int) -> QuantumCircuit:
    """D = 2|s><s| - I, implemented as H X (MCZ) X H (up to a global phase of -1)."""
    qc = QuantumCircuit(num_qubits, name="Diffuser")
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))
    _mcz(qc, range(num_qubits))
    qc.x(range(num_qubits))
    qc.h(range(num_qubits))
    return qc


def grover_circuit(oracle: QuantumCircuit, iterations: int, measure: bool = True) -> QuantumCircuit:
    n = oracle.num_qubits
    qc = QuantumCircuit(n, name=f"Grover_k{iterations}")
    qc.h(range(n))
    diff = diffuser(n)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diff, inplace=True)
    if measure:
        qc.measure_all()
    return qc


def optimal_iterations(num_states: int, num_marked: int) -> int:
    """k* = floor(pi/4 * sqrt(N/M)), the iteration count from the project spec."""
    if not 0 < num_marked <= num_states:
        raise ValueError("need 0 < M <= N")
    return math.floor(math.pi / 4 * math.sqrt(num_states / num_marked))


def grover_angle(num_states: int, num_marked: int) -> float:
    """theta with sin(theta) = sqrt(M/N); each iteration rotates the state by 2*theta."""
    return math.asin(math.sqrt(num_marked / num_states))


def ideal_success_probability(num_states: int, num_marked: int, iterations: int) -> float:
    """P_k = sin^2((2k+1) theta), the exact noiseless success probability."""
    theta = grover_angle(num_states, num_marked)
    return math.sin((2 * iterations + 1) * theta) ** 2


def best_iterations(num_states: int, num_marked: int) -> int:
    """Integer k that maximises sin^2((2k+1) theta); usually equals k*."""
    theta = grover_angle(num_states, num_marked)
    lo = max(0, math.floor(math.pi / (4 * theta) - 0.5))
    return max((lo, lo + 1), key=lambda k: ideal_success_probability(num_states, num_marked, k))
