"""Boolean CNF formulas (DIMACS-style literals) and small 3-SAT instance generation."""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

Clause = Tuple[int, ...]
Assignment = Tuple[int, ...]  # (x1, x2, ..., xn) with values in {0, 1}


@dataclass(frozen=True)
class CNF:
    """A CNF formula over variables 1..num_vars.

    Each clause is a tuple of non-zero ints: ``k`` means x_k, ``-k`` means NOT x_k.
    """

    num_vars: int
    clauses: Tuple[Clause, ...]
    name: str = ""
    description: str = field(default="", compare=False)

    def __post_init__(self):
        for clause in self.clauses:
            for lit in clause:
                if lit == 0 or abs(lit) > self.num_vars:
                    raise ValueError(f"literal {lit} out of range for {self.num_vars} variables")

    def evaluate(self, assignment: Sequence[int]) -> bool:
        """Return True if ``assignment`` (x1..xn) satisfies every clause."""
        if len(assignment) != self.num_vars:
            raise ValueError("assignment length does not match num_vars")
        return all(
            any((assignment[abs(lit) - 1] == 1) == (lit > 0) for lit in clause)
            for clause in self.clauses
        )

    def solutions(self) -> List[Assignment]:
        """All satisfying assignments, found by classical enumeration.

        Used to build the benchmark oracle and to score measurement outcomes;
        see README "Honest caveats" for why this is fine for a benchmark.
        """
        return [
            a for a in itertools.product((0, 1), repeat=self.num_vars) if self.evaluate(a)
        ]

    def to_dimacs(self) -> str:
        lines = [f"p cnf {self.num_vars} {len(self.clauses)}"]
        lines += [" ".join(map(str, c)) + " 0" for c in self.clauses]
        return "\n".join(lines)

    def pretty(self) -> str:
        def lit(l: int) -> str:
            return f"x{abs(l)}" if l > 0 else f"¬x{abs(l)}"

        return " ∧ ".join("(" + " ∨ ".join(lit(l) for l in c) + ")" for c in self.clauses)


def assignment_to_bitstring(assignment: Sequence[int]) -> str:
    """Qiskit count key for an assignment: qubit i holds x_{i+1}, keys are little-endian."""
    return "".join(str(b) for b in reversed(assignment))


def bitstring_to_assignment(bits: str) -> Assignment:
    return tuple(int(b) for b in reversed(bits))


def planted_3sat(num_vars: int, num_solutions: int = 1, seed: int = 0, name: str = "") -> CNF:
    """Generate a random 3-SAT formula with exactly ``num_solutions`` satisfying assignments.

    Planted solutions are chosen first; random 3-literal clauses that keep all planted
    solutions satisfied are added until every other assignment is excluded. Clauses
    that exclude nothing new are skipped, so the formula stays small.
    """
    if num_vars < 3:
        raise ValueError("3-SAT needs at least 3 variables")
    rng = random.Random(seed)
    space = list(itertools.product((0, 1), repeat=num_vars))
    planted = rng.sample(space, num_solutions)
    alive = set(space)
    clauses: List[Clause] = []
    while len(alive) > num_solutions:
        vars_ = sorted(rng.sample(range(1, num_vars + 1), 3))
        clause = tuple(v if rng.random() < 0.5 else -v for v in vars_)
        cnf = CNF(num_vars, (clause,))
        if not all(cnf.evaluate(p) for p in planted):
            continue
        killed = {a for a in alive if not cnf.evaluate(a)}
        if killed:
            clauses.append(clause)
            alive -= killed
    return CNF(num_vars, tuple(clauses), name=name or f"3sat_n{num_vars}_m{num_solutions}_s{seed}")
