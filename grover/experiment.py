"""Experiment sweeps: build circuits, run them on each platform, and store results."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from qiskit import QuantumCircuit
from qiskit.transpiler import generate_preset_pass_manager

from . import backends
from .circuits import grover_circuit, ideal_success_probability, optimal_iterations, sat_oracle
from .instances import get_instance
from .sat import CNF, assignment_to_bitstring
from .stats import wilson_interval

PLATFORMS = ("ideal", "noisy", "hardware")


@dataclass
class Record:
    instance: str
    num_qubits: int
    num_marked: int
    iterations: int
    k_opt: int
    platform: str  # one of PLATFORMS
    backend: str  # device the circuit was compiled for / run on
    shots: int
    successes: int
    success_prob: float
    ci_low: float
    ci_high: float
    theory_prob: float
    depth: int
    two_qubit_gates: int
    total_gates: int
    counts: Dict[str, int] = field(default_factory=dict)
    job_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))


@dataclass
class CompiledCircuit:
    cnf: CNF
    iterations: int
    circuit: QuantumCircuit  # ISA circuit for the target backend

    @property
    def depth(self) -> int:
        return self.circuit.depth()

    @property
    def two_qubit_gates(self) -> int:
        return sum(
            1 for inst in self.circuit.data
            if inst.operation.num_qubits == 2 and inst.operation.name not in ("barrier", "measure")
        )

    @property
    def total_gates(self) -> int:
        return sum(
            1 for inst in self.circuit.data if inst.operation.name not in ("barrier", "measure")
        )


def default_iterations(cnf: CNF, extra: int = 2) -> List[int]:
    """0 .. k*+extra, so the sweep shows both the rise and the over-rotation fall-off."""
    k_opt = optimal_iterations(2 ** cnf.num_vars, len(cnf.solutions()))
    return list(range(0, k_opt + extra + 1))


def compile_sweep(
    cnfs: Iterable[CNF],
    iterations: Optional[Sequence[int]],
    backend,
    optimization_level: int = 3,
    seed: int = 42,
    extra: int = 2,
) -> List[CompiledCircuit]:
    """Transpile every (instance, k) circuit to ``backend``'s native gates and layout.

    The same ISA circuit is executed on all three platforms, so depth and gate counts
    reported for the ideal and noisy runs are exactly those executed on hardware.
    """
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level, backend=backend, seed_transpiler=seed
    )
    out = []
    for cnf in cnfs:
        oracle = sat_oracle(cnf)
        ks = iterations if iterations is not None else default_iterations(cnf, extra)
        for k in ks:
            out.append(CompiledCircuit(cnf, k, pm.run(grover_circuit(oracle, k))))
    return out


def score(cnf: CNF, counts: Dict[str, int]) -> int:
    """Number of shots whose measured bitstring satisfies the formula."""
    marked = {assignment_to_bitstring(s) for s in cnf.solutions()}
    return sum(c for bits, c in counts.items() if bits in marked)


def make_record(cc: CompiledCircuit, platform: str, backend_name: str, counts: Dict[str, int],
                job_id: Optional[str] = None) -> Record:
    cnf = cc.cnf
    n, m = cnf.num_vars, len(cnf.solutions())
    shots = sum(counts.values())
    succ = score(cnf, counts)
    lo, hi = wilson_interval(succ, shots)
    return Record(
        instance=cnf.name,
        num_qubits=n,
        num_marked=m,
        iterations=cc.iterations,
        k_opt=optimal_iterations(2 ** n, m),
        platform=platform,
        backend=backend_name,
        shots=shots,
        successes=succ,
        success_prob=succ / shots if shots else 0.0,
        ci_low=lo,
        ci_high=hi,
        theory_prob=ideal_success_probability(2 ** n, m, cc.iterations),
        depth=cc.depth,
        two_qubit_gates=cc.two_qubit_gates,
        total_gates=cc.total_gates,
        counts=dict(counts),
        job_id=job_id,
    )


def run_simulations(
    compiled: List[CompiledCircuit],
    target,
    shots: int = 4096,
    platforms: Sequence[str] = ("ideal", "noisy"),
    seed: Optional[int] = 1234,
    log=print,
) -> List[Record]:
    sims = {}
    if "ideal" in platforms:
        sims["ideal"] = backends.ideal_simulator(seed)
    if "noisy" in platforms:
        sims["noisy"] = backends.noisy_simulator(target, seed)
    records = []
    for platform, sim in sims.items():
        circuits = [cc.circuit for cc in compiled]
        result = sim.run(circuits, shots=shots).result()
        for i, cc in enumerate(compiled):
            rec = make_record(cc, platform, target.name, result.get_counts(i))
            records.append(rec)
            log(f"  [{platform:5}] {rec.instance} k={rec.iterations}: "
                f"P={rec.success_prob:.3f} (theory {rec.theory_prob:.3f}, 2q gates {rec.two_qubit_gates})")
    return records


# --- Real hardware ---------------------------------------------------------------------

def submit_hardware(
    compiled: List[CompiledCircuit],
    backend,
    shots: int = 4096,
    dynamical_decoupling: bool = False,
    twirling: bool = False,
) -> dict:
    """Submit all circuits as one Sampler job (job mode works on the free Open plan).

    Returns a manifest to save to disk; the job can sit in the queue for hours, so
    results are collected later with :func:`fetch_hardware`.
    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    if dynamical_decoupling:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XpXm"
    if twirling:
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.enable_measure = True
    job = sampler.run([cc.circuit for cc in compiled])
    return {
        "job_id": job.job_id(),
        "backend": backend.name,
        "shots": shots,
        "submitted": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "options": {"dynamical_decoupling": dynamical_decoupling, "twirling": twirling},
        "circuits": [
            {"instance": cc.cnf.name, "iterations": cc.iterations, "depth": cc.depth,
             "two_qubit_gates": cc.two_qubit_gates, "total_gates": cc.total_gates}
            for cc in compiled
        ],
    }


def fetch_hardware(manifest: dict, service=None, log=print) -> Optional[List[Record]]:
    """Collect results for a submitted job; returns None if it has not finished yet."""
    service = service or backends.runtime_service()
    job = service.job(manifest["job_id"])
    status = job.status()
    status = getattr(status, "name", status)
    if status != "DONE":
        log(f"job {manifest['job_id']} status: {status}")
        return None
    result = job.result()
    records = []
    for entry, pub in zip(manifest["circuits"], result):
        counts = pub.data.meas.get_counts()
        cnf = get_instance(entry["instance"])
        cc = _StoredCircuit(cnf, entry)
        rec = make_record(cc, "hardware", manifest["backend"], counts, job_id=manifest["job_id"])
        records.append(rec)
        log(f"  [hw   ] {rec.instance} k={rec.iterations}: P={rec.success_prob:.3f} "
            f"(theory {rec.theory_prob:.3f})")
    return records


class _StoredCircuit:
    """Stand-in for CompiledCircuit when only the manifest's gate statistics are available."""

    def __init__(self, cnf: CNF, entry: dict):
        self.cnf = cnf
        self.iterations = entry["iterations"]
        self.depth = entry["depth"]
        self.two_qubit_gates = entry["two_qubit_gates"]
        self.total_gates = entry["total_gates"]


# --- Persistence -------------------------------------------------------------------------

def save_records(records: List[Record], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in records], indent=1))
    return path


def load_records(paths: Iterable[Path]) -> List[Record]:
    records = []
    for p in paths:
        records += [Record(**d) for d in json.loads(Path(p).read_text())]
    return records
