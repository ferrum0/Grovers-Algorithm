"""Grover's algorithm on real hardware: benchmarking CLI.

    python main.py instances
    python main.py simulate --backend fake_torino
    python main.py hardware submit --backend least_busy --qubits 3 4
    python main.py hardware fetch results/hw_job_<id>.json
    python main.py analyze results/*.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

from grover import backends
from grover.circuits import best_iterations, grover_circuit, ideal_success_probability, optimal_iterations, sat_oracle
from grover.experiment import (
    compile_sweep, fetch_hardware, load_records, run_simulations, save_records, submit_hardware,
)
from grover.instances import INSTANCES, get_instance, instances_for_qubits

RESULTS = Path("results")
FIGURES = Path("figures")


def _select_instances(args):
    if args.instances:
        return [get_instance(name) for name in args.instances]
    return instances_for_qubits(args.qubits)


def _iterations(args):
    if args.iterations:
        return args.iterations
    return None  # 0..k*+extra per instance


def cmd_instances(args):
    for name, cnf in INSTANCES.items():
        sols = cnf.solutions()
        N, M = 2 ** cnf.num_vars, len(sols)
        k = optimal_iterations(N, M)
        print(f"{name}: n={cnf.num_vars} clauses={len(cnf.clauses)} M={M} N={N}")
        print(f"  formula : {cnf.pretty()}")
        print(f"  solutions: {sols}")
        print(f"  k* = floor(pi/4*sqrt(N/M)) = {k}  ->  P_ideal = {ideal_success_probability(N, M, k):.4f}"
              f"  (best integer k = {best_iterations(N, M)})")
        if args.draw:
            print(grover_circuit(sat_oracle(cnf), 1).decompose().draw(output="text", fold=120))
        print()


def cmd_simulate(args):
    target = backends.target_backend(args.backend)
    cnfs = _select_instances(args)
    print(f"Compiling {len(cnfs)} instance(s) for {target.name} ...")
    compiled = compile_sweep(cnfs, _iterations(args), target, args.opt_level, args.seed, args.extra)
    platforms = [p for p in ("ideal", "noisy") if p not in args.skip]
    records = run_simulations(compiled, target, args.shots, platforms, seed=args.seed)
    out = save_records(records, RESULTS / f"sim_{target.name}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    print(f"Saved {len(records)} records to {out}")


def cmd_hw_submit(args):
    cnfs = _select_instances(args)
    target = backends.ibm_backend(args.backend, min_qubits=max(c.num_vars for c in cnfs))
    print(f"Compiling {len(cnfs)} instance(s) for {target.name} (live calibration) ...")
    compiled = compile_sweep(cnfs, _iterations(args), target, args.opt_level, args.seed, args.extra)
    for cc in compiled:
        print(f"  {cc.cnf.name} k={cc.iterations}: depth {cc.depth}, 2q gates {cc.two_qubit_gates}")
    if args.dry_run:
        print("Dry run: nothing submitted.")
        return
    if args.with_noisy_sim:
        # Snapshot the noise model at submission time so the comparison is like-for-like.
        recs = run_simulations(compiled, target, args.shots, ("ideal", "noisy"), seed=args.seed)
        save_records(recs, RESULTS / f"sim_{target.name}_{time.strftime('%Y%m%d-%H%M%S')}.json")
    manifest = submit_hardware(compiled, target, args.shots, args.dd, args.twirling)
    path = RESULTS / f"hw_job_{manifest['job_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1))
    print(f"Submitted job {manifest['job_id']} to {target.name}; manifest saved to {path}")
    print(f"Collect results later with: python main.py hardware fetch {path}")


def cmd_hw_fetch(args):
    service = backends.runtime_service()
    for path in args.manifests:
        manifest = json.loads(Path(path).read_text())
        records = fetch_hardware(manifest, service)
        if records is None:
            continue
        out = save_records(records, RESULTS / f"hw_{manifest['backend']}_{manifest['job_id']}.json")
        print(f"Saved {len(records)} hardware records to {out}")


def cmd_analyze(args):
    from grover import analysis
    from grover.sat import planted_3sat

    paths = [p for pattern in args.results for p in glob.glob(pattern)
             if not Path(p).name.startswith("hw_job_")]
    if not paths:
        sys.exit("no result files matched")
    records = load_records(paths)
    print(f"Loaded {len(records)} records from {len(paths)} file(s)")
    FIGURES.mkdir(exist_ok=True)

    fits = {p: f for p in ("noisy", "hardware") if (f := analysis.fit_decay(records, p))}

    # Gate counts of the k* circuit for each n, measured where available and compiled otherwise
    # (compiling is cheap even when running is not), to extrapolate the break-even point.
    gate_counts = {r.num_qubits: r.two_qubit_gates for r in records
                   if r.num_marked == 1 and r.iterations == r.k_opt}
    missing = [n for n in range(3, args.max_n + 1) if n not in gate_counts]
    if missing:
        target = backends.target_backend(args.backend)
        print(f"Compiling k* circuits for n={missing} on {target.name} for extrapolation ...")
        for n in missing:
            cnf = planted_3sat(n, 1, seed=n)
            cc = compile_sweep([cnf], [optimal_iterations(2 ** n, 1)], target, seed=args.seed)[0]
            gate_counts[n] = cc.two_qubit_gates

    print("Wrote", analysis.plot_success_vs_iterations(records, FIGURES / "success_vs_iterations.png"))
    print("Wrote", analysis.plot_success_vs_qubits(records, FIGURES / "success_vs_qubits.png"))
    print("Wrote", analysis.plot_break_even(records, fits, gate_counts, FIGURES / "break_even.png"))
    report = analysis.break_even_report(records, fits, gate_counts)
    (FIGURES / "report.md").write_text(report)
    print("Wrote", FIGURES / "report.md")
    for p, fit in fits.items():
        print(f"  {p}: effective error {fit.per_gate_error:.2%} per two-qubit gate")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("instances", help="list benchmark SAT instances and k*")
    p.add_argument("--draw", action="store_true", help="also draw the k=1 circuit")
    p.set_defaults(func=cmd_instances)

    def sweep_args(p, default_backend, extra):
        p.add_argument("--backend", default=default_backend)
        p.add_argument("--qubits", type=int, nargs="+", default=[3, 4, 5])
        p.add_argument("--instances", nargs="+", help="instance names (overrides --qubits)")
        p.add_argument("--iterations", type=int, nargs="+", help="explicit k values")
        p.add_argument("--extra", type=int, default=extra, help="sweep k = 0..k*+extra")
        p.add_argument("--shots", type=int, default=4096)
        p.add_argument("--opt-level", type=int, default=3)
        p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("simulate", help="ideal + noise-model simulation")
    sweep_args(p, "fake_torino", extra=2)
    p.add_argument("--skip", nargs="*", default=[], choices=["ideal", "noisy"])
    p.set_defaults(func=cmd_simulate)

    hw = sub.add_parser("hardware", help="IBM Quantum hardware runs").add_subparsers(dest="hwcmd", required=True)
    p = hw.add_parser("submit", help="compile and submit a job")
    sweep_args(p, "least_busy", extra=1)
    p.add_argument("--dd", action="store_true", help="enable dynamical decoupling")
    p.add_argument("--twirling", action="store_true", help="enable Pauli twirling")
    p.add_argument("--with-noisy-sim", action="store_true",
                   help="also run ideal + noise-model sims from today's calibration")
    p.add_argument("--dry-run", action="store_true", help="compile only, print circuit costs")
    p.set_defaults(func=cmd_hw_submit)
    p = hw.add_parser("fetch", help="collect results of submitted jobs")
    p.add_argument("manifests", nargs="+")
    p.set_defaults(func=cmd_hw_fetch)

    p = sub.add_parser("analyze", help="plots + break-even report")
    p.add_argument("results", nargs="*", default=["results/*.json"])
    p.add_argument("--backend", default="fake_torino", help="device used to compile extrapolation circuits")
    p.add_argument("--max-n", type=int, default=8, help="extrapolate break-even up to this many qubits")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_analyze)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
