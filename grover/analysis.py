"""Plots and the empirical break-even analysis."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .circuits import grover_angle, ideal_success_probability, optimal_iterations  # noqa: E402
from .experiment import Record  # noqa: E402
from .stats import classical_expected_queries, quantum_expected_queries  # noqa: E402

# Fixed categorical order: a platform keeps its colour and marker in every figure.
STYLE = {
    "ideal": {"color": "#2a78d6", "marker": "o", "label": "Ideal simulator"},
    "noisy": {"color": "#eb6834", "marker": "s", "label": "Noise-model simulator"},
    "hardware": {"color": "#1baf7a", "marker": "^", "label": "IBM hardware"},
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
REFERENCE = "#8a8984"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "font.size": 10,
})


def _group(records: Sequence[Record]) -> Dict[Tuple[str, str], List[Record]]:
    g: Dict[Tuple[str, str], List[Record]] = defaultdict(list)
    for r in records:
        g[(r.instance, r.platform)].append(r)
    for v in g.values():
        v.sort(key=lambda r: r.iterations)
    return g


def _platforms(records: Sequence[Record]) -> List[str]:
    present = {r.platform for r in records}
    return [p for p in STYLE if p in present]


def _instances(records: Sequence[Record]) -> List[Record]:
    """One representative record per instance, ordered by (n, M)."""
    seen = {}
    for r in records:
        seen.setdefault(r.instance, r)
    return sorted(seen.values(), key=lambda r: (r.num_qubits, r.num_marked))


def _errorbar(ax, recs: List[Record], x, platform: str, **kw):
    st = STYLE[platform]
    y = np.array([r.success_prob for r in recs])
    yerr = np.array([[r.success_prob - r.ci_low for r in recs], [r.ci_high - r.success_prob for r in recs]])
    ax.errorbar(x, y, yerr=yerr, color=st["color"], marker=st["marker"], label=st["label"],
                capsize=3, markeredgecolor="#fcfcfb", markeredgewidth=1.5, **kw)


# --- Figure 1 --------------------------------------------------------------------------

def plot_success_vs_iterations(records: Sequence[Record], out: Path) -> Path:
    groups = _group(records)
    reps = _instances(records)
    cols = min(3, len(reps))
    rows = math.ceil(len(reps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.8 * rows), squeeze=False)
    for ax, rep in zip(axes.flat, reps):
        N, M = 2 ** rep.num_qubits, rep.num_marked
        kmax = max(r.iterations for r in records if r.instance == rep.instance)
        ks = np.linspace(0, kmax, 300)
        theta = grover_angle(N, M)
        ax.plot(ks, np.sin((2 * ks + 1) * theta) ** 2, color=REFERENCE, lw=1.2, ls="--",
                label="Theory sin²((2k+1)θ)")
        ax.axhline(M / N, color=REFERENCE, lw=1, ls=":", label="Random guess M/N")
        ax.axvline(rep.k_opt, color=GRID, lw=6, zorder=0)
        ax.text(rep.k_opt, 1.04, f"k*={rep.k_opt}", ha="center", color=INK_2, fontsize=9)
        for p in _platforms(records):
            recs = groups.get((rep.instance, p))
            if recs:
                _errorbar(ax, recs, [r.iterations for r in recs], p)
        ax.set_title(f"{rep.instance}  (n={rep.num_qubits}, M={M})", fontsize=11)
        ax.set_xlabel("Grover iterations k")
        ax.set_ylabel("Success probability")
        ax.set_ylim(-0.02, 1.1)
        ax.set_xticks(range(0, kmax + 1))
    for ax in axes.flat[len(reps):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --- Figure 2 --------------------------------------------------------------------------

def _at_kopt(records: Sequence[Record], m: int = 1) -> Dict[str, List[Record]]:
    out: Dict[str, List[Record]] = defaultdict(list)
    for r in records:
        if r.num_marked == m and r.iterations == r.k_opt:
            out[r.platform].append(r)
    for v in out.values():
        v.sort(key=lambda r: r.num_qubits)
    return out


def plot_success_vs_qubits(records: Sequence[Record], out: Path) -> Path:
    at = _at_kopt(records)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.9))
    ns = sorted({r.num_qubits for v in at.values() for r in v})
    if ns:
        grid = np.arange(min(ns), max(ns) + 1)
        ax.plot(grid, [ideal_success_probability(2 ** n, 1, optimal_iterations(2 ** n, 1)) for n in grid],
                color=REFERENCE, ls="--", lw=1.2, label="Theory at k*")
        ax.plot(grid, 1 / 2.0 ** grid, color=REFERENCE, ls=":", lw=1, label="Random guess 1/N")
    for p in _platforms(records):
        if at.get(p):
            _errorbar(ax, at[p], [r.num_qubits for r in at[p]], p)
    ax.set_title("Success probability at k* (M=1)")
    ax.set_xlabel("Qubits n")
    ax.set_ylabel("Success probability")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(ns)
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.92), fontsize=8.5)

    # Circuit cost is a property of the compiled circuit, identical on every platform.
    cost = {r.num_qubits: r.two_qubit_gates for v in at.values() for r in v}
    xs = sorted(cost)
    ax2.bar(xs, [cost[n] for n in xs], width=0.5, color="#2a78d6")
    for n in xs:
        ax2.text(n, cost[n], f"{cost[n]}", ha="center", va="bottom", color=INK, fontsize=9)
    ax2.set_title("Two-qubit gates in the compiled k* circuit")
    ax2.set_xlabel("Qubits n")
    ax2.set_ylabel("Two-qubit gate count")
    ax2.set_xticks(xs)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --- Break-even analysis ---------------------------------------------------------------

@dataclass
class DecayFit:
    platform: str
    rate: float  # lambda in F = exp(-lambda * G2q)
    points: List[Tuple[int, float]]  # (two-qubit gates, effective fidelity)

    @property
    def per_gate_error(self) -> float:
        """Effective error per two-qubit gate (also absorbs 1q, readout and idle errors)."""
        return 1 - math.exp(-self.rate)

    def fidelity(self, gates: float) -> float:
        return math.exp(-self.rate * gates)


def effective_fidelity(r: Record) -> Optional[float]:
    """F such that p_obs = F * p_ideal + (1 - F) * M/N (global depolarising model)."""
    base = r.num_marked / 2 ** r.num_qubits
    gap = r.theory_prob - base
    if gap < 0.3 or r.two_qubit_gates == 0:  # too little signal to estimate F
        return None
    return (r.success_prob - base) / gap


def fit_decay(records: Sequence[Record], platform: str) -> Optional[DecayFit]:
    """Least-squares fit of ln F = -lambda * G through the origin."""
    pts = []
    for r in records:
        if r.platform != platform:
            continue
        f = effective_fidelity(r)
        if f is not None and f > 0.02:
            pts.append((r.two_qubit_gates, min(f, 1.0)))
    if len(pts) < 2:
        return None
    g = np.array([p[0] for p in pts], float)
    lnf = np.log([p[1] for p in pts])
    rate = float(-(g @ lnf) / (g @ g))
    return DecayFit(platform, max(rate, 0.0), pts)


def max_gates_for_advantage(n: int, m: int = 1, k: Optional[int] = None, rate: float = 0.0) -> float:
    """Largest two-qubit gate count at which Grover at k still beats classical brute force."""
    N = 2 ** n
    k = optimal_iterations(N, m) if k is None else k
    p_needed = (k + 1) / classical_expected_queries(N, m)
    base, p_ideal = m / N, ideal_success_probability(N, m, k)
    if p_needed >= p_ideal:
        return 0.0
    f_min = (p_needed - base) / (p_ideal - base)
    if f_min <= 0:
        return math.inf
    return -math.log(f_min) / rate if rate > 0 else math.inf


def query_advantage(r: Record) -> float:
    """Classical expected queries / quantum expected queries (>1 means Grover wins)."""
    eq = quantum_expected_queries(r.iterations, r.success_prob)
    return classical_expected_queries(2 ** r.num_qubits, r.num_marked) / eq


def plot_break_even(records: Sequence[Record], fits: Dict[str, DecayFit],
                    gate_counts: Dict[int, int], out: Path) -> Path:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4))

    # Left: effective fidelity vs two-qubit gates, with the fitted exponential decay.
    gmax = 1.3 * max([g for f in fits.values() for g, _ in f.points] + [10])
    gs = np.linspace(0, gmax, 200)
    for p, fit in fits.items():
        st = STYLE[p]
        ax.plot(gs, np.exp(-fit.rate * gs), color=st["color"], lw=1.5, ls="--")
        ax.plot(*zip(*fit.points), ls="none", marker=st["marker"], color=st["color"],
                markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                label=f"{st['label']}: ε≈{fit.per_gate_error:.2%}/2q gate")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 1.5)
    ax.set_title("Effective circuit fidelity")
    ax.set_xlabel("Two-qubit gates")
    ax.set_ylabel("F  (p = F·p_ideal + (1−F)·M/N)")
    ax.legend(fontsize=8.5, loc="lower left")

    # Right: query advantage at k* vs n -- measured and extrapolated with the fitted model.
    ax2.axhline(1, color=INK_2, lw=1.2)
    ax2.text(max(gate_counts or [3]), 1.08, "break-even", color=INK_2, fontsize=9, va="bottom", ha="right")
    ns = sorted(gate_counts)
    for p in _platforms(records):
        st = STYLE[p]
        meas = [r for r in _at_kopt(records)[p]] if _at_kopt(records).get(p) else []
        if meas:
            ax2.plot([r.num_qubits for r in meas], [max(query_advantage(r), 1e-3) for r in meas],
                     ls="none", marker=st["marker"], color=st["color"], markeredgecolor="#fcfcfb",
                     markeredgewidth=1.5, markersize=8, label=f"{st['label']} (measured)")
        if p == "ideal":
            model = [_predicted_advantage(n, gate_counts[n], 0.0) for n in ns]
        elif p in fits:
            model = [_predicted_advantage(n, gate_counts[n], fits[p].rate) for n in ns]
        else:
            continue
        ax2.plot(ns, model, color=st["color"], lw=1.5, ls="--")
    ax2.set_yscale("log")
    ax2.set_title("Query advantage at k* (M=1)")
    ax2.set_xlabel("Qubits n")
    ax2.set_ylabel("Classical / quantum expected queries")
    ax2.set_xticks(ns)
    ax2.legend(fontsize=8.5, loc="lower left")
    fig.text(0.99, 0.01, "dashed: model extrapolation", ha="right", color=INK_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _predicted_advantage(n: int, gates: int, rate: float, m: int = 1) -> float:
    N = 2 ** n
    k = optimal_iterations(N, m)
    base, p_ideal = m / N, ideal_success_probability(N, m, k)
    p = base + math.exp(-rate * gates) * (p_ideal - base)
    return classical_expected_queries(N, m) / quantum_expected_queries(k, p)


def break_even_report(records: Sequence[Record], fits: Dict[str, DecayFit],
                      gate_counts: Dict[int, int]) -> str:
    lines = ["# Break-even analysis", ""]
    lines += [
        "Cost model: a Grover attempt costs k oracle queries plus 1 classical query to verify "
        "the measured candidate, repeated until success, so E_q = (k+1)/p. Classical brute force "
        "without replacement needs E_c = (N+1)/(M+1). Advantage = E_c / E_q; below 1, classical "
        "search would have been the better choice.",
        "",
        "## Measured results",
        "",
        "| platform | instance | n | M | k | 2q gates | depth | P(success) | 95% CI | E_q | E_c | advantage |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = {p: i for i, p in enumerate(STYLE)}
    for r in sorted(records, key=lambda r: (order.get(r.platform, 9), r.num_qubits, r.num_marked, r.iterations)):
        eq = quantum_expected_queries(r.iterations, r.success_prob)
        ec = classical_expected_queries(2 ** r.num_qubits, r.num_marked)
        lines.append(
            f"| {r.platform} | {r.instance} | {r.num_qubits} | {r.num_marked} | {r.iterations}"
            f"{'*' if r.iterations == r.k_opt else ''} | {r.two_qubit_gates} | {r.depth} | "
            f"{r.success_prob:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | {eq:.2f} | {ec:.2f} | "
            f"{ec / eq:.2f} |"
        )
    lines += ["", "`*` marks k = k*.", "", "## Noise model fit", ""]
    if not fits:
        lines.append("Not enough noisy data points to fit a decay rate.")
    for p, fit in fits.items():
        lines.append(
            f"- **{STYLE[p]['label']}**: F ≈ exp(−{fit.rate:.4f}·G₂q), i.e. an effective "
            f"{fit.per_gate_error:.2%} error per two-qubit gate (fit over {len(fit.points)} points)."
        )
    lines += ["", "## Extrapolated break-even (M=1, k=k*)", "",
              "| n | N | k* | 2q gates (compiled) | max 2q gates for advantage | "
              + " | ".join(f"advantage ({p})" for p in fits) + " |",
              "|---|---|---|---|---|" + "---|" * len(fits)]
    for n in sorted(gate_counts):
        N = 2 ** n
        g = gate_counts[n]
        budget = ", ".join(
            f"{p}: {max_gates_for_advantage(n, rate=f.rate):.0f}" for p, f in fits.items()
        ) or "–"
        adv = " | ".join(f"{_predicted_advantage(n, g, f.rate):.2f}" for f in fits.values())
        lines.append(f"| {n} | {N} | {optimal_iterations(N, 1)} | {g} | {budget} | {adv} |")
    for p, fit in fits.items():
        losing = [n for n in sorted(gate_counts) if _predicted_advantage(n, gate_counts[n], fit.rate) < 1]
        verdict = (f"Grover stops beating brute force from n = {losing[0]} "
                   f"({gate_counts[losing[0]]} two-qubit gates)." if losing
                   else "Grover keeps an advantage over the whole range examined.")
        lines += ["", f"**{STYLE[p]['label']}:** {verdict}"]
    lines += [
        "",
        "Note: the depolarising model is a first-order description. On real hardware the "
        "output distribution is biased (e.g. towards |0…0⟩ by T1 decay), so F is an effective, "
        "not a physical, fidelity.",
    ]
    return "\n".join(lines) + "\n"
