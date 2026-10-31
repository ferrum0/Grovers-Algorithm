# Grover's Algorithm on Real Hardware

How much of Grover's quadratic speedup survives on today's noisy quantum hardware?
This project runs the same Grover circuits for small 3-SAT instances on

1. an **ideal simulator** (theoretical baseline),
2. a **noise-model simulator** built from IBM device calibration data, and
3. **real IBM Quantum hardware** (free Open plan),

and measures where accumulated gate error cancels the query advantage.

## Layout

```
grover/
  sat.py          CNF formulas, evaluation, planted 3-SAT generator
  instances.py    frozen benchmark instances (n = 3, 4, 5; M = 1 or 2)
  circuits.py     phase oracle (X-conjugated multi-controlled Z), diffuser, k*, theory
  backends.py     ideal / noise-model Aer simulators, fake + live IBM backends
  experiment.py   compile sweeps, run, score, submit/fetch hardware jobs, save JSON
  stats.py        Wilson intervals, expected query counts
  analysis.py     figures + break-even report
main.py           CLI
tests/            unit + end-to-end tests
results/          raw run data (JSON, includes full counts)
figures/          generated plots and report.md
```

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Usage

```bash
# Inspect the instances, their solutions and k* = floor(pi/4 * sqrt(N/M))
python main.py instances

# Ideal + noise-model simulation, compiled for an offline calibration snapshot
python main.py simulate --backend fake_torino --qubits 3 4 5 --shots 4096
python main.py simulate --instances sat4_m2 sat5_m2      # instances with M = 2

# Plots + break-even report from every results/*.json file
python main.py analyze
```

### Real hardware

1. Create a free account at <https://quantum.cloud.ibm.com> and save your API key once:
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token="<API key>", set_as_default=True)
   ```
2. Check circuit costs first (`--dry-run` compiles only), then submit:
   ```bash
   python main.py hardware submit --backend least_busy --qubits 3 4 5 --dry-run
   python main.py hardware submit --backend least_busy --qubits 3 4 5 --with-noisy-sim
   ```
   All circuits go into one Sampler job (job mode, which the Open plan supports).
   `--with-noisy-sim` also runs both simulators using **that day's** calibration of the same
   device, so all three platforms are compared like for like. `--dd` and `--twirling`
   turn on dynamical decoupling and Pauli twirling, which helps you measure how much error suppression gains.
3. Jobs can queue for hours. Collect results later:
   ```bash
   python main.py hardware fetch results/hw_job_<job_id>.json
   python main.py analyze
   ```

The Open plan has a monthly QPU-time budget (10 minutes at the time of writing).
A full 3–5 qubit sweep at 4096 shots is roughly 18 circuits and stays well within it.

## Method

- **Oracle.** For a formula f, `U_f|x> = (-1)^f(x)|x>` is built as one multi-controlled Z
  per satisfying assignment, with X gates on the zero bits. Qiskit decomposes it into the
  device's native gates (CZ/ECR + single-qubit rotations).
- **Iterations.** The default sweep is k = 0 … k*+2 (k*+1 on hardware), so the plots show
  both the rise and the over-rotation fall-off. `best_iterations()` also reports the exact
  argmax of sin²((2k+1)θ).
- **One circuit, three platforms.** Each circuit is transpiled once for the target device
  (optimization level 3, fixed seed), and that exact ISA circuit runs everywhere. The
  depth and two-qubit gate counts are therefore the same on every platform.
- **Statistics.** Success probability = fraction of shots that satisfy the formula, with
  95% Wilson score intervals.

### Break-even analysis

- Cost of Grover: each attempt costs k oracle queries + 1 classical check of the
  measured answer, repeated until success → `E_q = (k+1)/p`.
- Cost of classical brute force (without replacement): `E_c = (N+1)/(M+1)`.
- Advantage = `E_c / E_q`. Below 1, classical search would have been more reliable.
- To extrapolate, noisy output is modelled as `p = F·p_ideal + (1−F)·M/N` with
  `F = exp(−λ·G₂q)`. λ is fitted on the measured points, which gives an effective error per
  two-qubit gate. The k* circuit is then **compiled** (not run) for larger n, and
  the report gives the n and gate count at which the advantage drops below 1.

Outputs: `figures/success_vs_iterations.png`, `figures/success_vs_qubits.png`,
`figures/break_even.png`, `figures/report.md`.

## Example result (noise-model simulator, FakeTorino snapshot)

| n | 2q gates at k* | P ideal | P noisy | advantage (noisy) |
|---|---|---|---|---|
| 3 | 39  | 0.95 | 0.80 | 1.23 |
| 4 | 165 | 0.96 | 0.50 | 1.10 |
| 5 | 501 | 1.00 | 0.13 | 0.45 |

Fitted effective error ≈ 0.4% per two-qubit gate. Under that model Grover loses to
brute force from n = 5. The gate count of the k* circuit grows much faster than the
√N query advantage, because the multi-controlled Z decomposes into many gates on
hardware with limited connectivity.

## Limitations

- The oracle is built from solutions found by classical enumeration. This is normal for a
  *benchmark*, since it measures how well the hardware carries out Grover, but it is not a
  real search. A real application would compile the clauses into a reversible circuit with
  ancilla qubits, which is even deeper.
- Counting oracle queries ignores the gate cost of each query. Wall-clock or energy
  comparisons with a classical CPU would be much less favourable still.
- The depolarising fidelity model is a first-order fit. Real devices also have biased errors
  such as T1 decay towards |0⟩ and readout asymmetry.
