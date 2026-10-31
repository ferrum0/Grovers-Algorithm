from grover import backends
from grover.experiment import compile_sweep, run_simulations
from grover.instances import get_instance


def test_ideal_and_noisy_pipeline():
    target = backends.target_backend("fake_torino")
    compiled = compile_sweep([get_instance("sat3_m1")], [0, 2], target)
    recs = run_simulations(compiled, target, shots=2000, log=lambda *_: None)
    by = {(r.platform, r.iterations): r for r in recs}
    assert abs(by["ideal", 2].success_prob - by["ideal", 2].theory_prob) < 0.04
    assert by["noisy", 2].success_prob < by["ideal", 2].success_prob
    assert by["noisy", 2].success_prob > 3 * by["noisy", 0].success_prob  # amplification survives at n=3
    assert by["ideal", 2].two_qubit_gates == by["noisy", 2].two_qubit_gates > 0
