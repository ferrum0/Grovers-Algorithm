"""Backend access: ideal simulator, calibrated noise-model simulator, IBM Quantum hardware."""

from __future__ import annotations

from typing import Optional

from qiskit_aer import AerSimulator


def fake_backend(name: str):
    """Instantiate an offline calibration snapshot, e.g. ``fake_torino`` -> FakeTorino()."""
    from qiskit_ibm_runtime import fake_provider

    cls_name = "Fake" + "".join(part.capitalize() for part in name.removeprefix("fake_").split("_"))
    cls = getattr(fake_provider, cls_name, None)
    if cls is None:
        available = sorted(n for n in dir(fake_provider) if n.startswith("Fake") and not n.endswith("V2"))
        raise ValueError(f"unknown fake backend {name!r}; available classes: {available}")
    return cls()


def runtime_service(channel: Optional[str] = None, instance: Optional[str] = None):
    """Connect to IBM Quantum using credentials saved by ``QiskitRuntimeService.save_account``."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    kwargs = {}
    if channel:
        kwargs["channel"] = channel
    if instance:
        kwargs["instance"] = instance
    return QiskitRuntimeService(**kwargs)


def ibm_backend(name: str, min_qubits: int = 5, service=None):
    """A real device by name, or ``least_busy`` for the least-loaded operational device."""
    service = service or runtime_service()
    if name in ("least_busy", "least-busy"):
        return service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
    return service.backend(name)


def target_backend(name: str, min_qubits: int = 5):
    """The device whose gate set, connectivity and error rates the circuits are compiled for.

    ``fake_*`` names work offline (calibration snapshot shipped with qiskit-ibm-runtime);
    anything else is fetched live from IBM Quantum, so its noise model reflects today's
    calibration data.
    """
    if name.startswith("fake_"):
        return fake_backend(name)
    return ibm_backend(name, min_qubits=min_qubits)


def ideal_simulator(seed: Optional[int] = None) -> AerSimulator:
    return AerSimulator(seed_simulator=seed)


def noisy_simulator(backend, seed: Optional[int] = None) -> AerSimulator:
    """Aer simulator with a noise model built from ``backend``'s calibration data."""
    return AerSimulator.from_backend(backend, seed_simulator=seed)
