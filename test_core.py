"""Tests for GTC-R core."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from core.gtcr import baryon_mass, calibrate, benchmark, mass_gap_bound


def test_proton_mass():
    """Proton is the calibration anchor — must be exact."""
    m = baryon_mass(J=0.5, I=0.5, S=0, n=0)
    assert abs(m - 938.27) < 1.0, f"Proton mass off: {m}"


def test_delta_mass():
    """Δ(1232) is the second anchor."""
    m = baryon_mass(J=1.5, I=1.5, S=0, n=0)
    assert abs(m - 1232.0) < 1.0, f"Delta mass off: {m}"


def test_omega_prediction():
    """Ω(1672) — strangeness-3 prediction."""
    m = baryon_mass(J=1.5, I=0.0, S=3, n=0)
    assert abs(m - 1672.45) < 50.0, f"Omega mass off: {m}"


def test_roper_prediction():
    """N*(1440) Roper — radial excitation n=1."""
    m = baryon_mass(J=0.5, I=0.5, S=0, n=1)
    assert abs(m - 1440.0) < 60.0, f"Roper mass off: {m}"


def test_benchmark_mae():
    """Mean absolute error across 12 baryons < 10%."""
    results = benchmark()
    mae = np.mean([abs(r["error_pct"]) for r in results])
    assert mae < 10.0, f"MAE too large: {mae:.2f}%"


def test_mass_gap_positive():
    """Yang-Mills mass gap lower bound must be positive."""
    g = mass_gap_bound()
    assert g["Delta_lower_MeV"] > 0, "Mass gap bound is not positive!"


def test_derrick_condition():
    """A ≈ B (Derrick equilibrium) from exact numerical solution."""
    from core.gtcr import _Integrals
    itg = _Integrals.get()
    ratio = itg["A"] / itg["B"]
    assert abs(ratio - 1.0) < 1e-3, f"Derrick condition violated: A/B = {ratio}"


def test_baryon_number():
    """Topological baryon number B = 1."""
    from core.gtcr import _Integrals
    import numpy as np
    itg = _Integrals.get()
    f   = itg["f"]
    B   = 0.5 * (-np.cos(f[-1]) + np.cos(f[0]))
    assert abs(abs(B) - 1.0) < 1e-3, f"Baryon number off: B = {B}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
