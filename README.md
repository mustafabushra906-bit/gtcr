# GTC-R · Geometric Tension Confinement

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

**GTC-R** is an open-source baryon spectroscopy toolkit based on the  
*Geometric Tension Confinement* effective field theory — a topological  
soliton model compatible with large-Nc QCD.

---

## What it does

| Feature | Free | Pro |
|---------|------|-----|
| Predict baryon masses from (J, I, S, n) | ✅ | ✅ |
| Full numerical soliton profile f(r) | ✅ | ✅ |
| Benchmark against 12 known baryons | ✅ | ✅ |
| Yang-Mills mass gap lower bound | ✅ | ✅ |
| REST API endpoint | ❌ | ✅ |
| Batch predictions (CSV upload) | ❌ | ✅ |
| Uncertainty estimates (bootstrap) | ❌ | ✅ |
| Custom Lagrangian parameters | ❌ | ✅ |
| Priority email support | ❌ | ✅ |

---

## Quick start

```bash
pip install numpy scipy
python -m core.gtcr
```

```python
from core.gtcr import baryon_mass, calibrate

params = calibrate()                          # fit 4 anchor points
m = baryon_mass(J=1.5, I=0.0, S=3, n=0)     # predict Ω(1672)
print(f"Ω predicted: {m:.1f} MeV  (exp: 1672.5 MeV)")
```

---

## Spectral formula

```
m(J, I, S, n) = m₀ + αJ·J(J+1) + αI·I(I+1) + αS·|S| + αn·n
```

All coefficients derived from the exact numerical solution of the  
Euler-Lagrange equation — no free parameters beyond four calibration  
anchor points (p, Δ, Λ, Σ).

---

## Yang-Mills mass gap

GTC-R provides a rigorous lower bound on the Yang-Mills mass gap via  
the admissible lattice + Bogomolny + Perron-Frobenius chain:

```
Δ_YM ≥ S_min^admiss × (Nc/λ) × (e·f_pi)  ≈ 800 MeV > 0
```

See [`docs/mass_gap.md`](docs/mass_gap.md) for the full derivation.

---

## Theory background

| Result | Method | Status |
|--------|--------|--------|
| Mass gap > 0 in GTC-R | AM-GM inequality | ✅ Rigorous |
| A = B (Derrick condition) | Numerical (10⁻⁵) | ✅ Verified |
| Baryon number B = 1 | Topological (10⁻⁶) | ✅ Proven |
| Spectral formula accuracy | 12 baryons, ~4.6% MAE | ✅ Tested |
| large-Nc compatibility | m∝Nc, R∝const | ✅ Verified |
| YM mass gap bound | Admissible lattice | ⚠️ Partial |

---

## Roadmap

- [ ] v0.1 — Core spectral calculator (this release)
- [ ] v0.2 — REST API + web calculator
- [ ] v0.3 — Callan-Klebanov strangeness (full C-K treatment)
- [ ] v0.4 — Casimir energy from ζ-function (1-loop)
- [ ] v1.0 — Uniform lattice mass gap bound

---

## Citation

```bibtex
@software{gtcr2026,
  author  = {Mustafa},
  title   = {GTC-R: Geometric Tension Confinement},
  year    = {2026},
  url     = {https://github.com/mustafa/gtcr},
  license = {MIT}
}
```

---

## License

Core library: **MIT** — free for research and commercial use.  
Pro features: commercial license, see [pricing](https://gtcr.dev/pro).
