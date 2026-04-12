"""
GTC-R Pro API
=============
FastAPI server — requires a valid Pro license key.

Run:
    pip install fastapi uvicorn
    uvicorn api.server:app --reload

License validation is checked per-request via X-License-Key header.
"""

from fastapi import FastAPI, HTTPException, Header, UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import csv
import io
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from core.gtcr import baryon_mass, calibrate, benchmark, mass_gap_bound

app = FastAPI(
    title="GTC-R Pro API",
    description="Baryon spectroscopy via topological soliton theory",
    version="0.1.0",
)

# ── License validation (stub — replace with real DB check) ──────────
VALID_KEYS = {"pro-demo-key-2026"}   # replace with real validation

def _check_license(key: Optional[str]):
    if key not in VALID_KEYS:
        raise HTTPException(status_code=403, detail="Invalid Pro license key.")


# ── Schemas ──────────────────────────────────────────────────────────
class BaryonRequest(BaseModel):
    J: float = Field(..., description="Total spin (0.5, 1.5, 2.5, …)")
    I: float = Field(..., description="Isospin (0, 0.5, 1, 1.5, …)")
    S: int   = Field(..., description="Strangeness magnitude |S| (0–3)")
    n: int   = Field(0,   description="Radial excitation number")

class BaryonResponse(BaseModel):
    J: float; I: float; S: int; n: int
    mass_MeV: float
    quantum_numbers: str


class MassGapResponse(BaseModel):
    S_min_admiss:    float
    E_top_MeV:       float
    E_cas_MeV:       float
    Delta_lower_MeV: float
    lambda_tHooft:   float
    Nc:              int
    note: str


# ── FREE endpoints ───────────────────────────────────────────────────
@app.get("/", tags=["info"])
def root():
    return {
        "project": "GTC-R",
        "version": "0.1.0",
        "docs": "/docs",
        "pro": "Pass X-License-Key header to unlock Pro endpoints.",
    }


@app.post("/predict", tags=["free"], response_model=BaryonResponse)
def predict_single(req: BaryonRequest):
    """Predict mass for a single baryon (free tier)."""
    m = baryon_mass(req.J, req.I, req.S, req.n)
    return BaryonResponse(
        J=req.J, I=req.I, S=req.S, n=req.n,
        mass_MeV=round(m, 2),
        quantum_numbers=f"J={req.J}, I={req.I}, S={req.S}, n={req.n}",
    )


@app.get("/benchmark", tags=["free"])
def run_benchmark():
    """Benchmark GTC-R against 12 known baryons."""
    results = benchmark()
    mean_err = np.mean([abs(r["error_pct"]) for r in results])
    return {"mean_abs_error_pct": round(mean_err, 2), "baryons": results}


@app.get("/mass-gap", tags=["free"], response_model=MassGapResponse)
def get_mass_gap():
    """Yang-Mills mass gap lower bound from GTC-R."""
    g = mass_gap_bound()
    return MassGapResponse(**g, note=(
        "Rigorous lower bound from admissible lattice + Bogomolny + AM-GM. "
        "Continuum limit requires RG argument (open problem)."
    ))


# ── PRO endpoints ────────────────────────────────────────────────────
@app.post("/pro/batch", tags=["pro"])
async def predict_batch(
    file: UploadFile = File(...),
    x_license_key: Optional[str] = Header(None),
):
    """
    [PRO] Batch predict from CSV.
    CSV columns: J, I, S, n  (header row required).
    Returns JSON array of predictions.
    """
    _check_license(x_license_key)
    content = await file.read()
    reader  = csv.DictReader(io.StringIO(content.decode()))
    results = []
    for row in reader:
        try:
            J = float(row["J"]); I = float(row["I"])
            S = int(row["S"]);   n = int(row.get("n", 0))
            m = baryon_mass(J, I, S, n)
            results.append({"J": J, "I": I, "S": S, "n": n, "mass_MeV": round(m, 2)})
        except (KeyError, ValueError) as e:
            results.append({"error": str(e), "row": row})
    return {"count": len(results), "predictions": results}


@app.post("/pro/uncertainty", tags=["pro"])
def predict_with_uncertainty(
    req: BaryonRequest,
    x_license_key: Optional[str] = Header(None),
    n_bootstrap: int = 500,
):
    """
    [PRO] Prediction with bootstrap uncertainty estimate.
    Randomly varies calibration anchor points within ±0.5% PDG uncertainty.
    """
    _check_license(x_license_key)
    from core.gtcr import calibrate
    masses = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        noise = rng.normal(0, 0.005, 4)   # ±0.5% noise
        mp = calibrate(
            m_proton =  938.27 * (1 + noise[0]),
            m_delta  = 1232.00 * (1 + noise[1]),
            m_lambda = 1115.68 * (1 + noise[2]),
            m_sigma  = 1192.64 * (1 + noise[3]),
        )
        masses.append(baryon_mass(req.J, req.I, req.S, req.n, mp))
    masses = np.array(masses)
    return {
        "mass_MeV":   round(float(np.median(masses)), 2),
        "sigma_MeV":  round(float(np.std(masses)), 2),
        "ci_68":      [round(float(np.percentile(masses, 16)), 2),
                       round(float(np.percentile(masses, 84)), 2)],
        "ci_95":      [round(float(np.percentile(masses, 2.5)), 2),
                       round(float(np.percentile(masses, 97.5)), 2)],
        "n_bootstrap": n_bootstrap,
    }


@app.post("/pro/custom-params", tags=["pro"])
def predict_custom(
    req: BaryonRequest,
    m0: float = 909.41, alpha_J: float = 59.43,
    alpha_I: float = 38.48, alpha_S: float = 206.27, alpha_n: float = 432.0,
    x_license_key: Optional[str] = Header(None),
):
    """
    [PRO] Predict with custom Lagrangian parameters.
    Override any spectral coefficient directly.
    """
    _check_license(x_license_key)
    m = (m0
         + alpha_J * req.J * (req.J + 1)
         + alpha_I * req.I * (req.I + 1)
         + alpha_S * abs(req.S)
         + alpha_n * req.n)
    return {"mass_MeV": round(m, 2), "params_used": {
        "m0": m0, "alpha_J": alpha_J, "alpha_I": alpha_I,
        "alpha_S": alpha_S, "alpha_n": alpha_n,
    }}
