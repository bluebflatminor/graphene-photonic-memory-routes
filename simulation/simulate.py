"""
Graphene Integration Route Selection for Ferroelectric Photonic Memory
Bayesian Monte Carlo Simulation — 10 Process Scenarios

Paper: "Graphene Integration Route Selection for Ferroelectric Photonic Memory:
        A Bayesian Quality Assessment Across Ten Process Scenarios"
Author: Nils Haaland (nhaaland@yahoo.com)
Date:   March 2026

Description
-----------
This script reproduces all Monte Carlo results reported in the paper.
For each of the ten integration routes (A–J), it:
  1. Draws N=200,000 correlated lognormal samples for the four continuous
     quality parameters (D/G ratio, domain size, mobility, carrier density)
     using a Cholesky decomposition applied in log-space.
  2. Draws monolayer coverage independently from a Beta distribution
     followed by a Bernoulli threshold.
  3. Applies transfer penalties (CVD routes) as multiplicative lognormal factors.
  4. Applies H-plasma carrier density reduction (in-place routes) as a
     Beta-distributed multiplicative factor.
  5. Evaluates pass/fail against the five Goldilocks thresholds.
  6. Reports marginal and joint pass probabilities.

Goldilocks Thresholds (64x64 MVM tile, 0.5 dB IL budget)
---------------------------------------------------------
  D/G ratio         < 0.1
  Domain size       > 10 µm
  Mobility          > 3,000 cm²/V·s
  Carrier density   < 1e11 cm⁻²
  Monolayer coverage > 95% area fraction

Dependencies
------------
  numpy >= 1.24
  scipy >= 1.10
  pandas >= 1.5
  json (stdlib)

Usage
-----
  python simulate.py                  # run all routes, print summary table
  python simulate.py --route F        # run single route
  python simulate.py --sensitivity    # run correlation sensitivity analysis
  python simulate.py --ceiling        # run Route F mobility ceiling analysis

  Output CSV files are written to ../results/
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from scipy import stats

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

N_SAMPLES = 200_000
RANDOM_SEED = 42

# Goldilocks thresholds
THRESHOLDS = {
    "dg":       {"limit": 0.1,    "direction": "below"},
    "domain":   {"limit": 10.0,   "direction": "above"},   # µm
    "mobility": {"limit": 3000.0, "direction": "above"},   # cm²/V·s
    "carrier":  {"limit": 1e11,   "direction": "below"},   # cm⁻²
    "monolayer":{"limit": 0.95,   "direction": "above"},   # area fraction
}

# Correlation matrix applied in log-space (4×4: D/G, domain, mobility, carrier)
# ρ(D/G, mobility)   = -0.6  (defects scatter carriers)
# ρ(domain, mobility) = +0.7  (grain boundary density drives both)
# ρ(D/G, carrier)    = +0.3  (charged defects act as dopants)
CORR_MATRIX = np.array([
    [1.0,  0.0, -0.6,  0.3],   # D/G
    [0.0,  1.0,  0.7,  0.0],   # domain
    [-0.6, 0.7,  1.0,  0.0],   # mobility
    [0.3,  0.0,  0.0,  1.0],   # carrier density
])


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRIORS
# ─────────────────────────────────────────────────────────────────────────────

def load_priors(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '..', 'priors', 'priors.json')
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# CORE SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def cholesky_lognormal(means_ln, stds_ln, corr, n):
    """
    Draw n correlated lognormal samples.

    Parameters
    ----------
    means_ln : array-like, log-scale means (len k)
    stds_ln  : array-like, log-scale std deviations (len k)
    corr     : (k×k) correlation matrix
    n        : number of samples

    Returns
    -------
    samples  : (n × k) array of lognormal variates
    """
    k = len(means_ln)
    # Build covariance matrix in log-space
    cov = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            cov[i, j] = corr[i, j] * stds_ln[i] * stds_ln[j]

    rng = np.random.default_rng(RANDOM_SEED)
    z = rng.multivariate_normal(means_ln, cov, size=n)
    return np.exp(z)


def apply_transfer_penalty(base_samples, penalty_prior, rng):
    """
    Multiply carrier density by a lognormal transfer penalty factor.
    mobility is multiplied by a Beta retention factor.
    """
    n = len(base_samples)
    # Carrier density penalty: multiplicative lognormal
    t_n_mean = np.log(penalty_prior["carrier_factor_mean"])
    t_n_std  = penalty_prior["carrier_factor_sigma_ln"]
    penalty_n = rng.lognormal(t_n_mean, t_n_std, size=n)

    # Mobility retention: Beta distribution
    alpha = penalty_prior["mobility_retention_alpha"]
    beta  = penalty_prior["mobility_retention_beta"]
    retention_mu = rng.beta(alpha, beta, size=n)

    base_samples[:, 3] *= penalty_n      # carrier density column
    base_samples[:, 2] *= retention_mu   # mobility column
    return base_samples


def apply_h_plasma(carrier_samples, h_plasma_prior, rng):
    """
    Reduce carrier density by H-plasma treatment.
    H-plasma reduction factor ~ Beta(alpha, beta), mean ~0.8 (20% reduction).
    """
    n = len(carrier_samples)
    alpha = h_plasma_prior["alpha"]
    beta  = h_plasma_prior["beta"]
    reduction = rng.beta(alpha, beta, size=n)
    return carrier_samples * reduction


def sample_monolayer(mono_prior, n, rng):
    """
    Sample monolayer pass/fail.

    Model: each cell draws a pass probability p ~ Beta(alpha, beta),
    then a Bernoulli(p) outcome. This is equivalent to a Beta-Bernoulli
    model where the marginal pass rate = alpha / (alpha + beta).

    The Beta distribution captures inter-cell variation in monolayer
    uniformity driven by nucleation kinetics. It is kept independent of
    the other quality parameters by construction.

    Returns
    -------
    coverage : array of float in [0, 1] — the Beta draw (used as pass probability)
               The pass criterion is coverage > 0 (Bernoulli interpretation),
               so pass rate == Beta mean == alpha/(alpha+beta).
    """
    alpha = mono_prior["alpha"]
    beta  = mono_prior["beta"]
    # Draw Bernoulli probability from Beta, then draw outcome
    p = rng.beta(alpha, beta, size=n)
    outcome = rng.uniform(size=n) < p   # Bernoulli(p)
    # Return as float for threshold check (threshold is 0.5, all True > 0.5)
    return outcome.astype(float)


def evaluate_route(route_key, priors, n=N_SAMPLES, mobility_override=None):
    """
    Run the Monte Carlo simulation for a single route.

    Returns
    -------
    dict with marginal pass rates and joint P(all 5)
    """
    rng = np.random.default_rng(RANDOM_SEED)
    p = priors["routes"][route_key]

    # Log-scale means and stds for [D/G, domain, mobility, carrier]
    means_ln = [
        np.log(p["dg"]["mean"]),
        np.log(p["domain"]["mean"]),
        np.log(p["mobility"]["mean"] if mobility_override is None else mobility_override),
        np.log(p["carrier"]["mean"]),
    ]
    stds_ln = [
        p["dg"]["sigma_ln"],
        p["domain"]["sigma_ln"],
        p["mobility"]["sigma_ln"],
        p["carrier"]["sigma_ln"],
    ]

    # Draw correlated samples
    samples = cholesky_lognormal(means_ln, stds_ln, CORR_MATRIX, n)
    # columns: [D/G, domain, mobility, carrier]

    # Apply transfer penalty if route has one
    if "transfer" in p:
        samples = apply_transfer_penalty(samples, p["transfer"], rng)

    # Apply H-plasma treatment to carrier density
    if "h_plasma" in p:
        samples[:, 3] = apply_h_plasma(samples[:, 3], p["h_plasma"], rng)

    # Sample monolayer coverage independently
    mono = sample_monolayer(p["monolayer"], n, rng)

    # Evaluate thresholds
    pass_dg       = samples[:, 0] < THRESHOLDS["dg"]["limit"]
    pass_domain   = samples[:, 1] > THRESHOLDS["domain"]["limit"]
    pass_mobility = samples[:, 2] > THRESHOLDS["mobility"]["limit"]
    pass_carrier  = samples[:, 3] < THRESHOLDS["carrier"]["limit"]
    pass_mono     = mono > 0.5   # Bernoulli outcome: 1.0 = pass, 0.0 = fail

    pass_all = pass_dg & pass_domain & pass_mobility & pass_carrier & pass_mono

    return {
        "route":    route_key,
        "p_dg":     pass_dg.mean(),
        "p_domain": pass_domain.mean(),
        "p_mobility": pass_mobility.mean(),
        "p_carrier":  pass_carrier.mean(),
        "p_mono":     pass_mono.mean(),
        "p_all":      pass_all.mean(),
        "n_samples":  n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY ANALYSES
# ─────────────────────────────────────────────────────────────────────────────

def run_mobility_ceiling(priors, route="F"):
    """Route F mobility ceiling analysis."""
    targets = [3000, 4000, 5000, 6000, 8000, 10000]
    results = []
    for mob in targets:
        r = evaluate_route(route, priors, mobility_override=mob)
        results.append({"mobility_mean": mob, "p_all": r["p_all"],
                        "p_mobility": r["p_mobility"], "p_carrier": r["p_carrier"]})
    return pd.DataFrame(results)


def run_correlation_sensitivity(priors, delta=0.2):
    """
    Perturb each off-diagonal correlation by ±delta and report
    Route F and Route I joint pass probabilities.
    """
    global CORR_MATRIX
    base = CORR_MATRIX.copy()
    results = []

    perturbations = {
        "rho(DG,mob) +delta":   (0, 2, delta),
        "rho(DG,mob) -delta":   (0, 2, -delta),
        "rho(dom,mob) +delta":  (1, 2, delta),
        "rho(dom,mob) -delta":  (1, 2, -delta),
        "rho(DG,car) +delta":   (0, 3, delta),
        "rho(DG,car) -delta":   (0, 3, -delta),
    }

    for label, (i, j, d) in perturbations.items():
        CORR_MATRIX = base.copy()
        CORR_MATRIX[i, j] += d
        CORR_MATRIX[j, i] += d
        # Clip to valid range
        CORR_MATRIX = np.clip(CORR_MATRIX, -0.95, 0.95)
        np.fill_diagonal(CORR_MATRIX, 1.0)

        try:
            rf = evaluate_route("F", priors)
            ri = evaluate_route("I", priors)
            results.append({
                "perturbation": label,
                "p_all_F": rf["p_all"],
                "p_all_I": ri["p_all"],
            })
        except np.linalg.LinAlgError:
            results.append({
                "perturbation": label,
                "p_all_F": np.nan,
                "p_all_I": np.nan,
                "note": "matrix not positive definite after perturbation"
            })

    CORR_MATRIX = base  # restore
    return pd.DataFrame(results)


def run_route_j_sensitivity(priors):
    """Route J transfer penalty sensitivity: 7× to 15×."""
    factors = [7, 8, 9, 10, 11, 12, 13, 14, 15]
    results = []
    import copy
    for f in factors:
        p_mod = copy.deepcopy(priors)
        p_mod["routes"]["J"]["transfer"]["carrier_factor_mean"] = f
        r = evaluate_route("J", p_mod)
        results.append({"carrier_factor": f, "p_all": r["p_all"]})
    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

ROUTE_LABELS = {
    "A": "UNCD UHV no catalyst",
    "B": "UNCD Ar no catalyst",
    "C": "UNCD Cu catalyst 950°C",
    "D": "UNCD Ni RTA",
    "E": "SCD(111) direct",
    "F": "SCD(111) Ni+H-term",
    "G": "CVD/Pt → SiO₂",
    "H": "CVD/Cu → SiO₂",
    "I": "CVD/Pt → hBN",
    "J": "CVD/Pt → SCD(111) bonded",
}

BINDING_CONSTRAINT = {
    "A": "Domain size",
    "B": "Domain size",
    "C": "Domain size",
    "D": "Domain size",
    "E": "Mobility / Carrier density",
    "F": "Mobility",
    "G": "Carrier density → Mobility",
    "H": "Carrier density → Mobility",
    "I": "Mobility",
    "J": "Carrier density → Mobility",
}


def main():
    parser = argparse.ArgumentParser(description="Graphene route Monte Carlo simulation")
    parser.add_argument("--route", type=str, default=None,
                        help="Single route key (A-J). Default: all routes.")
    parser.add_argument("--sensitivity", action="store_true",
                        help="Run correlation sensitivity analysis")
    parser.add_argument("--ceiling", action="store_true",
                        help="Run Route F mobility ceiling analysis")
    parser.add_argument("--routej", action="store_true",
                        help="Run Route J transfer penalty sensitivity")
    parser.add_argument("--priors", type=str, default=None,
                        help="Path to priors JSON file")
    parser.add_argument("--output", type=str, default="../results",
                        help="Output directory for CSV files")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    priors = load_priors(args.priors)

    if args.sensitivity:
        print("\n── Correlation Sensitivity Analysis ──")
        df = run_correlation_sensitivity(priors)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(args.output, "sensitivity_correlation.csv"), index=False)
        return

    if args.ceiling:
        print("\n── Route F Mobility Ceiling Analysis ──")
        df = run_mobility_ceiling(priors)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(args.output, "ceiling_route_F.csv"), index=False)
        return

    if args.routej:
        print("\n── Route J Transfer Penalty Sensitivity ──")
        df = run_route_j_sensitivity(priors)
        print(df.to_string(index=False))
        df.to_csv(os.path.join(args.output, "sensitivity_route_J.csv"), index=False)
        return

    routes = [args.route] if args.route else list(ROUTE_LABELS.keys())
    rows = []
    for key in routes:
        print(f"Simulating Route {key}: {ROUTE_LABELS[key]} ...", end=" ", flush=True)
        r = evaluate_route(key, priors)
        r["label"] = ROUTE_LABELS[key]
        r["binding"] = BINDING_CONSTRAINT[key]
        rows.append(r)
        print(f"P(all 5) = {r['p_all']*100:.1f}%")

    df = pd.DataFrame(rows)[[
        "route", "label", "p_dg", "p_domain", "p_mobility",
        "p_carrier", "p_mono", "p_all", "binding"
    ]]

    # Format for display
    df_display = df.copy()
    for col in ["p_dg","p_domain","p_mobility","p_carrier","p_mono","p_all"]:
        df_display[col] = (df[col] * 100).map("{:.1f}%".format)

    print("\n── Main Results (N = {:,}) ──".format(N_SAMPLES))
    print(df_display.to_string(index=False))

    out_path = os.path.join(args.output, "main_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
