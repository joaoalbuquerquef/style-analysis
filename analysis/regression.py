"""
Sharpe (1992) Return-Based Style Analysis (RBSA).

Model
-----
  R_fund(t) = b1·CDI(t) + b2·IMAB(t) + b3·Ibovespa(t) + b4·BRL_USD(t) + e(t)

Constraints (Sharpe 1992, pure form — no intercept)
  sum(bi) = 1,   bi >= 0  for all i

The weights represent implied asset-class allocations.  After fitting, alpha
(manager selection return) is the annualised mean of the residual series.

Outputs per fund
  weight_CDI, weight_IMAB, weight_Ibovespa, weight_BRL_USD  (style exposures)
  alpha          annualised selection return  = (1 + mean_daily_residual)^252 - 1
  r_squared      fraction of variance explained by style benchmark
  tracking_error annualised std of daily residuals  = std(e) * sqrt(252)
  n_obs          number of daily observations used
  converged      whether SLSQP reached a solution
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Allow running as a script from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FACTORS: List[str] = ["CDI", "IMA-B", "Ibovespa", "BRL_USD"]
FACTOR_LABELS: List[str] = ["CDI", "IMAB", "Ibovespa", "BRL_USD"]   # safe column names
TRADING_DAYS: int = 252
MIN_OBS: int = 120          # skip funds with fewer aligned observations
OUTLIER_CLIP: float = 0.25  # drop rows where |fund return| > this (quota resets, data errors)

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
RESULTS_PATH = RAW_DIR / "style_results.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load fund returns and factor returns from data/raw/.
    Returns (fund_returns_df, factors_df) with no alignment — call _align()
    per-fund inside the loop.
    """
    fund_ret = pd.read_csv(RAW_DIR / "fund_returns.csv", index_col=0, parse_dates=True)
    fund_ret.index = pd.to_datetime(fund_ret.index)

    factors = pd.read_csv(RAW_DIR / "factors.csv", index_col=0, parse_dates=True)
    factors.index = pd.to_datetime(factors.index)

    log.info(
        "Loaded fund returns %s and factors %s",
        fund_ret.shape, factors.shape,
    )
    return fund_ret, factors


def _load_fund_names() -> Dict[str, str]:
    """Return {cnpj: name} from the fund list cache (best-effort)."""
    path = RAW_DIR / "funds_list.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str)
    if "cnpj" in df.columns and "nome" in df.columns:
        return dict(zip(df["cnpj"], df["nome"]))
    return {}


# ---------------------------------------------------------------------------
# Core solver
# ---------------------------------------------------------------------------

def _sharpe_ols(
    y: np.ndarray,
    X: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    """
    Solve the Sharpe (1992) quadratic programme:
      min  ||y - X @ b||²
      s.t. sum(b) = 1,   b >= 0

    Uses SLSQP with an analytical Jacobian for speed and precision.

    Parameters
    ----------
    y : (T,)   fund return series
    X : (T, K) factor return matrix

    Returns
    -------
    b          (K,) optimal style weights (clipped and renormalised)
    converged  True if SLSQP reported success
    """
    K = X.shape[1]

    def objective(b: np.ndarray) -> float:
        resid = y - X @ b
        return float(resid @ resid)

    def jacobian(b: np.ndarray) -> np.ndarray:
        resid = y - X @ b
        return -2.0 * (X.T @ resid)

    result = minimize(
        objective,
        x0=np.full(K, 1.0 / K),
        method="SLSQP",
        jac=jacobian,
        bounds=[(0.0, 1.0)] * K,
        constraints={"type": "eq", "fun": lambda b: b.sum() - 1.0},
        options={"ftol": 1e-12, "maxiter": 2000},
    )

    b = np.clip(result.x, 0.0, None)
    total = b.sum()
    if total > 0:
        b /= total   # re-normalise to absorb floating-point rounding
    else:
        b = np.full(K, 1.0 / K)

    return b, result.success


# ---------------------------------------------------------------------------
# Per-fund analysis
# ---------------------------------------------------------------------------

def analyze_fund(
    cnpj: str,
    fund_series: pd.Series,
    factors: pd.DataFrame,
    min_obs: int = MIN_OBS,
) -> Optional[Dict]:
    """
    Align dates, run Sharpe RBSA, and return a result dict for one fund.
    Returns None if the fund has fewer than min_obs complete observations.

    Parameters
    ----------
    cnpj        fund identifier (used as dict key only)
    fund_series daily return Series for the fund
    factors     DataFrame with columns FACTORS (CDI, IMA-B, Ibovespa, BRL_USD)
    """
    # Inner join on dates; drop any row where fund OR any factor is NaN
    df = pd.concat(
        [fund_series.rename("fund"), factors[FACTORS]],
        axis=1,
        join="inner",
    ).dropna()

    # Remove extreme outliers from fund series (CVM quota resets produce
    # artifactual ±100% or +99M% single-day returns that dominate OLS)
    n_before = len(df)
    df = df[df["fund"].abs() <= OUTLIER_CLIP]
    n_removed = n_before - len(df)
    if n_removed > 0:
        log.warning(
            "  %s: dropped %d outlier row(s) (|return| > %.0f%%) — likely quota resets.",
            cnpj, n_removed, OUTLIER_CLIP * 100,
        )

    if len(df) < min_obs:
        log.warning(
            "  %s: only %d aligned obs (< %d) — skipped.", cnpj, len(df), min_obs
        )
        return None

    y = df["fund"].to_numpy(dtype=float)
    X = df[FACTORS].to_numpy(dtype=float)

    weights, converged = _sharpe_ols(y, X)
    if not converged:
        log.warning("  %s: SLSQP did not converge — using best-found weights.", cnpj)

    fitted = X @ weights
    residuals = y - fitted

    # R-squared
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Alpha: annualised mean daily residual (compounded)
    alpha_daily = float(residuals.mean())
    alpha_annual = float((1.0 + alpha_daily) ** TRADING_DAYS - 1.0)

    # Tracking error: annualised std of daily residuals
    tracking_error = float(residuals.std(ddof=1) * np.sqrt(TRADING_DAYS))

    return {
        "cnpj":            cnpj,
        "weight_CDI":      float(weights[0]),
        "weight_IMAB":     float(weights[1]),
        "weight_Ibovespa": float(weights[2]),
        "weight_BRL_USD":  float(weights[3]),
        "alpha":           alpha_annual,
        "r_squared":       r_squared,
        "tracking_error":  tracking_error,
        "n_obs":           len(df),
        "converged":       bool(converged),
    }


# ---------------------------------------------------------------------------
# Run all funds
# ---------------------------------------------------------------------------

def run_all_funds(
    save_path: Optional[str] = None,
    min_obs: int = MIN_OBS,
    force: bool = False,
) -> pd.DataFrame:
    """
    Run Sharpe RBSA for every fund in data/raw/fund_returns.csv.

    Parameters
    ----------
    save_path   where to write style_results.csv (default: data/raw/style_results.csv)
    min_obs     minimum aligned observations required per fund
    force       if True, recompute even if cache exists

    Returns
    -------
    pd.DataFrame with one row per fund, columns:
        cnpj, nome, weight_CDI, weight_IMAB, weight_Ibovespa, weight_BRL_USD,
        alpha, r_squared, tracking_error, n_obs, converged, dominant_style
    """
    out_path = Path(save_path) if save_path else RESULTS_PATH

    if out_path.exists() and not force:
        log.info("Style results loaded from cache (%s).", out_path)
        return pd.read_csv(out_path)

    fund_ret, factors = load_data()
    fund_names = _load_fund_names()

    results: List[Dict] = []
    cnpjs = fund_ret.columns.tolist()
    log.info("Running Sharpe RBSA for %d funds…", len(cnpjs))

    for i, cnpj in enumerate(cnpjs, 1):
        log.info("[%d/%d] %s", i, len(cnpjs), cnpj)
        res = analyze_fund(cnpj, fund_ret[cnpj], factors, min_obs=min_obs)
        if res is not None:
            results.append(res)

    if not results:
        raise RuntimeError("No funds produced valid style results.")

    df = pd.DataFrame(results)

    # Attach fund names
    df["nome"] = df["cnpj"].map(fund_names).fillna("")

    # Dominant style (factor with highest weight)
    weight_cols = ["weight_CDI", "weight_IMAB", "weight_Ibovespa", "weight_BRL_USD"]
    label_map = dict(zip(weight_cols, FACTOR_LABELS))
    df["dominant_style"] = df[weight_cols].idxmax(axis=1).map(label_map)

    # Reorder columns
    df = df[[
        "cnpj", "nome",
        "weight_CDI", "weight_IMAB", "weight_Ibovespa", "weight_BRL_USD",
        "alpha", "r_squared", "tracking_error",
        "n_obs", "converged", "dominant_style",
    ]]

    df.to_csv(out_path, index=False)
    log.info("Saved %d style results → %s", len(df), out_path)

    _print_summary(df)
    return df


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    """Print a formatted summary table of style analysis results."""
    print()
    print("=" * 110)
    print("  SHARPE (1992) RETURN-BASED STYLE ANALYSIS  —  SUMMARY")
    print("=" * 110)

    # Build display frame
    display = pd.DataFrame()
    display["CNPJ"] = df["cnpj"].str.replace(r"[./\-]", "", regex=True).str[:14]
    display["Name"] = df["nome"].str[:32].str.ljust(32)
    display["CDI%"] = (df["weight_CDI"] * 100).map("{:5.1f}".format)
    display["IMAB%"] = (df["weight_IMAB"] * 100).map("{:5.1f}".format)
    display["Ibov%"] = (df["weight_Ibovespa"] * 100).map("{:5.1f}".format)
    display["FX%"] = (df["weight_BRL_USD"] * 100).map("{:5.1f}".format)
    display["Alpha%"] = (df["alpha"] * 100).map("{:+6.2f}".format)
    display["R²"] = df["r_squared"].map("{:.3f}".format)
    display["TE%"] = (df["tracking_error"] * 100).map("{:5.2f}".format)
    display["Dominant"] = df["dominant_style"].str.ljust(8)
    display["N"] = df["n_obs"]

    print(display.to_string(index=False))
    print("=" * 110)

    # Aggregate stats
    print()
    print("  AGGREGATE STATISTICS")
    print(f"  Funds analysed:        {len(df)}")
    print(f"  Converged:             {df['converged'].sum()} / {len(df)}")
    print(f"  Mean R²:               {df['r_squared'].mean():.3f}")
    print(f"  Median alpha:          {df['alpha'].median() * 100:+.2f}%")
    print(f"  Mean tracking error:   {df['tracking_error'].mean() * 100:.2f}%")
    print()
    print("  STYLE DISTRIBUTION  (dominant style by fund count)")
    print(df["dominant_style"].value_counts().to_string())
    print()
    print("  AVERAGE STYLE WEIGHTS")
    for col, label in zip(weight_cols := [
        "weight_CDI", "weight_IMAB", "weight_Ibovespa", "weight_BRL_USD"
    ], FACTOR_LABELS):
        print(f"    {label:<12}: {df[col].mean() * 100:5.1f}%")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_all_funds(force="--force" in sys.argv)
    print(f"\nresults shape: {results.shape}")
    print(f"saved to: {RESULTS_PATH}")
