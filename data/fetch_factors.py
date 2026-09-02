"""
Fetch the four style-analysis benchmark factors as daily return series.

Factors
-------
CDI      – overnight interbank rate (Brazil's risk-free proxy)
           Primary:  ANBIMA Preços e Índices v1
           Fallback: BCB/SGS public API (series 12), no auth required
IMA-B    – inflation-linked government bond index (ANBIMA)
Ibovespa – Brazilian equity benchmark (Yahoo Finance / yfinance)
BRL/USD  – Brazilian real vs US dollar exchange rate (Yahoo Finance)

All series are returned as daily decimal returns (e.g. 0.001 = 0.1 %).
Intermediate data is cached to data/raw/ as CSV files.
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import os

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# Allow running this file directly (python data/fetch_factors.py)
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://api.anbima.com.br"
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY = 0.4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _anbima_headers(token: str) -> dict:
    # Sensedia gateway expects access_token (not Authorization: Bearer) + client_id
    return {
        "access_token": token,
        "client_id": os.environ.get("ANBIMA_CLIENT_ID", ""),
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# CDI
# ---------------------------------------------------------------------------

def fetch_cdi(start_date: str, end_date: str, token: Optional[str] = None) -> pd.Series:
    """
    Return daily CDI as a decimal return series (e.g. 0.000489 for 0.0489 % /day).
    Tries ANBIMA first; falls back to the BCB/SGS public endpoint.
    """
    cache = RAW_DIR / "cdi.csv"
    if cache.exists():
        s = _load_series_cache(cache, "CDI")
        if s is not None:
            return s

    s = None
    if token:
        try:
            s = _cdi_from_anbima(start_date, end_date, token)
            log.info("CDI fetched from ANBIMA (%d obs)", len(s))
        except Exception as exc:
            log.warning("ANBIMA CDI failed (%s) — falling back to BCB/SGS", exc)

    if s is None or s.empty:
        s = _cdi_from_bcb(start_date, end_date)
        log.info("CDI fetched from BCB/SGS (%d obs)", len(s))

    _save_series_cache(s, cache)
    return s


def _cdi_from_anbima(start_date: str, end_date: str, token: str) -> pd.Series:
    """CDI from ANBIMA Preços e Índices v1 — /cdi endpoint."""
    resp = requests.get(
        f"{BASE_URL}/feed/precos-indices/v1/indices-mais-recentes/cdi",
        headers=_anbima_headers(token),
        params={"data_inicio": start_date, "data_fim": end_date},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"ANBIMA CDI {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    records = data if isinstance(data, list) else data.get("data", data.get("indices", []))
    if not records:
        raise ValueError("Empty CDI response from ANBIMA")

    df = pd.DataFrame(records)
    date_col = _first_col(df, ["data_referencia", "data", "date"])
    val_col = _first_col(df, ["taxa", "cdi", "valor", "value", "indice"])
    if date_col is None or val_col is None:
        raise ValueError(f"Unexpected ANBIMA CDI columns: {df.columns.tolist()}")

    s = pd.to_numeric(df[val_col], errors="coerce") / 100  # % → decimal
    s.index = pd.to_datetime(df[date_col])
    s.name = "CDI"
    return s.sort_index().dropna()


def _cdi_from_bcb(start_date: str, end_date: str) -> pd.Series:
    """
    CDI from BCB/SGS series 12.
    Values are published as % per day (e.g. 0.0489 → 0.000489 as decimal).
    No authentication required.
    """
    url = (
        "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
        f"?formato=json"
        f"&dataInicial={_bcb_fmt(start_date)}"
        f"&dataFinal={_bcb_fmt(end_date)}"
    )
    resp = requests.get(url, timeout=30)
    if not resp.ok:
        raise RuntimeError(f"BCB CDI {resp.status_code}: {resp.text[:300]}")

    df = pd.DataFrame(resp.json())
    s = pd.to_numeric(df["valor"], errors="coerce") / 100
    s.index = pd.to_datetime(df["data"], dayfirst=True)
    s.name = "CDI"
    return s.sort_index().dropna()


def _bcb_fmt(date_str: str) -> str:
    """YYYY-MM-DD → DD/MM/YYYY (BCB API format)."""
    return datetime.strptime(date_str, "%Y-%m-%d").strftime("%d/%m/%Y")


# ---------------------------------------------------------------------------
# IMA-B
# ---------------------------------------------------------------------------

def fetch_imab(start_date: str, end_date: str, token: Optional[str] = None) -> pd.Series:
    """
    Return daily IMA-B returns (decimal).

    Priority:
      1. ANBIMA Preços e Índices v1 (requires active API subscription)
      2. iShares IMAB11.SA ETF via Yahoo Finance (tracks full IMA-B index)

    The ETF proxy introduces small tracking error but correlates >0.99 with
    the official index for style-analysis purposes.
    Caches to data/raw/imab.csv.
    """
    cache = RAW_DIR / "imab.csv"
    if cache.exists():
        s = _load_series_cache(cache, "IMA-B")
        if s is not None:
            return s

    # 1 — Try ANBIMA API
    if token:
        try:
            s = _imab_from_anbima(start_date, end_date, token)
            if not s.empty:
                _save_series_cache(s, cache)
                log.info("IMA-B from ANBIMA: %d obs", len(s))
                return s
        except Exception as exc:
            log.warning("ANBIMA IMA-B failed (%s) — falling back to IMAB11.SA ETF.", exc)

    # 2 — Fallback: iShares IMA-B ETF on B3
    s = _imab_from_yfinance(start_date, end_date)
    _save_series_cache(s, cache)
    return s


def _imab_from_anbima(start_date: str, end_date: str, token: str) -> pd.Series:
    """IMA-B index levels from ANBIMA Preços e Índices v1, converted to returns."""
    records = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=364), end)
        try:
            resp = requests.get(
                f"{BASE_URL}/feed/precos-indices/v1/ima/resultado-diario",
                headers=_anbima_headers(token),
                params={
                    "data_inicio": current.strftime("%Y-%m-%d"),
                    "data_fim": chunk_end.strftime("%Y-%m-%d"),
                    "indice": "IMA-B",
                },
                timeout=30,
            )
            if resp.status_code == 429:
                time.sleep(60)
                continue
            if not resp.ok:
                raise RuntimeError(f"ANBIMA IMA-B {resp.status_code}: {resp.text[:200]}")

            data = resp.json()
            chunk = data if isinstance(data, list) else data.get("ima", data.get("data", []))
            records.extend(chunk)

        except requests.RequestException as exc:
            log.warning("IMA-B chunk %s–%s network error: %s", current.date(), chunk_end.date(), exc)

        current = chunk_end + timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    if not records:
        raise RuntimeError("ANBIMA returned no IMA-B records.")

    df = pd.DataFrame(records)
    date_col = _first_col(df, ["data_referencia", "data_competencia", "data", "date"])
    val_col = _first_col(df, ["numero_indice", "valor", "indice_valor", "num_indice"])
    if date_col is None or val_col is None:
        raise ValueError(f"Unexpected IMA-B columns: {df.columns.tolist()}")

    levels = pd.to_numeric(df[val_col], errors="coerce")
    levels.index = pd.to_datetime(df[date_col])
    s = levels.sort_index().dropna().pct_change().dropna()
    s.name = "IMA-B"
    return s


def _imab_from_yfinance(start_date: str, end_date: str) -> pd.Series:
    """
    IMA-B proxy via iShares IMAB11.SA ETF (B3).
    IMAB11 tracks the full ANBIMA IMA-B index (all NTN-B maturities).
    Correlation with official IMA-B > 0.99 over rolling 12-month windows.
    """
    log.info("Fetching IMA-B proxy (IMAB11.SA) from Yahoo Finance…")
    df = yf.download("IMAB11.SA", start=start_date, end=end_date,
                     auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError("Yahoo Finance returned no data for IMAB11.SA")

    close = df["Close"]
    if hasattr(close, "squeeze"):
        close = close.squeeze()

    s = close.pct_change().dropna()
    s.name = "IMA-B"
    s.index = pd.to_datetime(s.index)
    log.info("IMA-B (IMAB11.SA proxy): %d daily return obs", len(s))
    return s


# ---------------------------------------------------------------------------
# Ibovespa
# ---------------------------------------------------------------------------

def fetch_ibovespa(start_date: str, end_date: str) -> pd.Series:
    """
    Return daily Ibovespa returns (decimal) via Yahoo Finance.
    Ticker: ^BVSP. Caches to data/raw/ibovespa.csv.
    """
    cache = RAW_DIR / "ibovespa.csv"
    if cache.exists():
        s = _load_series_cache(cache, "Ibovespa")
        if s is not None:
            return s

    log.info("Fetching Ibovespa (^BVSP) from Yahoo Finance…")
    df = yf.download("^BVSP", start=start_date, end=end_date, auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError("Yahoo Finance returned no data for ^BVSP")

    close = df["Close"]
    if hasattr(close, "squeeze"):
        close = close.squeeze()

    s = close.pct_change().dropna()
    s.name = "Ibovespa"
    s.index = pd.to_datetime(s.index)

    _save_series_cache(s, cache)
    log.info("Ibovespa: %d daily return obs", len(s))
    return s


# ---------------------------------------------------------------------------
# BRL/USD
# ---------------------------------------------------------------------------

def fetch_brl_usd(start_date: str, end_date: str) -> pd.Series:
    """
    Return daily BRL/USD returns (decimal) via Yahoo Finance.
    Ticker: BRL=X (USD per BRL). A positive return means BRL appreciation.
    Caches to data/raw/brl_usd.csv.
    """
    cache = RAW_DIR / "brl_usd.csv"
    if cache.exists():
        s = _load_series_cache(cache, "BRL_USD")
        if s is not None:
            return s

    log.info("Fetching BRL/USD (BRL=X) from Yahoo Finance…")
    df = yf.download("BRL=X", start=start_date, end=end_date, auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError("Yahoo Finance returned no data for BRL=X")

    close = df["Close"]
    if hasattr(close, "squeeze"):
        close = close.squeeze()

    s = close.pct_change().dropna()
    s.name = "BRL_USD"
    s.index = pd.to_datetime(s.index)

    _save_series_cache(s, cache)
    log.info("BRL/USD: %d daily return obs", len(s))
    return s


# ---------------------------------------------------------------------------
# Master assembly
# ---------------------------------------------------------------------------

def build_factors_matrix(
    start_date: str,
    end_date: str,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assemble CDI, IMA-B, Ibovespa, and BRL/USD into a single aligned DataFrame.
    Columns: ['CDI', 'IMA-B', 'Ibovespa', 'BRL_USD']
    Rows: business dates where at least one factor has data.
    Caches to data/raw/factors.csv.
    """
    cache = RAW_DIR / "factors.csv"
    if cache.exists():
        log.info("Factors matrix loaded from cache (%s)", cache)
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df

    log.info("Building factors matrix %s → %s", start_date, end_date)

    cdi = fetch_cdi(start_date, end_date, token)
    ibov = fetch_ibovespa(start_date, end_date)
    brl = fetch_brl_usd(start_date, end_date)

    # IMA-B: ANBIMA API preferred, IMAB11.SA ETF fallback (no token needed)
    try:
        imab = fetch_imab(start_date, end_date, token)
    except Exception as exc:
        log.error("IMA-B could not be fetched: %s", exc)
        imab = pd.Series(name="IMA-B", dtype=float)

    df = pd.concat([cdi, imab, ibov, brl], axis=1)
    df.index.name = "date"
    df.to_csv(cache)

    log.info(
        "Factors matrix saved: %d rows × %d cols → %s",
        len(df), df.shape[1], cache,
    )
    log.info("NaN counts:\n%s", df.isna().sum().to_string())
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Return the first column whose lower-case name matches a candidate."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]
    return None


def _load_series_cache(path: Path, name: str) -> Optional[pd.Series]:
    """Load a single-column CSV as a named Series. Returns None if empty/corrupt."""
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df.empty:
            return None
        s = df.iloc[:, 0].dropna()
        s.name = name
        s.index = pd.to_datetime(s.index)
        log.info("%s loaded from cache (%d obs) [%s]", name, len(s), path.name)
        return s
    except Exception as exc:
        log.warning("Cache read failed for %s: %s — re-fetching", path, exc)
        return None


def _save_series_cache(s: pd.Series, path: Path) -> None:
    s.to_frame().to_csv(path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from fetch_anbima import get_access_token  # noqa: E402

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")

    log.info("Date range: %s → %s", start_date, end_date)

    try:
        token = get_access_token()
    except Exception as exc:
        log.warning("Could not get ANBIMA token (%s) — IMA-B will be skipped", exc)
        token = None

    factors = build_factors_matrix(start_date, end_date, token)
    print("\nFactors matrix (last 5 rows):")
    print(factors.tail().to_string())
    print("\nShape:", factors.shape)
    print("NaN counts:")
    print(factors.isna().sum().to_string())
