"""
Fetch Brazilian Multimercado fund data.

Primary source: CVM open data (free, no auth required)
  - Fund registry : https://dados.cvm.gov.br/dados/FI/CAD/DADOS/registro_fundo_classe.zip
  - Daily quotas  : https://dados.cvm.gov.br/dados/FI/DOC/INF_DIARIO/DADOS/inf_diario_fi_YYYYMM.zip

Upgrade path: ANBIMA Fundos v2 API (requires active plan subscription)
  - Activate at: developers.anbima.com.br → My Apps → API Products → "Fundos Abertos v2"
  - Once activated, the ANBIMA functions below will work automatically.

Note on "RJ domicile": CVM does not publish geographic location of funds,
so the CVM path selects the top MAX_FUNDS active Multimercado funds by AUM instead.
"""

import base64
import io
import logging
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://api.anbima.com.br"
CVM_BASE = "https://dados.cvm.gov.br/dados/FI"
RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY = 0.4
MAX_FUNDS: int = 50  # top N funds by AUM when using CVM source

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ANBIMA Auth (Sensedia gateway)
# ---------------------------------------------------------------------------

_token_cache: Dict[str, object] = {}


def get_access_token() -> str:
    """OAuth2 client-credentials flow. Caches token until near-expiry."""
    now = datetime.utcnow()
    if _token_cache.get("expires_at") and now < _token_cache["expires_at"]:
        return _token_cache["access_token"]  # type: ignore[return-value]

    client_id = os.environ.get("ANBIMA_CLIENT_ID", "")
    client_secret = os.environ.get("ANBIMA_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        raise EnvironmentError("ANBIMA_CLIENT_ID and ANBIMA_CLIENT_SECRET must be set in .env")

    creds = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        f"{BASE_URL}/oauth/access-token",
        headers={
            "Authorization": f"Basic {creds}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    if not resp.ok:
        raise RuntimeError(f"Token request failed {resp.status_code}: {resp.text[:300]}")

    payload = resp.json()
    token = payload.get("access_token") or payload.get("token")
    if not token:
        raise RuntimeError(f"No access_token in response: {payload}")

    expires_in = int(payload.get("expires_in", 3600))
    _token_cache["access_token"] = token
    _token_cache["expires_at"] = now + timedelta(seconds=expires_in - 60)
    log.info("ANBIMA token acquired (expires in %ds)", expires_in)
    return token


def _anbima_headers(token: str) -> dict:
    # Sensedia gateway: access_token header (not Authorization: Bearer) + client_id
    return {
        "access_token": token,
        "client_id": os.environ.get("ANBIMA_CLIENT_ID", ""),
        "Accept": "application/json",
    }


def check_api_access(token: str) -> None:
    """
    Probe the Fundos v2 endpoint and log a diagnostic.
    403 → subscribe at developers.anbima.com.br → My Apps → API Products.
    """
    resp = requests.get(
        f"{BASE_URL}/feed/fundos/v2/fundos",
        headers=_anbima_headers(token),
        params={"pagina": 1, "quantidade": 1},
        timeout=15,
    )
    if resp.ok:
        log.info("ANBIMA Fundos v2: accessible ✓")
    elif resp.status_code == 403:
        log.error(
            "ANBIMA Fundos v2: 403 — subscription not active.\n"
            "  Fix: developers.anbima.com.br → My Apps → <app> → API Products "
            "→ subscribe 'Fundos Abertos v2'.\n"
            "  Falling back to CVM open data."
        )
    else:
        log.warning("ANBIMA Fundos v2: unexpected %d — %s", resp.status_code, resp.text[:150])


# ---------------------------------------------------------------------------
# CNPJ helpers
# ---------------------------------------------------------------------------

def _fmt_cnpj(raw: str) -> str:
    """Normalize any CNPJ representation to the canonical XX.XXX.XXX/XXXX-XX format."""
    digits = "".join(c for c in str(raw) if c.isdigit()).zfill(14)
    return f"{digits[:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:14]}"


def _cnpj_col(df: pd.DataFrame) -> str:
    """Return the CNPJ column name from a CVM quota DataFrame (old or new format)."""
    for name in ("CNPJ_FUNDO_CLASSE", "CNPJ_FUNDO"):
        if name in df.columns:
            return name
    raise KeyError(f"No CNPJ column found in {df.columns.tolist()}")


# ---------------------------------------------------------------------------
# CVM — fund list
# ---------------------------------------------------------------------------

def _cvm_load_fund_list(max_funds: int = MAX_FUNDS) -> pd.DataFrame:
    """
    Download CVM fund registry (ICVM 175/2022) and return active Multimercado funds.
    Selects top max_funds by AUM since CVM does not carry geographic location data.
    """
    cache = RAW_DIR / "cvm_fund_list.csv"
    if cache.exists():
        log.info("CVM fund list loaded from cache (%s)", cache)
        return pd.read_csv(cache, dtype=str)

    log.info("Downloading CVM fund registry (registro_fundo_classe.zip)…")
    r = requests.get(
        f"{CVM_BASE}/CAD/DADOS/registro_fundo_classe.zip", timeout=120
    )
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open("registro_classe.csv") as f:
            df = pd.read_csv(f, sep=";", encoding="latin-1", dtype=str, low_memory=False)

    # Active Multimercado only
    df = df[
        (df["Classificacao"] == "Multimercado") &
        (df["Situacao"] == "Em Funcionamento Normal")
    ].copy()
    log.info("Active Multimercado funds: %d", len(df))

    # Sort by AUM, take top N
    df["aum"] = pd.to_numeric(df["Patrimonio_Liquido"], errors="coerce").fillna(0)
    df = df.nlargest(max_funds, "aum").reset_index(drop=True)

    # Normalize CNPJ to canonical format
    df["cnpj"] = df["CNPJ_Classe"].apply(_fmt_cnpj)

    out = df[["cnpj", "Denominacao_Social", "Classificacao_Anbima", "aum"]].rename(
        columns={"Denominacao_Social": "nome", "Classificacao_Anbima": "classe_anbima"}
    )
    out.to_csv(cache, index=False)
    log.info("Saved top-%d Multimercado funds to %s", len(out), cache)
    return out


# ---------------------------------------------------------------------------
# CVM — monthly quota batch download
# ---------------------------------------------------------------------------

def _cvm_month_range(start_date: str, end_date: str) -> List[str]:
    """Return list of YYYYMM strings covering start_date..end_date inclusive."""
    start = datetime.strptime(start_date, "%Y-%m-%d").replace(day=1)
    end = datetime.strptime(end_date, "%Y-%m-%d")
    months = []
    cur = start
    while cur <= end:
        months.append(cur.strftime("%Y%m"))
        # advance one month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)
    return months


def _cvm_fetch_one_month(month_str: str, target_cnpjs: Set[str]) -> pd.DataFrame:
    """
    Download inf_diario_fi_YYYYMM.zip, filter to target CNPJs, and return
    a DataFrame with columns [cnpj, date, cota].
    Returns empty DataFrame if the month file is unavailable.
    """
    url = f"{CVM_BASE}/DOC/INF_DIARIO/DADOS/inf_diario_fi_{month_str}.zip"
    try:
        r = requests.get(url, timeout=120)
    except requests.RequestException as exc:
        log.warning("Network error downloading %s: %s", url, exc)
        return pd.DataFrame()

    if r.status_code == 404:
        log.debug("No quota file for month %s (404)", month_str)
        return pd.DataFrame()
    if not r.ok:
        log.warning("CVM quota %s → %d", month_str, r.status_code)
        return pd.DataFrame()

    chunks: List[pd.DataFrame] = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            for chunk in pd.read_csv(
                f,
                sep=";",
                encoding="latin-1",
                dtype=str,
                low_memory=False,
                chunksize=50_000,
            ):
                try:
                    col = _cnpj_col(chunk)
                except KeyError:
                    continue

                filtered = chunk[chunk[col].isin(target_cnpjs)]
                if not filtered.empty:
                    chunks.append(
                        filtered[[col, "DT_COMPTC", "VL_QUOTA"]].rename(
                            columns={col: "cnpj", "DT_COMPTC": "date", "VL_QUOTA": "cota"}
                        )
                    )

    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def _cvm_build_quota_history(
    target_cnpjs: Set[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Build full quota history for target_cnpjs by downloading monthly CVM files.
    Caches processed months individually to avoid re-downloading on partial runs.
    Returns long-format DataFrame with columns [cnpj, date, cota].
    """
    months = _cvm_month_range(start_date, end_date)
    log.info("Building CVM quota history for %d funds over %d months…", len(target_cnpjs), len(months))

    all_parts: List[pd.DataFrame] = []
    for month in months:
        month_cache = RAW_DIR / f"cvm_quotas_{month}.parquet"
        if month_cache.exists():
            part = pd.read_parquet(month_cache)
            log.debug("Month %s: loaded from cache (%d rows)", month, len(part))
        else:
            log.info("  Downloading month %s…", month)
            part = _cvm_fetch_one_month(month, target_cnpjs)
            if not part.empty:
                part.to_parquet(month_cache, index=False)
            time.sleep(0.2)

        if not part.empty:
            all_parts.append(part)

    if not all_parts:
        raise RuntimeError("CVM returned no quota data for any month in the date range.")

    df = pd.concat(all_parts, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["cota"] = pd.to_numeric(df["cota"], errors="coerce")
    return df.dropna(subset=["date", "cota"])


# ---------------------------------------------------------------------------
# ANBIMA — fund list (kept for future use when subscription is active)
# ---------------------------------------------------------------------------

def _anbima_fetch_fund_list(token: str) -> pd.DataFrame:
    """Fetch Multimercado/RJ fund list from ANBIMA Fundos v2."""
    funds: List[dict] = []
    page = 1
    while True:
        resp = requests.get(
            f"{BASE_URL}/feed/fundos/v2/fundos",
            headers=_anbima_headers(token),
            params={"tipo_fundo": "Multimercado", "domicilio": "RJ",
                    "pagina": page, "quantidade": 100},
            timeout=30,
        )
        if resp.status_code in (404, 403):
            raise RuntimeError(f"ANBIMA Fundos v2 returned {resp.status_code}: {resp.text[:200]}")
        if resp.status_code == 429:
            time.sleep(60)
            continue
        resp.raise_for_status()

        data = resp.json()
        items = data if isinstance(data, list) else data.get("fundos", data.get("data", []))
        if not items:
            break
        funds.extend(items)
        if len(items) < 100:
            break
        page += 1
        time.sleep(REQUEST_DELAY)

    if not funds:
        raise RuntimeError("ANBIMA returned 0 Multimercado/RJ funds.")
    return pd.DataFrame(funds)


# ---------------------------------------------------------------------------
# Public API — fund list (ANBIMA with CVM fallback)
# ---------------------------------------------------------------------------

def fetch_multimercado_funds_rj(token: Optional[str] = None) -> pd.DataFrame:
    """
    Return a DataFrame of Multimercado funds.
    Tries ANBIMA Fundos v2 first; falls back to CVM open data on 403/error.

    Columns (CVM path): cnpj, nome, classe_anbima, aum
    Columns (ANBIMA path): whatever the API returns (includes cnpj_fundo, etc.)
    """
    cache = RAW_DIR / "funds_list.csv"
    if cache.exists():
        log.info("Fund list loaded from cache (%s)", cache)
        return pd.read_csv(cache, dtype=str)

    if token:
        try:
            log.info("Trying ANBIMA Fundos v2…")
            df = _anbima_fetch_fund_list(token)
            df.to_csv(cache, index=False)
            log.info("ANBIMA: found %d funds", len(df))
            return df
        except RuntimeError as exc:
            log.warning("ANBIMA fund list failed (%s) — falling back to CVM.", exc)

    df = _cvm_load_fund_list(max_funds=MAX_FUNDS)
    df.to_csv(cache, index=False)
    return df


# ---------------------------------------------------------------------------
# Public API — quota history (ANBIMA per-fund with CVM batch fallback)
# ---------------------------------------------------------------------------

def fetch_fund_quota_history(
    cnpj: str,
    token: Optional[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch daily quota history for a single fund. Returns [date, cnpj, cota].
    This thin wrapper exists for callers that want per-fund data.
    For bulk use, call build_returns_matrix() which uses the CVM batch path.
    """
    safe = cnpj.replace("/", "_").replace(".", "_").replace("-", "_")
    cache = RAW_DIR / f"quotas_{safe}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])

    if token:
        try:
            df = _anbima_fetch_single_fund(cnpj, token, start_date, end_date)
            if not df.empty:
                df.to_csv(cache, index=False)
                return df
        except Exception as exc:
            log.warning("ANBIMA quota for %s failed (%s) — using CVM batch.", cnpj, exc)

    # CVM batch for this single CNPJ
    quota_df = _cvm_build_quota_history({cnpj}, start_date, end_date)
    quota_df.to_csv(cache, index=False)
    return quota_df


def _anbima_fetch_single_fund(
    cnpj: str, token: str, start_date: str, end_date: str
) -> pd.DataFrame:
    """Per-fund quota history from ANBIMA (chunked by year)."""
    records: List[dict] = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    cur = start

    while cur < end:
        chunk_end = min(cur + timedelta(days=364), end)
        resp = requests.get(
            f"{BASE_URL}/feed/fundos/v2/fundos/{cnpj}/informes-diarios",
            headers=_anbima_headers(token),
            params={"data_inicio": cur.strftime("%Y-%m-%d"),
                    "data_fim": chunk_end.strftime("%Y-%m-%d")},
            timeout=30,
        )
        if resp.status_code == 404:
            return pd.DataFrame()
        if resp.status_code in (401, 403):
            raise RuntimeError(f"ANBIMA quota {resp.status_code}: {resp.text[:150]}")
        if resp.status_code == 429:
            time.sleep(60)
            continue
        if not resp.ok:
            log.warning("ANBIMA quota %s %s→%s %d", cnpj, cur.date(), chunk_end.date(), resp.status_code)
            cur = chunk_end + timedelta(days=1)
            continue
        data = resp.json()
        chunk = data if isinstance(data, list) else data.get("informes", data.get("data", []))
        records.extend(chunk)
        cur = chunk_end + timedelta(days=1)
        time.sleep(REQUEST_DELAY)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    date_col = next((c for c in df.columns if "data" in c.lower()), None)
    cota_col = next((c for c in df.columns if c.lower() in ("vl_quota", "vl_cota", "cota")), None)
    if not date_col or not cota_col:
        return pd.DataFrame()

    return pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "cnpj": cnpj,
        "cota": pd.to_numeric(df[cota_col], errors="coerce"),
    }).dropna()


# ---------------------------------------------------------------------------
# Returns matrix
# ---------------------------------------------------------------------------

def build_returns_matrix(
    funds_df: pd.DataFrame,
    token: Optional[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Compute daily pct returns from fund quota values.
    Returns a wide DataFrame: index=date, columns=CNPJ strings.
    Caches result to data/raw/fund_returns.csv.
    """
    cache = RAW_DIR / "fund_returns.csv"
    if cache.exists():
        log.info("Returns matrix loaded from cache (%s)", cache)
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        return df

    # Identify CNPJ column (CVM uses 'cnpj', ANBIMA uses 'cnpj_fundo' variants)
    cnpj_col = next(
        (c for c in funds_df.columns if "cnpj" in c.lower()),
        None,
    )
    if cnpj_col is None:
        raise ValueError(f"No CNPJ column in fund list. Columns: {funds_df.columns.tolist()}")

    cnpjs = set(funds_df[cnpj_col].dropna().unique())
    log.info("Building returns matrix for %d funds…", len(cnpjs))

    # Use CVM batch download (much faster than per-fund ANBIMA calls)
    try:
        all_quotas = _cvm_build_quota_history(cnpjs, start_date, end_date)
    except RuntimeError:
        log.warning("CVM batch failed — trying per-fund ANBIMA calls…")
        parts = []
        for cnpj in cnpjs:
            q = fetch_fund_quota_history(str(cnpj), token, start_date, end_date)
            if not q.empty:
                parts.append(q)
        if not parts:
            raise RuntimeError("No quota data from ANBIMA or CVM.")
        all_quotas = pd.concat(parts, ignore_index=True)

    # Pivot and compute returns
    returns: Dict[str, pd.Series] = {}
    for cnpj, group in all_quotas.groupby("cnpj"):
        series = group.set_index("date")["cota"].sort_index().astype(float)
        series = series[~series.index.duplicated(keep="last")]
        daily = series.pct_change().dropna()
        if len(daily) >= 200:
            returns[str(cnpj)] = daily

    if not returns:
        raise RuntimeError("No fund return series with ≥200 observations.")

    matrix = pd.DataFrame(returns)
    matrix.index.name = "date"
    matrix.to_csv(cache)
    log.info("Returns matrix: %d rows × %d funds → %s", len(matrix), matrix.shape[1], cache)
    return matrix


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=3 * 365)).strftime("%Y-%m-%d")
    log.info("Date range: %s → %s", start_date, end_date)

    try:
        token = get_access_token()
        check_api_access(token)
    except Exception as exc:
        log.warning("ANBIMA auth failed (%s) — running CVM-only mode.", exc)
        token = None

    funds = fetch_multimercado_funds_rj(token)
    print(f"\nFunds found: {len(funds)}")
    print("Columns:", funds.columns.tolist())
    print(funds.head(5).to_string())

    print("\nBuilding returns matrix (this downloads ~36 monthly ZIPs)…")
    returns = build_returns_matrix(funds, token, start_date, end_date)
    print(f"\nReturns matrix: {returns.shape[0]} rows × {returns.shape[1]} funds")
    print(returns.tail(3).iloc[:, :5].to_string())
