"""
Pipeline de atualização do dashboard de análise de estilo.

O que faz
---------
- Mantém a lista de fundos acompanhados (fundos.csv), editável pelo app.
- Baixa as cotas diárias da CVM (inf_diario_fi_AAAAMM.zip) para esses fundos.
- Atualiza os fatores: CDI (BCB/SGS), IMA-B (IMAB11), Ibovespa (^BVSP) e BRL/USD (BRL=X).
- Reestima a análise de estilo de Sharpe (1992) para todos os fundos.

Onde guarda
-----------
Tudo fica em DATA_DIR (variável de ambiente). No Railway, aponte DATA_DIR para um
volume (ex.: /data) para que os fundos adicionados sobrevivam a novos deploys.
Na primeira execução a pasta é semeada com os dados que vêm no repositório (data/raw).

As tarefas rodam numa única thread de fundo, uma de cada vez.
"""

from __future__ import annotations

import io
import json
import logging
import os
import queue
import shutil
import threading
import time
import unicodedata
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import requests

from analysis.regression import FACTOR_LABELS, analyze_fund

log = logging.getLogger("pipeline")

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

PROJ_DIR = Path(__file__).parent
SEED_DIR = PROJ_DIR / "data" / "raw"
STORE = Path(os.getenv("DATA_DIR") or (PROJ_DIR / "data" / "store"))
COTAS_DIR = STORE / "cotas"

CVM_BASE = "https://dados.cvm.gov.br/dados/FI"
JANELA_ANOS = 3
MESES_RECENTES = 2          # meses sempre baixados de novo (a CVM revisa o mês corrente)
INTERVALO_AUTO_H = 24       # atualização automática a cada 24 h
REGISTRO_VALIDADE_D = 7     # cadastro da CVM baixado de novo a cada 7 dias
FATORES = ["CDI", "IMA-B", "Ibovespa", "BRL_USD"]

ARQ_FUNDOS = "fundos.csv"
ARQ_FATORES = "factors.csv"
ARQ_RETORNOS = "fund_returns.csv"
ARQ_RESULTADOS = "style_results.csv"
ARQ_STATUS = "status.json"
ARQ_REGISTRO = "registro_cvm.csv"

HTTP = requests.Session()
HTTP.headers["User-Agent"] = "style-analysis-dashboard/1.0"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def fmt_cnpj(raw: str) -> str:
    d = "".join(c for c in str(raw) if c.isdigit())
    if len(d) > 14 or len(d) < 11:
        raise ValueError("CNPJ inválido")
    d = d.zfill(14)
    return f"{d[:2]}.{d[2:5]}.{d[5:8]}/{d[8:12]}-{d[12:]}"


def cnpj_valido(raw: str) -> bool:
    d = "".join(c for c in str(raw) if c.isdigit()).zfill(14)
    if len(d) != 14 or d == d[0] * 14:
        return False
    def dv(base: str, pesos: List[int]) -> str:
        s = sum(int(a) * b for a, b in zip(base, pesos)) % 11
        return "0" if s < 2 else str(11 - s)
    p1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    p2 = [6] + p1
    return d[12] == dv(d[:12], p1) and d[13] == dv(d[:13], p2)


def _sem_acento(txt: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", str(txt)) if not unicodedata.combining(c)).lower()


def _agora() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _salvar_csv(df: pd.DataFrame, nome: str, **kw) -> None:
    """Grava de forma atômica (arquivo temporário + rename)."""
    destino = STORE / nome
    tmp = destino.with_suffix(destino.suffix + ".tmp")
    df.to_csv(tmp, **kw)
    tmp.replace(destino)


def _janela() -> tuple[pd.Timestamp, pd.Timestamp]:
    fim = pd.Timestamp.today().normalize()
    inicio = fim - pd.DateOffset(years=JANELA_ANOS)
    return inicio, fim


def _meses(inicio: pd.Timestamp, fim: pd.Timestamp) -> List[str]:
    return [p.strftime("%Y%m") for p in pd.period_range(inicio, fim, freq="M")]


# ---------------------------------------------------------------------------
# Armazenamento
# ---------------------------------------------------------------------------

_lock_arquivos = threading.RLock()


def inicializar() -> None:
    """Cria DATA_DIR e semeia com os dados do repositório na primeira vez."""
    STORE.mkdir(parents=True, exist_ok=True)
    COTAS_DIR.mkdir(parents=True, exist_ok=True)
    with _lock_arquivos:
        if not (STORE / ARQ_FUNDOS).exists():
            log.info("Semeando %s com os dados do repositório", STORE)
            seed = pd.read_csv(SEED_DIR / "funds_list.csv", dtype=str)
            fundos = pd.DataFrame({
                "cnpj": seed["cnpj"],
                "nome": seed["nome"],
                "classe": seed.get("classe_anbima", ""),
                "adicionado_em": "",
            })
            _salvar_csv(fundos, ARQ_FUNDOS, index=False)
            for nome in (ARQ_FATORES, ARQ_RETORNOS, ARQ_RESULTADOS):
                if (SEED_DIR / nome).exists():
                    shutil.copy(SEED_DIR / nome, STORE / nome)
            for p in SEED_DIR.glob("cvm_quotas_*.parquet"):
                mes = p.stem.split("_")[-1]
                shutil.copy(p, COTAS_DIR / f"{mes}.parquet")
        if not (STORE / ARQ_STATUS).exists():
            _gravar_status({"ultima_atualizacao": None, "dados_ate": _dados_ate_arquivo()})


def carregar_fundos() -> pd.DataFrame:
    with _lock_arquivos:
        df = pd.read_csv(STORE / ARQ_FUNDOS, dtype=str).fillna("")
    return df


def carregar_resultados() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(style_results, fund_returns, factors) para o app."""
    with _lock_arquivos:
        style = pd.read_csv(STORE / ARQ_RESULTADOS)
        ret = pd.read_csv(STORE / ARQ_RETORNOS, index_col=0, parse_dates=True)
        fat = pd.read_csv(STORE / ARQ_FATORES, index_col=0, parse_dates=True)
    style["nome"] = style["nome"].fillna("").astype(str)
    return style, ret, fat


def _dados_ate_arquivo() -> Optional[str]:
    try:
        ret = pd.read_csv(STORE / ARQ_RETORNOS, usecols=[0])
        return str(pd.to_datetime(ret.iloc[:, 0]).max().date())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Status e fila de tarefas
# ---------------------------------------------------------------------------

_lock_status = threading.Lock()
_tarefa: Dict = {"estado": "ociosa", "tipo": None, "etapa": "", "progresso": 0,
                 "inicio": None, "fim": None, "erro": None, "mensagem": None}
_fila: "queue.Queue[tuple]" = queue.Queue()
_pendentes: List[str] = []
_ao_concluir: List[Callable[[], None]] = []


def _gravar_status(extra: Dict) -> None:
    caminho = STORE / ARQ_STATUS
    atual = {}
    if caminho.exists():
        try:
            atual = json.loads(caminho.read_text())
        except Exception:
            atual = {}
    atual.update(extra)
    tmp = caminho.with_suffix(".tmp")
    tmp.write_text(json.dumps(atual, ensure_ascii=False, indent=1))
    tmp.replace(caminho)


def status() -> Dict:
    try:
        persistido = json.loads((STORE / ARQ_STATUS).read_text())
    except Exception:
        persistido = {}
    with _lock_status:
        t = dict(_tarefa)
        pend = list(_pendentes)
    return {**persistido, "tarefa": t, "fila": pend}


def _etapa(texto: str, progresso: Optional[float] = None) -> None:
    with _lock_status:
        _tarefa["etapa"] = texto
        if progresso is not None:
            _tarefa["progresso"] = int(max(0, min(100, progresso)))
    log.info(texto)


def ao_concluir(fn: Callable[[], None]) -> None:
    """Registra uma função chamada depois de cada tarefa (ex.: recarregar o app)."""
    _ao_concluir.append(fn)


def enfileirar(tipo: str, arg: Optional[str] = None) -> bool:
    """tipo: 'atualizar' | 'adicionar' | 'recalcular'. Retorna False se já está na fila."""
    chave = f"{tipo}:{arg or ''}"
    with _lock_status:
        if chave in _pendentes or (_tarefa["estado"] == "rodando" and _tarefa.get("chave") == chave):
            return False
        _pendentes.append(chave)
    _fila.put((tipo, arg, chave))
    return True


def _trabalhador() -> None:
    while True:
        tipo, arg, chave = _fila.get()
        with _lock_status:
            if chave in _pendentes:
                _pendentes.remove(chave)
            _tarefa.update({"estado": "rodando", "tipo": tipo, "chave": chave, "etapa": "Iniciando…",
                            "progresso": 0, "inicio": _agora(), "fim": None, "erro": None, "mensagem": None})
        try:
            if tipo == "atualizar":
                msg = _atualizar()
            elif tipo == "adicionar":
                msg = _adicionar(arg)
            else:
                msg = _recalcular()
            with _lock_status:
                _tarefa.update({"estado": "concluida", "etapa": "Concluído", "progresso": 100,
                                "fim": _agora(), "mensagem": msg})
        except Exception as exc:  # noqa: BLE001
            log.exception("Tarefa %s falhou", chave)
            with _lock_status:
                _tarefa.update({"estado": "erro", "fim": _agora(), "erro": str(exc)})
        finally:
            for fn in _ao_concluir:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    log.exception("Falha ao recarregar dados")


def _agendador() -> None:
    while True:
        try:
            st = status()
            ultima = st.get("ultima_atualizacao")
            velha = (not ultima) or (datetime.now() - datetime.fromisoformat(ultima) > timedelta(hours=INTERVALO_AUTO_H))
            if velha and st["tarefa"]["estado"] != "rodando" and not st["fila"]:
                log.info("Atualização automática")
                enfileirar("atualizar")
        except Exception:  # noqa: BLE001
            log.exception("Agendador")
        time.sleep(30 * 60)


_iniciado = False


def iniciar(auto: bool) -> None:
    """Inicia a thread de tarefas e, se auto=True, a atualização diária."""
    global _iniciado
    if _iniciado:
        return
    _iniciado = True
    inicializar()
    threading.Thread(target=_trabalhador, daemon=True, name="pipeline").start()
    if auto:
        threading.Thread(target=_agendador, daemon=True, name="agendador").start()


# ---------------------------------------------------------------------------
# CVM: cadastro de fundos
# ---------------------------------------------------------------------------

def _coluna(df: pd.DataFrame, opcoes: List[str]) -> Optional[str]:
    baixo = {c.lower(): c for c in df.columns}
    for o in opcoes:
        if o.lower() in baixo:
            return baixo[o.lower()]
    return None


def _registro() -> pd.DataFrame:
    """
    Cadastro da CVM (classes e fundos), reduzido às colunas necessárias.
    Colunas: cnpj, nome, classificacao, classe_anbima, situacao, cnpj_fundo
    """
    caminho = STORE / ARQ_REGISTRO
    if caminho.exists() and time.time() - caminho.stat().st_mtime < REGISTRO_VALIDADE_D * 86400:
        return pd.read_csv(caminho, dtype=str).fillna("")

    log.info("Baixando cadastro da CVM…")
    r = HTTP.get(f"{CVM_BASE}/CAD/DADOS/registro_fundo_classe.zip", timeout=180)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        nomes = z.namelist()
        def ler(parte: str) -> pd.DataFrame:
            arq = next(n for n in nomes if parte in n.lower())
            with z.open(arq) as f:
                return pd.read_csv(f, sep=";", encoding="latin-1", dtype=str, low_memory=False).fillna("")
        classes = ler("registro_classe")
        fundos = ler("registro_fundo")

    id_c = _coluna(classes, ["ID_Registro_Fundo"])
    id_f = _coluna(fundos, ["ID_Registro_Fundo"])
    cnpj_f = _coluna(fundos, ["CNPJ_Fundo"])
    mapa = {}
    if id_c and id_f and cnpj_f:
        mapa = dict(zip(fundos[id_f], fundos[cnpj_f]))

    out = pd.DataFrame({
        "cnpj": classes[_coluna(classes, ["CNPJ_Classe"])].map(_fmt_seguro),
        "nome": classes[_coluna(classes, ["Denominacao_Social"])],
        "classificacao": classes.get(_coluna(classes, ["Classificacao"]) or "", ""),
        "classe_anbima": classes.get(_coluna(classes, ["Classificacao_Anbima"]) or "", ""),
        "situacao": classes.get(_coluna(classes, ["Situacao"]) or "", ""),
        "cnpj_fundo": classes[id_c].map(mapa).map(_fmt_seguro) if id_c else "",
    })
    out = out[out["cnpj"] != ""]

    # fundos ainda não adaptados à RCVM 175 (sem classe): o CNPJ do fundo é o das cotas
    if cnpj_f:
        f_cnpj = fundos[cnpj_f].map(_fmt_seguro)
        ja = set(out["cnpj"]) | set(out["cnpj_fundo"])
        sem_classe = fundos[(f_cnpj != "") & ~f_cnpj.isin(ja)]
        if not sem_classe.empty:
            out = pd.concat([out, pd.DataFrame({
                "cnpj": sem_classe[cnpj_f].map(_fmt_seguro),
                "nome": sem_classe.get(_coluna(fundos, ["Denominacao_Social"]) or "", ""),
                "classificacao": sem_classe.get(_coluna(fundos, ["Tipo_Fundo"]) or "", ""),
                "classe_anbima": "",
                "situacao": sem_classe.get(_coluna(fundos, ["Situacao"]) or "", ""),
                "cnpj_fundo": sem_classe[cnpj_f].map(_fmt_seguro),
            })], ignore_index=True)

    out = out.drop_duplicates("cnpj", keep="last")
    tmp = caminho.with_suffix(".tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(caminho)
    return out


def _fmt_seguro(x: str) -> str:
    try:
        return fmt_cnpj(x)
    except Exception:
        return ""


def buscar(termo: str, limite: int = 15) -> List[Dict]:
    """Busca no cadastro da CVM por CNPJ (do fundo ou da classe) ou por parte do nome."""
    termo = (termo or "").strip()
    if len(termo) < 3:
        return []
    reg = _registro()
    digitos = "".join(c for c in termo if c.isdigit())
    so_cnpj = all(c.isdigit() or c in "./- " for c in termo)
    if so_cnpj and 11 <= len(digitos) <= 14:
        c = fmt_cnpj(digitos)
        achados = reg[(reg["cnpj"] == c) | (reg["cnpj_fundo"] == c)]
    else:
        palavras = _sem_acento(termo).split()
        nomes = reg["nome"].map(_sem_acento)
        mask = pd.Series(True, index=reg.index)
        for p in palavras:
            mask &= nomes.str.contains(p, regex=False)
        achados = reg[mask]
        ativo = achados["situacao"].str.contains("Funcionamento", case=False)
        achados = pd.concat([achados[ativo], achados[~ativo]])
    return achados.head(limite).to_dict(orient="records")


# ---------------------------------------------------------------------------
# CVM: cotas diárias
# ---------------------------------------------------------------------------

_COLS_DIARIO = {"CNPJ_FUNDO_CLASSE", "CNPJ_FUNDO", "DT_COMPTC", "VL_QUOTA", "ID_SUBCLASSE"}


def _baixar_mes(mes: str, alvos: Set[str]) -> Optional[pd.DataFrame]:
    """
    Cotas de um mês para os CNPJs pedidos. Colunas: cnpj, date, cota, sub.
    Retorna None se o arquivo do mês não existe na CVM.
    """
    url = f"{CVM_BASE}/DOC/INF_DIARIO/DADOS/inf_diario_fi_{mes}.zip"
    for tentativa in range(3):
        try:
            r = HTTP.get(url, timeout=300)
            break
        except requests.RequestException as exc:
            if tentativa == 2:
                raise RuntimeError(f"Falha ao baixar {mes} da CVM: {exc}") from exc
            time.sleep(5 * (tentativa + 1))
    if r.status_code == 404:
        return None
    r.raise_for_status()

    partes = []
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(z.namelist()[0]) as f:
            for bloco in pd.read_csv(f, sep=";", encoding="latin-1", dtype=str,
                                     usecols=lambda c: c in _COLS_DIARIO, chunksize=200_000):
                col = "CNPJ_FUNDO_CLASSE" if "CNPJ_FUNDO_CLASSE" in bloco.columns else "CNPJ_FUNDO"
                sel = bloco[bloco[col].isin(alvos)]
                if sel.empty:
                    continue
                partes.append(pd.DataFrame({
                    "cnpj": sel[col].values,
                    "date": sel["DT_COMPTC"].values,
                    "cota": sel["VL_QUOTA"].values,
                    "sub": sel["ID_SUBCLASSE"].fillna("").values if "ID_SUBCLASSE" in sel.columns else "",
                }))
    del r
    if not partes:
        return pd.DataFrame(columns=["cnpj", "date", "cota", "sub"])
    return pd.concat(partes, ignore_index=True)


def _ler_mes(mes: str) -> pd.DataFrame:
    p = COTAS_DIR / f"{mes}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=["cnpj", "date", "cota", "sub"])
    df = pd.read_parquet(p)
    if "sub" not in df.columns:
        df["sub"] = ""
    df["date"] = df["date"].astype(str)
    df["cota"] = df["cota"].astype(str)
    return df[["cnpj", "date", "cota", "sub"]]


def _gravar_mes(mes: str, df: pd.DataFrame) -> None:
    p = COTAS_DIR / f"{mes}.parquet"
    tmp = p.with_suffix(".tmp")
    df = df.astype({"cnpj": str, "date": str, "cota": str, "sub": str})
    df.to_parquet(tmp, index=False)
    tmp.replace(p)


# ---------------------------------------------------------------------------
# Fatores
# ---------------------------------------------------------------------------

def _cdi_bcb(inicio: pd.Timestamp, fim: pd.Timestamp) -> pd.Series:
    url = ("https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados?formato=json"
           f"&dataInicial={inicio:%d/%m/%Y}&dataFinal={fim:%d/%m/%Y}")
    r = HTTP.get(url, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    s = pd.to_numeric(df["valor"], errors="coerce") / 100
    s.index = pd.to_datetime(df["data"], dayfirst=True)
    return s.sort_index().dropna().rename("CDI")


def _yahoo(ticker: str, nome: str, inicio: pd.Timestamp, fim: pd.Timestamp) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start=(inicio - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                     end=(fim + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                     auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"Yahoo Finance sem dados para {ticker}")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    s = close.pct_change().dropna()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    return s.rename(nome)


def _atualizar_fatores() -> List[str]:
    """Atualiza factors.csv. Fatores que falharem mantêm a série anterior. Retorna avisos."""
    inicio, fim = _janela()
    anterior = pd.read_csv(STORE / ARQ_FATORES, index_col=0, parse_dates=True) if (STORE / ARQ_FATORES).exists() else pd.DataFrame()
    fontes = {
        "CDI": lambda: _cdi_bcb(inicio, fim),
        "IMA-B": lambda: _yahoo("IMAB11.SA", "IMA-B", inicio, fim),
        "Ibovespa": lambda: _yahoo("^BVSP", "Ibovespa", inicio, fim),
        "BRL_USD": lambda: _yahoo("BRL=X", "BRL_USD", inicio, fim),
    }
    series, avisos = [], []
    for nome, fn in fontes.items():
        try:
            s = fn()
            if nome in anterior.columns:
                # completa com o histórico anterior onde a fonte nova não cobre
                s = s.combine_first(anterior[nome].dropna())
            series.append(s.rename(nome))
        except Exception as exc:  # noqa: BLE001
            avisos.append(f"{nome}: não atualizado ({exc})")
            log.warning("Fator %s falhou: %s", nome, exc)
            if nome in anterior.columns:
                series.append(anterior[nome].rename(nome))
    fat = pd.concat(series, axis=1)[FATORES] if series else anterior
    fat = fat[fat.index >= inicio - pd.Timedelta(days=5)].sort_index()
    fat.index.name = "date"
    with _lock_arquivos:
        _salvar_csv(fat, ARQ_FATORES)
    return avisos


# ---------------------------------------------------------------------------
# Retornos e regressão
# ---------------------------------------------------------------------------

def _serie_retornos(g: pd.DataFrame) -> pd.Series:
    """Retornos diários de um fundo; usa a classe sem subclasse, ou a subclasse mais frequente."""
    if (g["sub"] == "").any():
        g = g[g["sub"] == ""]
    else:
        g = g[g["sub"] == g["sub"].value_counts().idxmax()]
    s = pd.Series(pd.to_numeric(g["cota"].str.replace(",", ".", regex=False), errors="coerce").values,
                  index=pd.to_datetime(g["date"], errors="coerce")).dropna()
    s = s[s > 0].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s.pct_change().dropna()


def _recalcular() -> str:
    """Monta a matriz de retornos e reestima a regressão de todos os fundos da lista."""
    _etapa("Montando retornos dos fundos…", 85)
    fundos = carregar_fundos()
    alvos = set(fundos["cnpj"])
    inicio, fim = _janela()
    cotas = [m for m in (_ler_mes(mes) for mes in _meses(inicio, fim)) if not m.empty]
    cotas = pd.concat(cotas, ignore_index=True) if cotas else pd.DataFrame(columns=["cnpj", "date", "cota", "sub"])
    cotas = cotas[cotas["cnpj"].isin(alvos)]

    retornos = {}
    for cnpj, g in cotas.groupby("cnpj"):
        s = _serie_retornos(g)
        s = s[s.index >= inicio]
        if len(s) >= 20:
            retornos[cnpj] = s
    matriz = pd.DataFrame(retornos).sort_index()
    matriz.index.name = "date"

    _etapa("Estimando a análise de estilo…", 92)
    fat = pd.read_csv(STORE / ARQ_FATORES, index_col=0, parse_dates=True)
    nomes = dict(zip(fundos["cnpj"], fundos["nome"]))
    res = []
    for cnpj in matriz.columns:
        r = analyze_fund(cnpj, matriz[cnpj], fat)
        if r is not None:
            res.append(r)
    pesos = ["weight_CDI", "weight_IMAB", "weight_Ibovespa", "weight_BRL_USD"]
    colunas = ["cnpj", "nome", *pesos, "alpha", "r_squared", "tracking_error", "n_obs", "converged", "dominant_style"]
    df = pd.DataFrame(res)
    if df.empty:
        df = pd.DataFrame(columns=colunas)
    else:
        df["nome"] = df["cnpj"].map(nomes).fillna("")
        df["dominant_style"] = df[pesos].idxmax(axis=1).map(dict(zip(pesos, FACTOR_LABELS)))
        df = df[colunas]

    with _lock_arquivos:
        _salvar_csv(matriz, ARQ_RETORNOS)
        _salvar_csv(df, ARQ_RESULTADOS, index=False)
        dados_ate = str(matriz.index.max().date()) if len(matriz) else None
        _gravar_status({"dados_ate": dados_ate, "ultimo_calculo": _agora()})
    sem = len(alvos) - len(df)
    return f"{len(df)} fundos analisados" + (f"; {sem} sem dados suficientes" if sem else "")


def _atualizar() -> str:
    """Baixa os meses que faltam e os mais recentes da CVM, atualiza fatores e recalcula."""
    inicio, fim = _janela()
    meses = _meses(inicio, fim)
    recentes = set(meses[-MESES_RECENTES:])
    alvos = set(carregar_fundos()["cnpj"])
    baixar = [m for m in meses if m in recentes or not (COTAS_DIR / f"{m}.parquet").exists()]

    for i, mes in enumerate(baixar, 1):
        _etapa(f"Baixando cotas da CVM: {mes[4:]}/{mes[:4]} ({i}/{len(baixar)})", 5 + 65 * i / max(len(baixar), 1))
        df = _baixar_mes(mes, alvos)
        if df is not None:
            _gravar_mes(mes, df)

    # remove meses fora da janela
    for p in COTAS_DIR.glob("*.parquet"):
        if p.stem < meses[0]:
            p.unlink(missing_ok=True)

    _etapa("Atualizando fatores (CDI, IMA-B, Ibovespa, câmbio)…", 75)
    avisos = _atualizar_fatores()
    msg = _recalcular()
    _gravar_status({"ultima_atualizacao": _agora(), "avisos": avisos})
    return msg + (" · " + "; ".join(avisos) if avisos else "")


# ---------------------------------------------------------------------------
# Adicionar e remover fundos
# ---------------------------------------------------------------------------

def preparar_inclusao(cnpj_digitado: str) -> Dict:
    """Valida o CNPJ e acha a classe correspondente no cadastro. Não grava nada."""
    if not cnpj_valido(cnpj_digitado):
        raise ValueError("CNPJ inválido. Confira os dígitos.")
    cnpj = fmt_cnpj(cnpj_digitado)
    if cnpj in set(carregar_fundos()["cnpj"]):
        raise ValueError("Esse fundo já está na lista.")
    reg = _registro()
    linha = reg[reg["cnpj"] == cnpj]
    if linha.empty:
        classes = reg[reg["cnpj_fundo"] == cnpj]
        if len(classes) == 1:
            linha = classes
        elif len(classes) > 1:
            opcoes = "; ".join(f"{r.nome} ({r.cnpj})" for r in classes.itertuples())
            raise ValueError(f"Esse CNPJ é de um fundo com várias classes. Escolha uma: {opcoes}")
        else:
            raise ValueError("CNPJ não encontrado no cadastro da CVM.")
    r = linha.iloc[0]
    if r["cnpj"] in set(carregar_fundos()["cnpj"]):
        raise ValueError("Esse fundo já está na lista.")
    return r.to_dict()


def _adicionar(cnpj: str) -> str:
    info = preparar_inclusao(cnpj)
    cnpj = info["cnpj"]
    inicio, fim = _janela()
    meses = _meses(inicio, fim)
    demais = set(carregar_fundos()["cnpj"])
    achou = 0
    for i, mes in enumerate(meses, 1):
        _etapa(f"Buscando histórico de {info['nome'][:40]}: {mes[4:]}/{mes[:4]} ({i}/{len(meses)})",
               3 + 77 * i / len(meses))
        existe = (COTAS_DIR / f"{mes}.parquet").exists()
        # mês ainda não baixado: aproveita o download para todos os fundos da lista
        novo = _baixar_mes(mes, {cnpj} if existe else demais | {cnpj})
        if novo is None:
            continue
        achou += int((novo["cnpj"] == cnpj).sum())
        with _lock_arquivos:
            if existe:
                atual = _ler_mes(mes)
                novo = pd.concat([atual[atual["cnpj"] != cnpj], novo], ignore_index=True)
            _gravar_mes(mes, novo)
    if achou == 0:
        raise RuntimeError("A CVM não tem cotas diárias desse fundo nos últimos 3 anos.")

    with _lock_arquivos:
        fundos = carregar_fundos()
        fundos = pd.concat([fundos, pd.DataFrame([{
            "cnpj": cnpj, "nome": info["nome"], "classe": info.get("classe_anbima") or info.get("classificacao", ""),
            "adicionado_em": _agora(),
        }])], ignore_index=True)
        _salvar_csv(fundos, ARQ_FUNDOS, index=False)
    msg = _recalcular()
    return f"{info['nome']} adicionado · {msg}"


def remover(cnpj: str) -> None:
    cnpj = fmt_cnpj(cnpj)
    with _lock_arquivos:
        fundos = carregar_fundos()
        if cnpj not in set(fundos["cnpj"]):
            raise ValueError("Fundo não está na lista.")
        _salvar_csv(fundos[fundos["cnpj"] != cnpj], ARQ_FUNDOS, index=False)
    enfileirar("recalcular")
