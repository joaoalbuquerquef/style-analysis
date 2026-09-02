"""
Style Analysis Dashboard — Flask app.

Data is loaded once at startup from data/raw/*.csv.
No regressions are re-run per request.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from flask import Flask, abort, render_template, request

# ---------------------------------------------------------------------------
# Startup data loading
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data" / "raw"

style_df   = pd.read_csv(DATA_DIR / "style_results.csv")
fund_ret   = pd.read_csv(DATA_DIR / "fund_returns.csv",  index_col=0, parse_dates=True)
factors_df = pd.read_csv(DATA_DIR / "factors.csv",       index_col=0, parse_dates=True)

FACTOR_COLS   = ["weight_CDI", "weight_IMAB", "weight_Ibovespa", "weight_BRL_USD"]
FACTOR_LABELS = ["CDI", "IMA-B", "Ibovespa", "BRL/USD"]
FACTOR_COLORS = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6"]
FAC_SERIES    = ["CDI", "IMA-B", "Ibovespa", "BRL_USD"]   # columns in factors_df

# URL-safe fund identifier = 14 raw digits
def _to_id(cnpj: str) -> str:
    return "".join(c for c in cnpj if c.isdigit())

style_df["id"] = style_df["cnpj"].apply(_to_id)
ID_TO_ROW: dict = {row["id"]: row for _, row in style_df.iterrows()}

# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

_CHART_BASE = dict(
    paper_bgcolor="white",
    plot_bgcolor="white",
    font=dict(family="Inter, system-ui, sans-serif", size=13, color="#334155"),
    hoverlabel=dict(bgcolor="white", font_size=12),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

_AXIS = dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0", linecolor="#e2e8f0")


def _to_json(fig: go.Figure) -> str:
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _style_bar_chart(row: pd.Series) -> str:
    """Horizontal bar chart of the 4 factor exposures for one fund."""
    weights = [row[c] for c in FACTOR_COLS]

    fig = go.Figure()
    for label, color, w in zip(FACTOR_LABELS, FACTOR_COLORS, weights):
        fig.add_trace(go.Bar(
            x=[w * 100],
            y=[label],
            orientation="h",
            marker_color=color,
            text=[f"{w*100:.1f}%"],
            textposition="outside",
            cliponaxis=False,
            hovertemplate=f"<b>{label}</b>: %{{x:.1f}}%<extra></extra>",
            showlegend=False,
        ))

    fig.update_layout(**_CHART_BASE)
    fig.update_layout(
        barmode="relative",
        height=220,
        margin=dict(l=8, r=60, t=36, b=8),
        title=dict(text="Style Exposures", font=dict(size=14, color="#0f172a")),
        xaxis=dict(**_AXIS, range=[0, 115], ticksuffix="%"),
        yaxis=dict(**_AXIS, categoryorder="array",
                   categoryarray=FACTOR_LABELS[::-1], tickfont=dict(size=13)),
    )
    return _to_json(fig)


def _return_chart(cnpj: str, row: pd.Series) -> str:
    """Cumulative return of fund vs style benchmark, indexed to 100."""
    weights = np.array([row[c] for c in FACTOR_COLS])

    if cnpj not in fund_ret.columns:
        return None

    aligned = pd.concat(
        [fund_ret[cnpj].rename("fund"), factors_df[FAC_SERIES]],
        axis=1, join="inner",
    ).dropna()
    aligned = aligned[aligned["fund"].abs() <= 0.25]

    bench_daily = aligned[FAC_SERIES].values @ weights
    cum_fund  = (1 + aligned["fund"]).cumprod() * 100
    cum_bench = (1 + pd.Series(bench_daily, index=aligned.index)).cumprod() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=aligned.index, y=cum_fund.values,
        name="Fund",
        line=dict(color="#2563eb", width=2),
        hovertemplate="%{x|%d %b %Y}: <b>%{y:.1f}</b><extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=aligned.index, y=cum_bench.values,
        name="Style Benchmark",
        line=dict(color="#f59e0b", width=2, dash="dash"),
        hovertemplate="%{x|%d %b %Y}: <b>%{y:.1f}</b><extra></extra>",
    ))
    fig.update_layout(**_CHART_BASE)
    fig.update_layout(
        height=320,
        margin=dict(l=8, r=8, t=40, b=8),
        title=dict(text="Cumulative Return  (base = 100)", font=dict(size=14, color="#0f172a")),
        xaxis=dict(**_AXIS),
        yaxis=dict(**_AXIS),
        hovermode="x unified",
    )
    return _to_json(fig)


def _compare_chart(ids: list) -> str:
    """Grouped bar chart of style exposures for up to 5 funds."""
    rows = [ID_TO_ROW[i] for i in ids if i in ID_TO_ROW]

    # Short names for legend (first ~18 chars of fund name)
    def _short(r):
        name = str(r["nome"])[:28] or r["cnpj"]
        return name

    fig = go.Figure()
    for label, color, col in zip(FACTOR_LABELS, FACTOR_COLORS, FACTOR_COLS):
        fig.add_trace(go.Bar(
            name=label,
            x=[_short(r) for r in rows],
            y=[r[col] * 100 for r in rows],
            marker_color=color,
            hovertemplate="<b>%{x}</b><br>" + label + ": %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(**_CHART_BASE)
    fig.update_layout(
        barmode="group",
        height=420,
        margin=dict(l=8, r=8, t=50, b=80),
        title=dict(text="Style Exposures — Side by Side", font=dict(size=14, color="#0f172a")),
        xaxis=dict(**_AXIS, tickangle=-20),
        yaxis=dict(**_AXIS, ticksuffix="%", range=[0, 108]),
    )
    return _to_json(fig)


def _cum_compare_chart(ids: list) -> str:
    """Cumulative return overlay for selected funds + benchmarks."""
    rows = [ID_TO_ROW[i] for i in ids if i in ID_TO_ROW]
    PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

    fig = go.Figure()
    for row, color in zip(rows, PALETTE):
        cnpj = row["cnpj"]
        weights = np.array([row[c] for c in FACTOR_COLS])
        if cnpj not in fund_ret.columns:
            continue
        aligned = pd.concat(
            [fund_ret[cnpj].rename("fund"), factors_df[FAC_SERIES]],
            axis=1, join="inner",
        ).dropna()
        aligned = aligned[aligned["fund"].abs() <= 0.25]

        cum = (1 + aligned["fund"]).cumprod() * 100
        short = str(row["nome"])[:24] or cnpj
        fig.add_trace(go.Scatter(
            x=aligned.index, y=cum.values,
            name=short,
            line=dict(color=color, width=2),
            hovertemplate="%{x|%d %b %Y}: <b>%{y:.1f}</b><extra></extra>",
        ))

    fig.update_layout(**_CHART_BASE)
    fig.update_layout(
        height=380,
        margin=dict(l=8, r=8, t=50, b=8),
        title=dict(text="Cumulative Return Comparison  (base = 100)", font=dict(size=14, color="#0f172a")),
        hovermode="x unified",
        yaxis=dict(**_AXIS),
    )
    return _to_json(fig)


# ---------------------------------------------------------------------------
# Template context helpers
# ---------------------------------------------------------------------------

def _pct(v: float, decimals: int = 2) -> str:
    return f"{v*100:+.{decimals}f}%"

def _fmt_r2(v: float) -> str:
    return f"{v:.3f}"

def _alpha_class(alpha: float) -> str:
    return "text-emerald-600 font-semibold" if alpha >= 0 else "text-red-500 font-semibold"

def _style_badge(style: str) -> str:
    colors = {
        "CDI":      "bg-blue-100 text-blue-700",
        "IMAB":     "bg-amber-100 text-amber-700",
        "Ibovespa": "bg-emerald-100 text-emerald-700",
        "BRL_USD":  "bg-violet-100 text-violet-700",
    }
    return colors.get(style, "bg-slate-100 text-slate-600")


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.jinja_env.globals.update(
    pct=_pct, fmt_r2=_fmt_r2, alpha_class=_alpha_class, style_badge=_style_badge,
)


@app.route("/")
def index():
    funds = style_df.to_dict(orient="records")
    return render_template("index.html", funds=funds)


@app.route("/fund/<cnpj_id>")
def fund_detail(cnpj_id: str):
    row = ID_TO_ROW.get(cnpj_id)
    if row is None:
        abort(404)

    row_dict    = row.to_dict()   # plain dict — safe in Jinja2
    cnpj        = row_dict["cnpj"]
    bar_json    = _style_bar_chart(row)
    return_json = _return_chart(cnpj, row)

    return render_template(
        "fund.html",
        row=row_dict,
        bar_json=bar_json,
        return_json=return_json,
    )


@app.route("/compare")
def compare():
    selected_ids = [s.strip() for s in request.args.get("funds", "").split(",") if s.strip()]
    selected_ids = selected_ids[:5]

    compare_json  = _compare_chart(selected_ids) if len(selected_ids) >= 2 else None
    cum_json      = _cum_compare_chart(selected_ids) if len(selected_ids) >= 2 else None

    selected_rows = [ID_TO_ROW[i].to_dict() for i in selected_ids if i in ID_TO_ROW]
    funds = style_df.to_dict(orient="records")

    return render_template(
        "compare.html",
        funds=funds,
        selected_ids=selected_ids,
        selected_rows=selected_rows,
        compare_json=compare_json,
        cum_json=cum_json,
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5001)
