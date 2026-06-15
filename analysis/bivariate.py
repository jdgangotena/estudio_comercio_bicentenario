"""
Análisis bivariado: relaciones entre pares de variables.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from data_cleaner import explode_multiselect

PALETTE = px.colors.qualitative.Safe


def _fig_defaults(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Arial"), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        margin=dict(l=40, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#ccc")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig


def crosstab_heatmap(df: pd.DataFrame, col_x: str, col_y: str,
                     label_x: str, label_y: str) -> go.Figure:
    """Tabla cruzada como heatmap de porcentajes por fila."""
    ct = pd.crosstab(df[col_x], df[col_y])
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig = go.Figure(go.Heatmap(
        z=ct_pct.values,
        x=ct_pct.columns.tolist(),
        y=ct_pct.index.tolist(),
        colorscale="Blues",
        text=np.round(ct_pct.values, 1),
        texttemplate="%{text}%",
        hovertemplate=f"{label_y}: %{{x}}<br>{label_x}: %{{y}}<br>%: %{{z:.1f}}%<extra></extra>",
    ))
    _fig_defaults(fig, f"{label_x} vs {label_y}")
    fig.update_xaxes(title_text=label_y)
    fig.update_yaxes(title_text=label_x)
    return fig


def crosstab_stacked(df: pd.DataFrame, col_x: str, col_color: str,
                     label_x: str, label_color: str) -> go.Figure:
    """Barras apiladas para dos variables categóricas."""
    ct = pd.crosstab(df[col_x], df[col_color], normalize="index") * 100
    fig = go.Figure()
    for i, col in enumerate(ct.columns):
        fig.add_trace(go.Bar(
            name=str(col),
            x=ct.index.tolist(),
            y=ct[col].round(1).tolist(),
            marker_color=PALETTE[i % len(PALETTE)],
        ))
    fig.update_layout(barmode="stack")
    _fig_defaults(fig, f"{label_x} por {label_color}")
    fig.update_xaxes(title_text=label_x)
    fig.update_yaxes(title_text="Porcentaje (%)")
    return fig


def crosstab_grouped(df: pd.DataFrame, col_x: str, col_color: str,
                     label_x: str, label_color: str) -> go.Figure:
    """Barras agrupadas para dos variables categóricas."""
    ct = pd.crosstab(df[col_x], df[col_color])
    fig = go.Figure()
    for i, col in enumerate(ct.columns):
        fig.add_trace(go.Bar(
            name=str(col),
            x=ct.index.tolist(),
            y=ct[col].tolist(),
            marker_color=PALETTE[i % len(PALETTE)],
        ))
    fig.update_layout(barmode="group")
    _fig_defaults(fig, f"{label_x} por {label_color}")
    fig.update_xaxes(title_text=label_x)
    fig.update_yaxes(title_text="Frecuencia")
    return fig


def boxplot_continua_vs_cat(df: pd.DataFrame, col_num: str, col_cat: str,
                            label_num: str, label_cat: str) -> go.Figure:
    """Boxplot de variable numérica por categoría."""
    cats = df[col_cat].dropna().unique()
    fig = go.Figure()
    for i, cat in enumerate(sorted(cats)):
        vals = df.loc[df[col_cat] == cat, col_num].dropna()
        fig.add_trace(go.Box(
            y=vals, name=str(cat),
            marker_color=PALETTE[i % len(PALETTE)],
            boxmean="sd",
        ))
    _fig_defaults(fig, f"{label_num} por {label_cat}")
    fig.update_yaxes(title_text=label_num)
    return fig


def scatter_continuas(df: pd.DataFrame, col_x: str, col_y: str,
                      col_color: str = None, label_x: str = "",
                      label_y: str = "") -> go.Figure:
    """Scatter plot de dos variables continuas."""
    d = df[[col_x, col_y]].dropna()
    if col_color:
        d = df[[col_x, col_y, col_color]].dropna()
        fig = px.scatter(d, x=col_x, y=col_y, color=col_color,
                         color_discrete_sequence=PALETTE,
                         trendline="ols",
                         labels={col_x: label_x, col_y: label_y})
    else:
        r, p = stats.pearsonr(d[col_x], d[col_y])
        fig = px.scatter(d, x=col_x, y=col_y,
                         trendline="ols",
                         labels={col_x: label_x, col_y: label_y})
        fig.add_annotation(
            x=0.98, y=0.95, xref="paper", yref="paper",
            text=f"r = {r:.3f}  (p = {p:.4f})",
            showarrow=False, font=dict(size=12, color="gray"), align="right"
        )
    _fig_defaults(fig, f"{label_x} vs {label_y}")
    return fig


def stats_tabla_bivariada(df: pd.DataFrame, col_num: str, col_cat: str) -> pd.DataFrame:
    """Estadísticos descriptivos de variable numérica por categoría."""
    grouped = df.groupby(col_cat)[col_num].agg(
        N="count", Media="mean", Mediana="median",
        DesvEst="std", Min="min", Max="max"
    ).round(2).reset_index()
    grouped.columns = [col_cat, "N", "Media", "Mediana", "Desv. Est.", "Mínimo", "Máximo"]
    return grouped


def consumo_vs_gasto(df: pd.DataFrame) -> go.Figure:
    """Barras agrupadas: disposición de gasto por género."""
    if "gasto_dispuesto" not in df.columns or "genero" not in df.columns:
        return go.Figure()
    order = ["Menos de $2", "Entre $2 y $5", "Entre $5 y $10", "Entre $10 y $20", "Más de $20"]
    ct = pd.crosstab(df["gasto_dispuesto"], df["genero"])
    ct = ct.reindex([o for o in order if o in ct.index])
    fig = go.Figure()
    for i, col in enumerate(ct.columns):
        fig.add_trace(go.Bar(name=col, x=ct.index.tolist(), y=ct[col].tolist(),
                             marker_color=PALETTE[i]))
    fig.update_layout(barmode="group")
    _fig_defaults(fig, "Gasto dispuesto a pagar por género")
    fig.update_xaxes(title_text="Rango de gasto")
    fig.update_yaxes(title_text="Frecuencia")
    return fig


def chi2_test(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    """Prueba Chi-cuadrado de independencia."""
    ct = pd.crosstab(df[col_x].dropna(), df[col_y].dropna())
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    return {
        "chi2": round(chi2, 4),
        "p_valor": round(p, 6),
        "grados_libertad": dof,
        "interpretacion": "Dependientes (p<0.05)" if p < 0.05 else "Independientes (p≥0.05)",
    }
