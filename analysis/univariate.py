"""
Análisis univariado: distribuciones, frecuencias y gráficos para cada variable.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from data_cleaner import count_multiselect, CALIFICACION_LABELS, GASTO_ORDER

PALETTE = px.colors.qualitative.Safe
COLOR_PRIMARY = "#1f77b4"
COLOR_SECONDARY = "#ff7f0e"


def _fig_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Arial"), x=0.02),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        margin=dict(l=40, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#ccc")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", showline=False)
    return fig


# ── Continuas ──────────────────────────────────────────────────────────────────

def hist_gaussian(series: pd.Series, col_label: str, unit: str = "",
                  bins: int = 20) -> go.Figure:
    """
    Histograma con campana de Gauss superpuesta.
    """
    data = series.dropna().astype(float)
    mu, sigma = data.mean(), data.std()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=bins,
        name="Frecuencia observada",
        marker_color=COLOR_PRIMARY,
        opacity=0.7,
        histnorm="probability density",
    ))

    x_range = np.linspace(data.min() - sigma * 0.5, data.max() + sigma * 0.5, 300)
    y_gauss = stats.norm.pdf(x_range, mu, sigma)
    fig.add_trace(go.Scatter(
        x=x_range, y=y_gauss,
        mode="lines",
        name=f"Normal (μ={mu:.1f}, σ={sigma:.1f})",
        line=dict(color=COLOR_SECONDARY, width=2.5),
    ))

    fig.add_vline(x=mu, line_dash="dash", line_color="gray",
                  annotation_text=f"Media: {mu:.1f}{unit}", annotation_position="top right")

    _fig_defaults(fig, f"Distribución de {col_label}")
    fig.update_xaxes(title_text=f"{col_label} {unit}".strip())
    fig.update_yaxes(title_text="Densidad de probabilidad")
    return fig


def table_stats_continua(series: pd.Series) -> pd.DataFrame:
    data = series.dropna().astype(float)
    sk = stats.skew(data)
    ku = stats.kurtosis(data)
    _, pval = stats.shapiro(data.sample(min(len(data), 50), random_state=42))
    return pd.DataFrame({
        "Estadístico": ["N", "Media", "Mediana", "Moda", "Desv. Estándar",
                        "Mínimo", "Máximo", "Asimetría", "Curtosis", "Shapiro-Wilk p"],
        "Valor": [
            int(len(data)), round(data.mean(), 2), round(data.median(), 2),
            round(data.mode().iloc[0], 2) if not data.mode().empty else np.nan,
            round(data.std(), 2), round(data.min(), 2), round(data.max(), 2),
            round(sk, 3), round(ku, 3), round(pval, 4),
        ]
    })


def hist_rangos(series: pd.Series, rangos: pd.Categorical,
                col_label: str) -> go.Figure:
    """Histograma agrupado por rangos."""
    counts = rangos.value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(1)

    fig = go.Figure(go.Bar(
        x=[str(r) for r in counts.index],
        y=counts.values,
        text=[f"{p}%" for p in pct.values],
        textposition="outside",
        marker_color=PALETTE[:len(counts)],
    ))
    _fig_defaults(fig, f"{col_label} por rangos")
    fig.update_xaxes(title_text="Rango")
    fig.update_yaxes(title_text="Número de respuestas")
    return fig


# ── Categóricas ────────────────────────────────────────────────────────────────

def bar_categorica(series: pd.Series, col_label: str,
                   order: list = None, horizontal: bool = False) -> go.Figure:
    counts = series.dropna().value_counts()
    if order:
        counts = counts.reindex([o for o in order if o in counts.index]).dropna()

    pct = (counts / counts.sum() * 100).round(1)
    labels = [str(l) for l in counts.index]
    values = counts.values
    texts = [f"{p}%" for p in pct.values]

    if horizontal:
        fig = go.Figure(go.Bar(
            y=labels[::-1], x=values[::-1],
            text=texts[::-1], textposition="outside",
            orientation="h",
            marker_color=PALETTE[:len(labels)],
        ))
        _fig_defaults(fig, f"Distribución: {col_label}")
        fig.update_xaxes(title_text="Frecuencia")
        # Altura dinámica para que todas las barras sean visibles
        n = len(labels)
        fig.update_layout(
            height=max(320, n * 46 + 90),
            margin=dict(l=10, r=70, t=55, b=30),
        )
        # Dar espacio suficiente al eje Y para etiquetas largas
        max_label_len = max(len(l) for l in labels) if labels else 10
        left_margin = min(max_label_len * 7, 220)
        fig.update_layout(margin=dict(l=left_margin, r=70, t=55, b=30))
    else:
        fig = go.Figure(go.Bar(
            x=labels, y=values,
            text=texts, textposition="outside",
            marker_color=PALETTE[:len(labels)],
        ))
        _fig_defaults(fig, f"Distribución: {col_label}")
        fig.update_yaxes(title_text="Frecuencia")
    return fig


def pie_categorica(series: pd.Series, col_label: str) -> go.Figure:
    counts = series.dropna().value_counts()
    total = counts.sum()
    # Slices pequeños (< 8%) solo muestran el %; las etiquetas van a la leyenda
    custom_text = [
        f"{v / total * 100:.1f}%" for v in counts.values
    ]
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.38,
        marker_colors=PALETTE,
        text=custom_text,
        textinfo="text",
        textposition="inside",
        insidetextorientation="radial",
        hovertemplate="<b>%{label}</b><br>%{value} respuestas (%{percent})<extra></extra>",
        showlegend=True,
    ))
    _fig_defaults(fig, f"Distribución: {col_label}")
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="middle", y=0.5,
            xanchor="left", x=1.01,
            font=dict(size=10),
        ),
        margin=dict(l=10, r=140, t=55, b=10),
    )
    return fig


def table_frecuencia(series: pd.Series, order: list = None) -> pd.DataFrame:
    counts = series.dropna().value_counts()
    if order:
        counts = counts.reindex([o for o in order if o in counts.index]).dropna()
    pct = (counts / counts.sum() * 100).round(1)
    pct_acum = pct.cumsum().round(1)
    df = pd.DataFrame({
        "Opción": counts.index,
        "Frecuencia": counts.values,
        "% del total": pct.values,
        "% acumulado": pct_acum.values,
    })
    return df


# ── Multi-selección ────────────────────────────────────────────────────────────

def bar_multiselect(df: pd.DataFrame, column: str, col_label: str,
                    top_n: int = 15) -> go.Figure:
    counts_df = count_multiselect(df, column).head(top_n)
    fig = go.Figure(go.Bar(
        y=counts_df["opcion"].tolist()[::-1],
        x=counts_df["frecuencia"].tolist()[::-1],
        text=[f"{p}%" for p in counts_df["pct_respuestas"].tolist()[::-1]],
        textposition="outside",
        orientation="h",
        marker_color=PALETTE[:len(counts_df)],
    ))
    _fig_defaults(fig, f"{col_label} (selección múltiple)")
    fig.update_xaxes(title_text="Frecuencia de mención")
    fig.add_annotation(
        x=0, y=-0.12, xref="paper", yref="paper",
        text="* Los porcentajes pueden superar 100% por ser selección múltiple",
        showarrow=False, font=dict(size=10, color="gray"), align="left"
    )
    return fig


def table_multiselect(df: pd.DataFrame, column: str) -> pd.DataFrame:
    return count_multiselect(df, column).rename(columns={
        "opcion": "Opción",
        "frecuencia": "Frecuencia",
        "pct_respuestas": "% de encuestados que lo mencionaron",
    })


# ── Calificación (escala 1-5) ──────────────────────────────────────────────────

def bar_calificacion(series: pd.Series, col_label: str) -> go.Figure:
    counts = series.dropna().astype(int).value_counts().sort_index()
    pct = (counts / counts.sum() * 100).round(1)
    labels = [f"{k} - {CALIFICACION_LABELS.get(k, '')}" for k in counts.index]

    colors = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1565c0"]
    bar_colors = [colors[k - 1] for k in counts.index]

    fig = go.Figure(go.Bar(
        x=labels, y=counts.values,
        text=[f"{p}%" for p in pct.values],
        textposition="outside",
        marker_color=bar_colors,
    ))
    _fig_defaults(fig, f"Calificación: {col_label}")
    fig.update_yaxes(title_text="Frecuencia")
    return fig
