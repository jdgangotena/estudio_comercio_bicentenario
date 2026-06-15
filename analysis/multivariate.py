"""
Análisis multivariado: PCA, clustering y análisis de correspondencias.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_cleaner import explode_multiselect

PALETTE = px.colors.qualitative.Safe


def _fig_defaults(fig, title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Arial"), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        margin=dict(l=40, r=20, t=60, b=60),
    )
    return fig


def _encode_df(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Encode categórico a numérico para análisis."""
    enc = df[cols].copy()
    le = LabelEncoder()
    for col in enc.select_dtypes(include="object").columns:
        enc[col] = le.fit_transform(enc[col].astype(str))
    enc = enc.apply(pd.to_numeric, errors="coerce").fillna(enc.median())
    return enc


def pca_biplot(df: pd.DataFrame, cols: list, labels: dict,
               color_col: str = None) -> tuple[go.Figure, pd.DataFrame]:
    """
    PCA con biplot (scores + loadings) y varianza explicada.
    Retorna (figura, varianza_explicada_df).
    """
    enc = _encode_df(df, cols)
    scaler = StandardScaler()
    X = scaler.fit_transform(enc)

    pca = PCA(n_components=min(len(cols), 5))
    scores = pca.fit_transform(X)

    var_exp = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(pca.n_components_)],
        "Varianza explicada (%)": np.round(pca.explained_variance_ratio_ * 100, 2),
        "Varianza acumulada (%)": np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2),
    })

    fig = go.Figure()

    # Scores
    colors = PALETTE
    if color_col and color_col in df.columns:
        color_vals = df[color_col].fillna("N/A").astype(str)
        unique_cats = color_vals.unique()
        for i, cat in enumerate(unique_cats):
            mask = color_vals == cat
            fig.add_trace(go.Scatter(
                x=scores[mask, 0], y=scores[mask, 1],
                mode="markers",
                name=str(cat),
                marker=dict(color=colors[i % len(colors)], size=7, opacity=0.7),
            ))
    else:
        fig.add_trace(go.Scatter(
            x=scores[:, 0], y=scores[:, 1],
            mode="markers",
            name="Observaciones",
            marker=dict(color=colors[0], size=7, opacity=0.6),
        ))

    # Loadings (vectores)
    scale = np.max(np.abs(scores[:, :2])) * 0.4
    for j, col in enumerate(cols):
        lx, ly = pca.components_[0, j] * scale, pca.components_[1, j] * scale
        fig.add_annotation(
            ax=0, ay=0, x=lx, y=ly,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=2, arrowcolor="#e74c3c",
        )
        fig.add_annotation(
            x=lx * 1.1, y=ly * 1.1,
            text=labels.get(col, col),
            font=dict(size=10, color="#e74c3c"),
            showarrow=False,
        )

    pc1_var = round(pca.explained_variance_ratio_[0] * 100, 1)
    pc2_var = round(pca.explained_variance_ratio_[1] * 100, 1)
    _fig_defaults(fig, "Biplot PCA - Componentes Principales")
    fig.update_xaxes(title_text=f"PC1 ({pc1_var}% varianza)", zerolinecolor="#ddd")
    fig.update_yaxes(title_text=f"PC2 ({pc2_var}% varianza)", zerolinecolor="#ddd")
    return fig, var_exp


def scree_plot(df: pd.DataFrame, cols: list) -> go.Figure:
    """Gráfico de sedimentación (scree plot) para PCA."""
    enc = _encode_df(df, cols)
    X = StandardScaler().fit_transform(enc)
    pca = PCA(n_components=min(len(cols), 8))
    pca.fit(X)

    var_pct = pca.explained_variance_ratio_ * 100
    cum_var = np.cumsum(var_pct)
    comps = [f"PC{i+1}" for i in range(len(var_pct))]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=comps, y=var_pct, name="Var. individual (%)",
                         marker_color="#1f77b4", opacity=0.8))
    fig.add_trace(go.Scatter(x=comps, y=cum_var, name="Var. acumulada (%)",
                             line=dict(color="#ff7f0e", width=2.5), mode="lines+markers"))
    fig.add_hline(y=80, line_dash="dot", line_color="gray",
                  annotation_text="80% varianza acumulada")
    _fig_defaults(fig, "Gráfico de Sedimentación (Scree Plot)")
    fig.update_yaxes(title_text="Varianza explicada (%)")
    return fig


def kmeans_clusters(df: pd.DataFrame, cols: list, n_clusters: int = 3,
                    color_labels: list = None) -> tuple[go.Figure, pd.Series, pd.DataFrame]:
    """
    K-Means clustering. Retorna (figura PCA-coloreada, etiquetas, perfiles).
    """
    enc = _encode_df(df, cols)
    X = StandardScaler().fit_transform(enc)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    pc1_var = round(pca.explained_variance_ratio_[0] * 100, 1)
    pc2_var = round(pca.explained_variance_ratio_[1] * 100, 1)

    fig = go.Figure()
    for k in range(n_clusters):
        mask = labels == k
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers",
            name=f"Segmento {k+1}",
            marker=dict(color=PALETTE[k % len(PALETTE)], size=8, opacity=0.75),
        ))
    _fig_defaults(fig, f"Segmentación de visitantes (K-Means, k={n_clusters})")
    fig.update_xaxes(title_text=f"PC1 ({pc1_var}%)")
    fig.update_yaxes(title_text=f"PC2 ({pc2_var}%)")

    label_series = pd.Series(labels, index=df.index, name="cluster")

    # Perfil de cada cluster
    profile_df = df[cols].copy()
    profile_df["cluster"] = labels
    profile = profile_df.groupby("cluster").mean(numeric_only=True).round(2)
    profile.index = [f"Segmento {k+1}" for k in profile.index]

    return fig, label_series, profile


def elbow_silhouette(df: pd.DataFrame, cols: list,
                     max_k: int = 8) -> go.Figure:
    """Gráfico de codo + silhouette para determinar k óptimo."""
    enc = _encode_df(df, cols)
    X = StandardScaler().fit_transform(enc)

    inertias, silhouettes = [], []
    ks = range(2, max_k + 1)
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(ks), y=inertias, mode="lines+markers",
                             name="Inercia (codo)", yaxis="y1",
                             line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=list(ks), y=silhouettes, mode="lines+markers",
                             name="Silhouette", yaxis="y2",
                             line=dict(color="#ff7f0e", dash="dot")))
    fig.update_layout(
        yaxis=dict(title="Inercia", showgrid=True, gridcolor="#f0f0f0"),
        yaxis2=dict(title="Silhouette score", overlaying="y", side="right",
                    range=[0, 1], showgrid=False),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        title=dict(text="Método del Codo y Silhouette para K óptimo",
                   font=dict(size=16), x=0.02),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(title="Número de clusters (k)", showgrid=False),
    )
    return fig


def correlation_matrix(df: pd.DataFrame, cols: list,
                        labels: dict = None) -> go.Figure:
    """Matriz de correlación de variables numéricas/codificadas."""
    enc = _encode_df(df, cols)
    corr = enc.corr()
    col_names = [labels.get(c, c) if labels else c for c in corr.columns]

    fig = go.Figure(go.Heatmap(
        z=np.round(corr.values, 2),
        x=col_names, y=col_names,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        hovertemplate="X: %{x}<br>Y: %{y}<br>r: %{z}<extra></extra>",
    ))
    _fig_defaults(fig, "Matriz de Correlación entre Variables")
    return fig
