"""
Prueba de hipótesis para las dos preguntas clave del estudio de mercado.

H1: ¿Se deben mantener 50 kioskos como lo determina el plan masa de la SHOT?
H2: ¿La gente está de acuerdo con implementar kioskos en el Parque Bicentenario?
"""

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import norm


# ── Utilidades de visualización ───────────────────────────────────────────────

def _z_dist_fig(z_stat: float, alpha: float, tails: str = "right",
                title: str = "") -> go.Figure:
    """
    Curva normal estándar mostrando región de rechazo, Z estadístico y p-valor.
    tails: 'right' (cola derecha), 'left' (cola izquierda).
    """
    x = np.linspace(-4.5, 4.5, 600)
    y = norm.pdf(x)
    z_crit = norm.ppf(1 - alpha) if tails == "right" else norm.ppf(alpha)

    fig = go.Figure()

    # Curva completa
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines",
                             line=dict(color="#2c3e50", width=2),
                             name="N(0,1)", showlegend=False))

    # Región de rechazo
    if tails == "right":
        x_rej = x[x >= z_crit]
        y_rej = norm.pdf(x_rej)
        label_rej = f"Región de rechazo (Z ≥ {z_crit:.3f})"
    else:
        x_rej = x[x <= z_crit]
        y_rej = norm.pdf(x_rej)
        label_rej = f"Región de rechazo (Z ≤ {z_crit:.3f})"

    fig.add_trace(go.Scatter(
        x=np.concatenate([[x_rej[0]], x_rej, [x_rej[-1]]]),
        y=np.concatenate([[0], y_rej, [0]]),
        fill="toself", fillcolor="rgba(231,76,60,0.3)",
        line=dict(color="rgba(231,76,60,0)"),
        name=label_rej,
    ))

    # Z crítico
    fig.add_vline(x=z_crit, line=dict(color="#e74c3c", dash="dash", width=2))
    fig.add_annotation(
        x=z_crit, y=1, xref="x", yref="paper",
        text=f"Z crítico = {z_crit:.3f}",
        showarrow=False,
        xanchor="right" if tails == "left" else "left",
        yanchor="bottom",
        font=dict(color="#e74c3c", size=11),
    )

    # Z estadístico
    z_clamp = max(-4.4, min(4.4, z_stat))
    fig.add_vline(x=z_clamp, line=dict(color="#27ae60", dash="solid", width=2.5))
    fig.add_annotation(
        x=z_clamp, y=0.88, xref="x", yref="paper",
        text=f"Z = {z_stat:.3f}",
        showarrow=False,
        xanchor="left" if tails == "right" else "right",
        yanchor="bottom",
        font=dict(color="#27ae60", size=12, family="Arial Bold"),
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        xaxis=dict(title="Valor Z", showgrid=False, range=[-4.5, 4.5]),
        yaxis=dict(title="Densidad", gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=70, b=40),
        height=320,
    )
    return fig


# ── Hipótesis 1: cantidad de kioskos (SHOT vs. modelo estadístico) ────────────

def test_h1_cantidad_kioskos(df_encuesta: pd.DataFrame, forecast: dict) -> dict:
    """
    H0: La demanda estadística justifica 50 kioskos (plan masa SHOT).
    H1: La demanda estadística justifica un número significativamente menor a 50 kioskos.
    Método: Z-test de una muestra sobre el número de kioskos estimado por demanda.
    El estimador es N_kioskos = Σ(visitantes_zona) × p̂_consumo / cap_kiosko_anual;
    la variabilidad proviene de p̂_consumo (proporción muestral de la encuesta).
    """
    alpha = 0.05
    mu0 = 50  # valor del plan masa SHOT

    p_hat = forecast["tasas"]["tasa_consumo"]
    n_enc = int(df_encuesta["consumiria"].dropna().shape[0])
    cap = forecast["capacidad_kiosko"]["clientes_anuales_por_kiosko"]
    total_vis = sum(z["visitantes_anuales_estimados"] for z in forecast["zonas"].values())

    N_dem = total_vis * p_hat / cap if cap > 0 else 0
    SE_N = total_vis * math.sqrt(p_hat * (1 - p_hat) / n_enc) / cap if cap > 0 else 1

    z_stat = (N_dem - mu0) / SE_N
    p_valor = float(norm.cdf(z_stat))  # cola izquierda (H1: μ < 50)
    rechaza_H0 = p_valor < alpha
    z_crit = float(norm.ppf(alpha))

    return {
        "id": "H1",
        "pregunta": "¿Se deben mantener 50 kioskos según el plan masa de la SHOT?",
        "H0": "La demanda estadística justifica ≥ 50 kioskos (plan masa SHOT es correcto)",
        "H1_alt": "La demanda estadística justifica < 50 kioskos (plan masa SHOT sobredimensiona la oferta)",
        "test": "Z-test de una muestra (cola izquierda)",
        "alpha": alpha,
        "z_stat": round(z_stat, 3),
        "z_crit": round(z_crit, 3),
        "p_valor": round(max(p_valor, 1e-10), 10),
        "rechaza_H0": rechaza_H0,
        "decision": "✅ Se rechaza H0" if rechaza_H0 else "❌ No se rechaza H0",
        "conclusion": (
            f"Con un nivel de significancia α = {alpha}, <b>se rechaza H0</b>. "
            f"El modelo estadístico basado en la encuesta estima <b>{N_dem:.1f} kioskos</b> "
            f"(IC 95%: {N_dem - 1.96*SE_N:.1f} – {N_dem + 1.96*SE_N:.1f}), "
            f"significativamente por debajo de los 50 del plan SHOT (Z = {z_stat:.1f}). "
            f"La demanda real y las restricciones físicas respaldan un plan de "
            f"<b>10 kioskos en 2026</b>, escalando hasta <b>25 en 2036</b> conforme crece el sector."
        ) if rechaza_H0 else (
            f"Con α = {alpha}, no se rechaza H0. La demanda estimada ({N_dem:.1f} kioskos) "
            f"es compatible con el plan de 50 kioskos de la SHOT."
        ),
        # Respuesta directa a la pregunta (para resumen ejecutivo)
        "respuesta_directa": (
            "NO — Los 50 kioskos del plan SHOT no están justificados estadísticamente. "
            f"El modelo técnico recomienda arrancar con <b>10 kioskos en 2026</b> "
            f"y escalar hasta <b>25 en 2036</b> conforme crece el sector."
        ) if rechaza_H0 else (
            "SÍ — La demanda estadística es compatible con mantener 50 kioskos."
        ),
        "respuesta_icono": "❌" if rechaza_H0 else "✅",
        "respuesta_color_bg":  "#fde8e8" if rechaza_H0 else "#d5f5e3",
        "respuesta_color_brd": "#e74c3c" if rechaza_H0 else "#27ae60",
        # Valores para la figura
        "_N_dem": round(N_dem, 2),
        "_SE_N": round(SE_N, 3),
        "_mu0": mu0,
        "_tails": "left",
        "_p_hat": round(p_hat, 3),
        "_n_enc": n_enc,
        "_total_vis": total_vis,
        "_cap_kiosko": cap,
    }


# ── Hipótesis 2: aprobación de kioskos ───────────────────────────────────────

def test_h2_aprobacion_kioskos(df_encuesta: pd.DataFrame) -> dict:
    """
    H0: La proporción de aprobación de kioskos ≤ 0.50 (no hay mayoría de apoyo).
    H1: La proporción de aprobación de kioskos > 0.50 (existe mayoría significativa de apoyo).
    Método: Z-test de una proporción (cola derecha).
    Variable: '¿Considera adecuada la implementación de kioskos comerciales?' = 'Sí'.
    """
    alpha = 0.05
    p0 = 0.50  # hipótesis nula: mitad o menos aprueba

    serie = df_encuesta["kiosko_adecuado"].dropna().str.strip().str.lower()
    n = len(serie)
    n_si = serie.isin(["sí", "si"]).sum()
    p_hat = n_si / n if n > 0 else 0

    z_stat = (p_hat - p0) / math.sqrt(p0 * (1 - p0) / n) if n > 0 else 0
    p_valor = float(1 - norm.cdf(z_stat))  # cola derecha (H1: p > 0.5)
    rechaza_H0 = p_valor < alpha
    z_crit = float(norm.ppf(1 - alpha))

    return {
        "id": "H2",
        "pregunta": "¿La gente está de acuerdo con implementar kioskos en el Parque Bicentenario?",
        "H0": "La proporción de aprobación de kioskos es ≤ 50% (no hay mayoría de apoyo)",
        "H1_alt": "La proporción de aprobación de kioskos es > 50% (existe mayoría significativa de apoyo)",
        "test": "Z-test de una proporción (cola derecha)",
        "alpha": alpha,
        "z_stat": round(z_stat, 3),
        "z_crit": round(z_crit, 3),
        "p_valor": round(p_valor, 8),
        "rechaza_H0": rechaza_H0,
        "decision": "✅ Se rechaza H0" if rechaza_H0 else "❌ No se rechaza H0",
        "conclusion": (
            f"Con α = {alpha}, <b>se rechaza H0</b>. "
            f"El <b>{p_hat*100:.1f}%</b> de los encuestados ({n_si} de {n}) considera adecuada "
            f"la implementación de kioskos, una proporción estadísticamente superior al 50% "
            f"(Z = {z_stat:.3f}, p = {p_valor:.2e}). "
            f"Existe evidencia suficiente para afirmar que la comunidad "
            f"<b>apoya mayoritariamente la implementación de kioskos comerciales</b> "
            f"en el Parque Bicentenario."
        ) if rechaza_H0 else (
            f"Con α = {alpha}, no se rechaza H0. La proporción de aprobación "
            f"({p_hat*100:.1f}%) no supera significativamente el 50%."
        ),
        # Respuesta directa a la pregunta (para resumen ejecutivo)
        "respuesta_directa": (
            f"SÍ — El <b>{p_hat*100:.1f}%</b> de los encuestados aprueba los kioskos. "
            f"Existe mayoría estadísticamente comprobada (p = {p_valor:.2e})."
        ) if rechaza_H0 else (
            f"INCONCLUSO — La aprobación ({p_hat*100:.1f}%) no supera el 50% con significancia estadística."
        ),
        "respuesta_icono": "✅" if rechaza_H0 else "❌",
        "respuesta_color_bg":  "#d5f5e3" if rechaza_H0 else "#fde8e8",
        "respuesta_color_brd": "#27ae60" if rechaza_H0 else "#e74c3c",
        # Valores para la figura
        "_p_hat": round(p_hat, 4),
        "_p0": p0,
        "_n": n,
        "_n_si": int(n_si),
        "_tails": "right",
    }


# ── Figura resumen de los dos tests ──────────────────────────────────────────

def fig_h1_distribucion(resultado: dict) -> go.Figure:
    return _z_dist_fig(
        z_stat=resultado["z_stat"],
        alpha=resultado["alpha"],
        tails="left",
        title="H1 – Distribución Z: ¿la demanda justifica 50 kioskos?",
    )


def fig_h2_distribucion(resultado: dict) -> go.Figure:
    return _z_dist_fig(
        z_stat=resultado["z_stat"],
        alpha=resultado["alpha"],
        tails="right",
        title="H2 – Distribución Z: ¿existe mayoría de aprobación de kioskos?",
    )


def fig_h1_intervalo_confianza(resultado: dict) -> go.Figure:
    """Gráfico de intervalo de confianza del estimador N_kioskos vs. la meta de 50."""
    N = resultado["_N_dem"]
    SE = resultado["_SE_N"]
    mu0 = resultado["_mu0"]
    ci_low, ci_high = N - 1.96 * SE, N + 1.96 * SE

    fig = go.Figure()
    # Línea de la meta SHOT
    fig.add_shape(type="line", x0=mu0, y0=0.1, x1=mu0, y1=0.9,
                  xref="x", yref="paper",
                  line=dict(color="#e74c3c", dash="dash", width=2))
    fig.add_annotation(x=mu0, y=0.95, xref="x", yref="paper",
                       text=f"<b>Meta SHOT: {mu0} kioskos</b>",
                       showarrow=False, font=dict(color="#e74c3c", size=12))

    # Punto estimado + CI
    fig.add_trace(go.Scatter(
        x=[N], y=[0.5],
        error_x=dict(type="data",
                     array=[ci_high - N], arrayminus=[N - ci_low],
                     color="#27ae60", thickness=3, width=12),
        mode="markers",
        marker=dict(color="#27ae60", size=14, symbol="diamond"),
        name=f"Estimación: {N:.1f} kioskos (IC 95%: {ci_low:.1f}–{ci_high:.1f})",
    ))
    fig.add_annotation(x=N, y=0.62, xref="x", yref="paper",
                       text=f"<b>{N:.1f}</b> kioskos estimados",
                       showarrow=True, ay=20, font=dict(color="#27ae60", size=12))

    fig.update_layout(
        title=dict(text="Intervalo de confianza 95% del número de kioskos estimado",
                   font=dict(size=14), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=12),
        xaxis=dict(title="Número de kioskos", showgrid=True, gridcolor="#f0f0f0",
                   range=[max(0, ci_low - 5), mu0 + 10]),
        yaxis=dict(showgrid=False, showticklabels=False, range=[0, 1]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=70, b=50),
        height=280,
    )
    return fig


def fig_h2_proporciones(resultado: dict) -> go.Figure:
    """Gauge con la proporción de aprobación vs. umbral del 50%."""
    p_hat = resultado["_p_hat"]
    n = resultado["_n"]
    n_si = resultado["_n_si"]
    p0 = resultado["_p0"]
    SE = math.sqrt(p0 * (1 - p0) / n)
    ci_low = max(0, p_hat - 1.96 * math.sqrt(p_hat * (1 - p_hat) / n))
    ci_high = min(1, p_hat + 1.96 * math.sqrt(p_hat * (1 - p_hat) / n))

    categorias = ["Aprueba kioskos (Sí)", "No aprueba kioskos (No)"]
    valores = [n_si, n - n_si]
    colores = ["#27ae60", "#e74c3c"]

    fig = go.Figure(go.Bar(
        x=categorias, y=valores,
        marker_color=colores,
        text=[f"<b>{v}</b> ({v/n*100:.1f}%)" for v in valores],
        textposition="outside",
        textfont=dict(size=13),
    ))
    # Línea del 50%
    fig.add_hline(y=n * p0, line=dict(color="#e67e22", dash="dot", width=2))
    fig.add_annotation(
        x=1, y=n * p0, xref="paper", yref="y",
        text="Umbral H0: 50%",
        showarrow=False, xanchor="right", yanchor="bottom",
        font=dict(color="#e67e22", size=11),
    )

    fig.update_layout(
        title=dict(text=f"Aprobación de kioskos — IC 95%: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]",
                   font=dict(size=14), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=12),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Número de encuestados", gridcolor="#f0f0f0"),
        margin=dict(l=40, r=20, t=70, b=40),
        height=300,
    )
    return fig
