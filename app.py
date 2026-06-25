"""
Estudio de Mercado – Kioskos Parque Bicentenario
Aplicación Streamlit principal
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "analysis"))
sys.path.insert(0, str(Path(__file__).parent / "models"))

import streamlit as st
import pandas as pd
import plotly.express as px
import base64

from pipeline import run_pipeline, get_kpi_summary
from analysis.univariate import (
    hist_gaussian, hist_rangos, bar_categorica, pie_categorica,
    table_frecuencia, table_stats_continua, bar_multiselect,
    table_multiselect, bar_calificacion,
)
from analysis.bivariate import (
    crosstab_heatmap, crosstab_stacked, crosstab_grouped,
    boxplot_continua_vs_cat, stats_tabla_bivariada,
    consumo_vs_gasto, chi2_test, anova_test,
)
from analysis.multivariate import (
    pca_biplot, scree_plot, kmeans_clusters, elbow_silhouette,
    correlation_matrix, pca_loadings_df, top_correlations,
)
from analysis.sentiment import (
    analyze_comments, sentiment_pie, topics_bar,
    wordcloud_img, top_keywords, sentiment_by_topic_heatmap,
)
from analysis.hypothesis import (
    test_h1_cantidad_kioskos, test_h2_aprobacion_kioskos,
    fig_h1_distribucion, fig_h2_distribucion,
    fig_h1_intervalo_confianza, fig_h2_proporciones,
)
from models.kiosko_model import (
    forecast_kioskos, fig_kioskos_por_zona,
    fig_visitantes_vs_demanda, fig_ingresos_potenciales,
    tabla_resumen_forecast, ZONAS,
    ZONE_DIMENSIONS, KIOSKO_TIPOS,
    GIROS, GIRO_ROTATION, COMMERCIAL_PARAMS,
    PROYECCION_PARAMS, PLAN_FASES,
    spatial_analysis_all, recommend_kiosko_type,
    fig_layout_zona, tabla_espacial_resumen,
    _max_kioskos_por_longitud, _ancho_util_por_kiosko,
    assign_giros, commercial_distance_analysis,
    fig_distribucion_comercial, tabla_distribucion_comercial,
    fig_trafico_parque_total, fig_demanda_vs_plan_parque,
    fig_plan_fases, tabla_plan_fases,
    fig_demanda_por_giro, demanda_por_giro,
)
from data_cleaner import GASTO_ORDER, FRECUENCIA_ORDER

# ── Configuración de página ────────────────────────────────────────────────────
_escudo_path = Path(__file__).parent / "src" / "img" / "logo_escudo.png"
st.set_page_config(
    page_title="Comercio Parque Bicentenario",
    page_icon=str(_escudo_path) if _escudo_path.exists() else "🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _footer() -> None:
    """Footer institucional flotante — logo arriba, texto abajo."""
    logo_html = ""
    if _escudo_path.exists():
        with open(_escudo_path, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode()
        logo_html = (
            f'<img src="data:image/png;base64,{_b64}" '
            f'style="height:48px;display:block;margin:0 auto 0.3rem;" alt="EPMMOP">'
        )
    st.markdown(f"""
    <style>
        .main .block-container {{ padding-bottom: 100px !important; }}
    </style>
    <div style="
        position: fixed;
        bottom: 0; left: 0; right: 0;
        z-index: 9999;
        background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 100%);
        color: white;
        padding: 0.55rem 1rem 0.45rem;
        text-align: center;
        box-shadow: 0 -2px 12px rgba(0,0,0,0.3);
        font-family: Arial, sans-serif;
    ">
        {logo_html}
        <div style="font-size:0.75rem;font-weight:600;letter-spacing:0.05em;line-height:1.4;">
            TODOS LOS DERECHOS RESERVADOS 2026 &nbsp;|&nbsp; EPMMOP
            &nbsp;|&nbsp; GERENCIA COMERCIAL &nbsp;|&nbsp; DPS
        </div>
        <div style="font-size:0.65rem;opacity:0.7;margin-top:0.1rem;">
            Powered by: JOGA
        </div>
    </div>
    """, unsafe_allow_html=True)


def _header(titulo: str, subtitulo: str) -> None:
    """Header institucional con escudo EPMMOP."""
    logo_html = ""
    if _escudo_path.exists():
        with open(_escudo_path, "rb") as _f:
            _b64 = base64.b64encode(_f.read()).decode()
        logo_html = (
            f'<img src="data:image/png;base64,{_b64}" '
            f'style="height:56px;margin-right:1.2rem;flex-shrink:0;">'
        )
    st.markdown(f"""
    <div class="main-header">
        <div style="display:flex;align-items:center;">
            {logo_html}
            <div>
                <h1 style="margin:0;font-size:1.6rem;line-height:1.2;">{titulo}</h1>
                <p style="margin:0.4rem 0 0;opacity:0.85;font-size:0.9rem;">{subtitulo}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _insight(texto: str, significativo=None) -> None:
    """Caja de interpretación. significativo=True → verde ✅, False → rojo ❌, None → azul 🔎."""
    if significativo is None:
        icono, bg, borde = "🔎", "#eaf4fb", "#2980b9"
    elif significativo:
        icono, bg, borde = "✅", "#eafaf1", "#27ae60"
    else:
        icono, bg, borde = "❌", "#fde8e8", "#e74c3c"
    st.markdown(
        f'<div style="background:{bg};border-left:4px solid {borde};'
        f'padding:0.8rem 1rem;border-radius:0 8px 8px 0;margin:0.8rem 0;font-size:0.92rem;">'
        f'{icono} <b>Interpretación:</b> {texto}</div>',
        unsafe_allow_html=True,
    )


def _top_categoria(series: pd.Series) -> tuple[str, float]:
    """Devuelve (categoría más frecuente, % que representa)."""
    counts = series.dropna().value_counts()
    if counts.empty:
        return "", 0.0
    top = counts.index[0]
    pct = round(counts.iloc[0] / counts.sum() * 100, 1)
    return str(top), pct


def _pct_si(series: pd.Series) -> float:
    """% de respuestas afirmativas ('sí'/'si')."""
    s = series.dropna().str.strip().str.lower()
    return round(s.isin(["sí", "si"]).sum() / len(s) * 100, 1) if len(s) else 0.0


# ── CSS personalizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2980b9 100%);
        color: white; padding: 1.5rem 2rem; border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }
    .kpi-card {
        background: white; border-radius: 10px; padding: 1.2rem 1rem;
        text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid #2980b9;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1a3a5c; }
    .kpi-label { font-size: 0.78rem; color: #666; margin-top: 0.2rem; }
    .section-header {
        color: #1a3a5c; font-size: 1.3rem; font-weight: 600;
        border-left: 4px solid #2980b9; padding-left: 0.7rem;
        margin: 1.5rem 0 1rem;
    }
    .insight-box {
        background: #eaf4fb; border-left: 4px solid #2980b9;
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0; margin: 0.8rem 0;
        font-size: 0.92rem;
    }
    .zona-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 1rem;
    }
    div[data-testid="stMetricValue"] { font-size: 1.6rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Carga de datos ─────────────────────────────────────────────────────────────
ctx = run_pipeline()
enc = ctx["encuesta"]
stats = ctx["estadisticas"]
kpis = get_kpi_summary(ctx)

# ── Barra lateral ──────────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = Path(__file__).parent / "src" / "img" / "logo_color.png"
    if logo_path.exists():
        st.image(str(logo_path), width="stretch")
    st.markdown("""
    <div style="text-align:center;padding:0.2rem 0 1rem;">
        <div style="font-size:1.1rem;font-weight:700;color:#1a3a5c;">GERENCIA COMERCIAL</div>
        <div style="font-size:0.75rem;color:#666;">Promoción de Servicios</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### Navegación")
    pagina = st.radio(
        "Seleccione sección:",
        [
            "🏠 Resumen Ejecutivo",
            "📊 Análisis Univariado",
            "🔗 Análisis Bivariado",
            "🌐 Análisis Multivariado",
            "💬 Recomendaciones",
            "📐 Prueba de Hipótesis",
            "🏪 Modelo comercial",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"**Encuestados:** {kpis['n_encuestados']}")
    st.markdown(f"**Fecha análisis:** Junio 2026")
    st.markdown(f"**Ubicación:** Parque Bicentenario, Quito")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠 Resumen Ejecutivo":
    _header(
        "Estudio de Mercado – Modelo Comercial Parque Bicentenario",
        "Análisis integral de viabilidad comercial · Parque Metropolitano Bicentenario · Quito, 2026",
    )

    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{kpis['n_encuestados']}</div>
            <div class="kpi-label">Encuestados</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{kpis['consumiria_pct']}%</div>
            <div class="kpi-label">Consumiría en el parque</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{kpis['aprueba_kiosko_pct']}%</div>
            <div class="kpi-label">Aprueba kioskos</div></div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{kpis['calificacion_promedio']}/5</div>
            <div class="kpi-label">Calificación actual del parque</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{kpis['edad_promedio']} años</div>
            <div class="kpi-label">Edad promedio del visitante</div></div>""", unsafe_allow_html=True)
    with col6:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">${kpis['gasto_promedio_usd']}</div>
            <div class="kpi-label">Gasto promedio dispuesto</div></div>""", unsafe_allow_html=True)
    with col7:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{kpis['mejora_experiencia_pct']}%</div>
            <div class="kpi-label">Cree que kioskos mejorarían exp.</div></div>""", unsafe_allow_html=True)
    with col8:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{kpis['total_visitas_parque_2025']:,}</div>
            <div class="kpi-label">Visitas totales 2025</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── OBJETIVOS DEL ESTUDIO ────────────────────────────────────────────────
    st.markdown('<p class="section-header">Objetivos del estudio</p>', unsafe_allow_html=True)

    obj_col1, obj_col2 = st.columns(2)
    _objetivos = [
        ("OBJ 1", "#2980b9", "Determinar la viabilidad estadística del plan de 50 kioskos",
         "Evaluar si la demanda real estimada por el modelo estadístico justifica la cifra "
         "de 50 kioskos propuesta por el plan masa de la SHOT, o si ese número sobredimensiona "
         "la oferta comercial.",
         "→ Validado con <b>Hipótesis H1</b>"),
        ("OBJ 2", "#27ae60", "Medir la aceptación ciudadana de los kioskos comerciales",
         "Cuantificar el porcentaje de visitantes que considera adecuada la implementación "
         "de kioskos en el Parque Bicentenario y contrastar si ese respaldo supera "
         "estadísticamente el umbral del 50 %.",
         "→ Validado con <b>Hipótesis H2</b>"),
        ("OBJ 3", "#8e44ad", "Caracterizar el perfil del visitante habitual",
         "Identificar las variables sociodemográficas (edad, género, frecuencia de visita, "
         "horario, acompañantes) y de comportamiento de consumo que definen a los distintos "
         "segmentos de visitantes del parque.",
         "→ Respaldado con <b>análisis univariado y bivariado</b>"),
        ("OBJ 4", "#e67e22", "Proponer un modelo comercial basado en la demanda real",
         "Identificar los giros comerciales con mayor respaldo en las preferencias de consumo "
         "de los encuestados y traducirlos en un plan de implementación por fases (2026–2036) "
         "alineado con la capacidad física de cada zona del parque.",
         "→ Desarrollado en <b>Modelo Comercial</b>"),
    ]
    for i, (etiqueta, color, titulo, desc, vinculo) in enumerate(_objetivos):
        with (obj_col1 if i % 2 == 0 else obj_col2):
            st.markdown(f"""
            <div style="background:#f8f9fa;border-radius:10px;padding:1rem 1.1rem;
                        border-left:5px solid {color};margin-bottom:0.8rem;">
                <div style="font-size:0.7rem;font-weight:700;color:{color};
                            text-transform:uppercase;letter-spacing:0.08em;">{etiqueta}</div>
                <div style="font-size:0.92rem;font-weight:600;color:#1a3a5c;
                            margin:0.25rem 0 0.4rem;">{titulo}</div>
                <div style="font-size:0.83rem;color:#444;line-height:1.5;">{desc}</div>
                <div style="font-size:0.78rem;color:{color};margin-top:0.5rem;">{vinculo}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── DISEÑO METODOLÓGICO Y CÁLCULO DE MUESTRA ────────────────────────────
    st.markdown('<p class="section-header">Diseño metodológico y cálculo de la muestra</p>',
                unsafe_allow_html=True)

    import math as _math
    _N_univ  = kpis["total_visitas_parque_2025"]   # universo: visitas anuales 2025
    _Z_conf  = 1.96   # nivel de confianza 95 %
    _p_var   = 0.50   # proporción esperada (máxima varianza)
    _q_var   = 0.50
    _e_err   = 0.07   # margen de error 7 %
    _n0      = (_Z_conf**2 * _p_var * _q_var) / (_e_err**2)                     # pob. infinita
    _n_corr  = _n0 / (1 + (_n0 - 1) / _N_univ)                                  # corrección finita
    _n_form  = int(_math.ceil(_n_corr))                                           # redondeado
    _n_real  = kpis["n_encuestados"]                                              # encuestas reales

    met_col1, met_col2 = st.columns([1.1, 1])
    with met_col1:
        st.markdown(f"""
        <div style="background:#f0f4fa;border-radius:10px;padding:1.1rem 1.3rem;">
            <div style="font-weight:600;color:#1a3a5c;font-size:0.95rem;margin-bottom:0.7rem;">
                Parámetros del diseño
            </div>
            <table style="width:100%;font-size:0.84rem;border-collapse:collapse;">
                <tr><td style="padding:0.25rem 0.5rem 0.25rem 0;color:#555;width:55%;">Universo (visitas anuales 2025)</td>
                    <td style="font-weight:600;color:#1a3a5c;"><b>{_N_univ:,}</b></td></tr>
                <tr><td style="padding:0.25rem 0.5rem 0.25rem 0;color:#555;">Nivel de confianza</td>
                    <td style="font-weight:600;color:#1a3a5c;"><b>95 % (Z = {_Z_conf})</b></td></tr>
                <tr><td style="padding:0.25rem 0.5rem 0.25rem 0;color:#555;">Proporción esperada</td>
                    <td style="font-weight:600;color:#1a3a5c;"><b>p = q = {_p_var}</b> (máxima varianza)</td></tr>
                <tr><td style="padding:0.25rem 0.5rem 0.25rem 0;color:#555;">Margen de error</td>
                    <td style="font-weight:600;color:#1a3a5c;"><b>e = {int(_e_err*100)} %</b></td></tr>
                <tr><td style="padding:0.25rem 0.5rem 0.25rem 0;color:#555;">Tipo de muestreo</td>
                    <td style="font-weight:600;color:#1a3a5c;">Aleatorio simple</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    with met_col2:
        st.markdown(f"""
        <div style="background:#f0f4fa;border-radius:10px;padding:1.1rem 1.3rem;height:100%;">
            <div style="font-weight:600;color:#1a3a5c;font-size:0.95rem;margin-bottom:0.7rem;">
                Fórmula de cálculo
            </div>
            <div style="font-size:0.82rem;color:#333;text-align:center;
                        background:white;border-radius:8px;padding:0.7rem;margin-bottom:0.6rem;
                        font-family:monospace;line-height:1.8;">
                &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Z² · p · q · N<br>
                n = ─────────────────────<br>
                &nbsp;&nbsp;e² · (N–1) + Z² · p · q
            </div>
            <div style="font-size:0.82rem;color:#333;text-align:center;
                        background:white;border-radius:8px;padding:0.6rem;margin-bottom:0.6rem;
                        font-family:monospace;line-height:1.8;">
                &nbsp;&nbsp;{_Z_conf}² · {_p_var} · {_q_var} · {_N_univ:,}<br>
                n = ─────────────────────────────<br>
                &nbsp;&nbsp;{_e_err}² · {_N_univ-1:,} + {_Z_conf}² · {_p_var} · {_q_var}
            </div>
            <div style="font-size:0.88rem;text-align:center;">
                <span style="color:#888;">Resultado fórmula: </span>
                <b style="color:#2980b9;">{_n_corr:.1f} encuestas</b>
                &nbsp;→&nbsp;
                <span style="background:#27ae60;color:white;border-radius:6px;
                             padding:0.15rem 0.5rem;font-weight:700;font-size:0.92rem;">
                    {_n_real} encuestas realizadas
                </span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Distribución del perfil del visitante
    st.markdown('<p class="section-header">Perfil del visitante encuestado</p>', unsafe_allow_html=True)

    # Fila 1: género (donut) + frecuencia (barras horizontales con altura propia)
    col_a, col_b = st.columns([1, 2])
    with col_a:
        if "genero" in enc.columns:
            st.plotly_chart(pie_categorica(enc["genero"], "Género"), width="stretch")
    with col_b:
        if "frecuencia_visita" in enc.columns:
            st.plotly_chart(
                bar_categorica(enc["frecuencia_visita"], "Frecuencia de visita",
                               order=FRECUENCIA_ORDER, horizontal=True),
                width="stretch"
            )

    # Fila 2: motivo de visita (donut) — tiene muchas categorías, necesita su propio espacio
    if "motivo_visita" in enc.columns:
        st.plotly_chart(pie_categorica(enc["motivo_visita"], "Motivo principal de visita"),
                        width="stretch")

    # Estadísticas del parque
    st.markdown('<p class="section-header">Estadísticas de uso del Parque Bicentenario 2025</p>',
                unsafe_allow_html=True)

    stats_summary = []
    for nombre, df_s in stats.items():
        if "total" in df_s.columns:
            total_anual = df_s["total"].sum()
            pico = df_s[["sabado", "domingo"]].sum(axis=1).max() if all(
                c in df_s.columns for c in ["sabado", "domingo"]) else 0
            stats_summary.append({
                "Zona / Actividad": nombre,
                "Visitas anuales": f"{int(total_anual):,}",
                "Pico fin de semana (semana más alta)": f"{int(pico):,}",
            })
    if stats_summary:
        st.dataframe(pd.DataFrame(stats_summary), width="stretch", hide_index=True)

    # Visitas mensuales – pista de ciclismo como referencia
    if "USO DE LA PISTA DE CICLISMO" in stats:
        df_pista = stats["USO DE LA PISTA DE CICLISMO"]
        st.markdown('<p class="section-header">Visitas mensuales – Pista de Ciclismo (actividad principal)</p>',
                    unsafe_allow_html=True)
        fig_mes = px.bar(df_pista, x="mes", y="total", color_discrete_sequence=["#2980b9"],
                         labels={"mes": "Mes", "total": "Visitas"},
                         text="total")
        fig_mes.update_traces(textposition="outside")
        fig_mes.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               xaxis=dict(showgrid=False), yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig_mes, width="stretch")

    # ── HALLAZGOS ANÁLISIS UNIVARIADO ───────────────────────────────────────
    st.markdown('<p class="section-header">Hallazgos del análisis univariado – perfil y comportamiento</p>',
                unsafe_allow_html=True)

    # ── helpers locales ──────────────────────────────────────────────────────
    def _top1(col):
        s = enc[col].dropna()
        if s.empty: return "–", 0.0
        v = s.value_counts()
        return str(v.index[0]), round(v.iloc[0] / len(s) * 100, 1)

    def _pct_val(col, val):
        s = enc[col].dropna().str.strip().str.lower()
        return round(s.isin([val.lower()]).sum() / len(s) * 100, 1) if len(s) else 0.0

    # Calcular valores clave
    _gen_top, _gen_pct   = _top1("genero") if "genero" in enc.columns else ("–", 0)
    _frec_top, _frec_pct = _top1("frecuencia_visita") if "frecuencia_visita" in enc.columns else ("–", 0)
    _motivo_top, _motivo_pct = _top1("motivo_visita") if "motivo_visita" in enc.columns else ("–", 0)
    _acomp_top, _acomp_pct   = _top1("acompanante") if "acompanante" in enc.columns else ("–", 0)
    _gasto_top, _gasto_pct   = _top1("gasto_dispuesto") if "gasto_dispuesto" in enc.columns else ("–", 0)

    _edad_med    = round(enc["edad"].dropna().median(), 0) if "edad" in enc.columns else kpis["edad_promedio"]

    # Productos de interés (multiselect — tomar moda)
    if "productos_interes" in enc.columns:
        _prod_counts = enc["productos_interes"].dropna().value_counts()
        _prod_top  = str(_prod_counts.index[0]) if len(_prod_counts) else "–"
        _prod_top2 = str(_prod_counts.index[1]) if len(_prod_counts) > 1 else ""
        _prod_top3 = str(_prod_counts.index[2]) if len(_prod_counts) > 2 else ""
        _prod_pct  = round(_prod_counts.iloc[0] / len(enc["productos_interes"].dropna()) * 100, 1) if len(_prod_counts) else 0
    else:
        _prod_top, _prod_top2, _prod_top3, _prod_pct = "–", "", "", 0

    _calif_med = round(enc["calificacion_oferta"].dropna().mean(), 2) if "calificacion_oferta" in enc.columns else kpis["calificacion_promedio"]
    _consumo_pct = kpis["consumiria_pct"]
    _aprueba_pct = kpis["aprueba_kiosko_pct"]
    _mejora_pct  = kpis["mejora_experiencia_pct"]

    # Tarjetas univariadas en dos bloques
    st.markdown("**Perfil sociodemográfico del visitante**")
    _u_cols = st.columns(4)
    _uni_cards_1 = [
        ("Género predominante", _gen_top, f"{_gen_pct}% de encuestados", "#2980b9"),
        ("Edad mediana", f"{int(_edad_med)} años", f"Promedio: {kpis['edad_promedio']} años", "#2980b9"),
        ("Frecuencia de visita más común", _frec_top, f"{_frec_pct}% de encuestados", "#8e44ad"),
        ("Motivo de visita principal", _motivo_top, f"{_motivo_pct}% de encuestados", "#8e44ad"),
    ]
    for col_u, (titulo, valor, sub, color) in zip(_u_cols, _uni_cards_1):
        with col_u:
            st.markdown(f"""
            <div style="background:#f8f9fa;border-radius:10px;padding:0.9rem 1rem;
                        border-top:4px solid {color};text-align:center;min-height:110px;">
                <div style="font-size:0.72rem;color:#666;margin-bottom:0.3rem;">{titulo}</div>
                <div style="font-size:1.25rem;font-weight:700;color:#1a3a5c;line-height:1.2;">{valor}</div>
                <div style="font-size:0.75rem;color:#888;margin-top:0.3rem;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _cu1, _cu2, _cu3, _cu4 = st.columns(4)

    # Tarjeta 1: acompañante
    with _cu1:
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:10px;padding:0.9rem 1rem;
                    border-top:4px solid #e67e22;text-align:center;min-height:130px;">
            <div style="font-size:0.72rem;color:#666;margin-bottom:0.3rem;">Visita principalmente con</div>
            <div style="font-size:1.25rem;font-weight:700;color:#1a3a5c;">{_acomp_top}</div>
            <div style="font-size:0.75rem;color:#888;margin-top:0.3rem;">{_acomp_pct}% de encuestados</div>
        </div>""", unsafe_allow_html=True)

    # Tarjeta 2: horario semana vs fin de semana
    with _cu2:
        st.markdown("""
        <div style="background:#f8f9fa;border-radius:10px;padding:0.9rem 1rem;
                    border-top:4px solid #e67e22;min-height:130px;">
            <div style="font-size:0.72rem;color:#666;margin-bottom:0.6rem;text-align:center;">
                Horario de visita por tipo de día
            </div>
            <table style="width:100%;font-size:0.82rem;border-collapse:collapse;">
                <tr>
                    <td style="color:#555;padding:0.2rem 0;">📅 Lun–Vie</td>
                    <td style="font-weight:700;color:#1a3a5c;text-align:right;">04:00–09:00 h</td>
                </tr>
                <tr style="height:0.5rem;"></tr>
                <tr>
                    <td style="color:#555;padding:0.2rem 0;">🏖️ Sáb–Dom</td>
                    <td style="font-weight:700;color:#1a3a5c;text-align:right;">09:00–12:00 h</td>
                </tr>
            </table>
        </div>""", unsafe_allow_html=True)

    # Tarjeta 3: calificación
    with _cu3:
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:10px;padding:0.9rem 1rem;
                    border-top:4px solid #e74c3c;text-align:center;min-height:130px;">
            <div style="font-size:0.72rem;color:#666;margin-bottom:0.3rem;">Calificación actual del parque</div>
            <div style="font-size:1.25rem;font-weight:700;color:#1a3a5c;">{_calif_med}/5</div>
            <div style="font-size:0.75rem;color:#888;margin-top:0.3rem;">Escala 1 (muy mala) a 5 (muy buena)</div>
        </div>""", unsafe_allow_html=True)

    # Tarjeta 4: producto
    with _cu4:
        st.markdown(f"""
        <div style="background:#f8f9fa;border-radius:10px;padding:0.9rem 1rem;
                    border-top:4px solid #27ae60;text-align:center;min-height:130px;">
            <div style="font-size:0.72rem;color:#666;margin-bottom:0.3rem;">Producto / servicio más demandado</div>
            <div style="font-size:1.25rem;font-weight:700;color:#1a3a5c;">{_prod_top}</div>
            <div style="font-size:0.75rem;color:#888;margin-top:0.3rem;">{_prod_pct}% lo prefiere</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Actitudes hacia los kioskos comerciales**")
    _u_cols3 = st.columns(3)
    _uni_cards_3 = [
        ("Consumiría en el parque", f"{_consumo_pct}%", "Demanda potencial de servicios comerciales", "#27ae60"),
        ("Aprueba la implementación de kioskos", f"{_aprueba_pct}%", "Aceptación ciudadana directa", "#27ae60"),
        ("Cree que mejoraría la experiencia", f"{_mejora_pct}%", "Impacto percibido positivo", "#27ae60"),
    ]
    for col_u, (titulo, valor, sub, color) in zip(_u_cols3, _uni_cards_3):
        with col_u:
            st.markdown(f"""
            <div style="background:#eafaf1;border-radius:10px;padding:0.9rem 1rem;
                        border-top:4px solid {color};text-align:center;min-height:100px;">
                <div style="font-size:0.72rem;color:#666;margin-bottom:0.3rem;">{titulo}</div>
                <div style="font-size:1.6rem;font-weight:700;color:{color};line-height:1.2;">{valor}</div>
                <div style="font-size:0.75rem;color:#888;margin-top:0.3rem;">{sub}</div>
            </div>""", unsafe_allow_html=True)

    if _prod_top2 or _prod_top3:
        _prods_str = " · ".join(filter(None, [_prod_top, _prod_top2, _prod_top3]))
        st.markdown(f"""
        <div style="background:#f0f4fa;border-radius:8px;padding:0.7rem 1rem;
                    margin-top:0.5rem;font-size:0.85rem;">
            <b>Top productos/servicios de interés:</b> {_prods_str}
            &nbsp;&mdash;&nbsp;
            <b>Gasto dispuesto más frecuente:</b> {_gasto_top} ({_gasto_pct}% de encuestados)
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── HALLAZGOS ANÁLISIS BIVARIADO ─────────────────────────────────────────
    st.markdown('<p class="section-header">Hallazgos del análisis bivariado – relaciones entre variables</p>',
                unsafe_allow_html=True)

    # Ejecutar todos los tests bivariados
    _gasto_map = {"Menos de $2": 1, "Entre $2 y $5": 3.5,
                  "Entre $5 y $10": 7.5, "Entre $10 y $20": 15, "Más de $20": 25}
    _enc_g = enc.copy()
    if "gasto_dispuesto" in enc.columns:
        _enc_g["gasto_num"] = _enc_g["gasto_dispuesto"].map(_gasto_map)

    _biv_tests = []
    _safe_chi  = lambda c1, c2: chi2_test(enc, c1, c2) if (c1 in enc.columns and c2 in enc.columns) else None
    _safe_anov = lambda col_n, col_c: anova_test(_enc_g, col_n, col_c) if (col_n in _enc_g.columns and col_c in _enc_g.columns) else None

    _tests_def = [
        ("Género",              "¿Consumiría en el parque?",    _safe_chi("genero", "consumiria"),
         "Determina si el género influye en la propensión de consumo."),
        ("Género",              "Gasto dispuesto (ANOVA)",      _safe_anov("gasto_num", "genero"),
         "Evalúa si hombres y mujeres difieren en cuánto pagarían."),
        ("Frecuencia de visita","Motivo de visita",             _safe_chi("frecuencia_visita", "motivo_visita"),
         "Analiza si el propósito de la visita determina su regularidad."),
        ("Edad",                "Aprobación de kioskos (ANOVA)",_safe_anov("edad", "kiosko_adecuado"),
         "Verifica si la edad influye en la actitud hacia los kioskos."),
        ("Sector de residencia","Aprobación de kioskos",        _safe_chi("sector_residencia", "kiosko_adecuado"),
         "Investiga si la zona de origen condiciona la aprobación."),
        ("Acompañante",         "Gasto dispuesto",              _safe_chi("acompanante", "gasto_dispuesto"),
         "Relaciona con quién se visita el parque y el gasto potencial."),
        ("Calificación oferta", "Frecuencia de visita (ANOVA)", _safe_anov("calificacion_oferta", "frecuencia_visita"),
         "Evalúa si visitantes frecuentes perciben mejor la oferta actual."),
        ("Horario de visita",   "Género (ANOVA)",               _safe_anov("horario_visita", "genero"),
         "Analiza si hombres y mujeres visitan en horarios distintos."),
        ("¿Consumiría?",        "Gasto dispuesto",              _safe_chi("consumiria", "gasto_dispuesto"),
         "Relaciona disposición de consumo con el monto que pagarían."),
    ]

    # Tabla resumen de tests
    _biv_col1, _biv_col2 = st.columns([1.5, 1])
    with _biv_col1:
        st.markdown("**Tabla resumen de pruebas de asociación (α = 0.05)**")
        _filas_html = ""
        _n_sig = 0
        for (v1, v2, res, _) in _tests_def:
            if res is None:
                continue
            _sig = res["significativo"]
            _ico = "✅" if _sig else "❌"
            _col_fila = "#eafaf1" if _sig else "#fff5f5"
            _stat_str = f"χ²={res['chi2']}" if "chi2" in res else f"F={res['f']}"
            _p_str = f"p={res['p_valor']}"
            if _sig: _n_sig += 1
            _filas_html += f"""
            <tr style="background:{_col_fila};">
                <td style="padding:0.35rem 0.6rem;font-size:0.8rem;color:#444;">{v1}</td>
                <td style="padding:0.35rem 0.6rem;font-size:0.8rem;color:#444;">{v2}</td>
                <td style="padding:0.35rem 0.6rem;font-size:0.8rem;font-family:monospace;">{_stat_str}</td>
                <td style="padding:0.35rem 0.6rem;font-size:0.8rem;font-family:monospace;">{_p_str}</td>
                <td style="padding:0.35rem 0.6rem;font-size:0.85rem;text-align:center;">{_ico}</td>
            </tr>"""
        st.markdown(f"""
        <table style="width:100%;border-collapse:collapse;border-radius:8px;overflow:hidden;">
            <thead>
                <tr style="background:#1a3a5c;color:white;">
                    <th style="padding:0.5rem 0.6rem;font-size:0.78rem;text-align:left;">Variable 1</th>
                    <th style="padding:0.5rem 0.6rem;font-size:0.78rem;text-align:left;">Variable 2</th>
                    <th style="padding:0.5rem 0.6rem;font-size:0.78rem;text-align:left;">Estadístico</th>
                    <th style="padding:0.5rem 0.6rem;font-size:0.78rem;text-align:left;">p-valor</th>
                    <th style="padding:0.5rem 0.6rem;font-size:0.78rem;text-align:center;">Sig.</th>
                </tr>
            </thead>
            <tbody>{_filas_html}</tbody>
        </table>""", unsafe_allow_html=True)

    with _biv_col2:
        st.markdown("**Interpretaciones clave**")
        _n_total = sum(1 for _, _, r, _ in _tests_def if r is not None)
        st.markdown(f"""
        <div style="background:#f0f4fa;border-radius:10px;padding:0.9rem 1rem;margin-bottom:0.7rem;">
            <div style="font-size:0.78rem;color:#555;margin-bottom:0.3rem;">Relaciones estadísticamente significativas</div>
            <div style="font-size:1.8rem;font-weight:700;color:#2980b9;">{_n_sig} <span style="font-size:1rem;color:#888;">de {_n_total}</span></div>
        </div>""", unsafe_allow_html=True)

        for (v1, v2, res, desc) in _tests_def:
            if res is None or not res["significativo"]:
                continue
            st.markdown(f"""
            <div style="border-left:4px solid #27ae60;background:#f0fdf4;border-radius:0 8px 8px 0;
                        padding:0.5rem 0.8rem;margin-bottom:0.5rem;">
                <div style="font-size:0.78rem;font-weight:600;color:#27ae60;">✅ {v1} × {v2}</div>
                <div style="font-size:0.78rem;color:#444;margin-top:0.2rem;">{desc}</div>
                <div style="font-size:0.75rem;color:#27ae60;margin-top:0.2rem;">{res['interpretacion'][:100]}...</div>
            </div>""", unsafe_allow_html=True)

    # Nube de palabras de comentarios
    st.markdown('<p class="section-header">Lo que pide la ciudadanía (nube de palabras)</p>',
                unsafe_allow_html=True)
    if "comentarios" in enc.columns and enc["comentarios"].notna().sum() > 5:
        img_b64 = wordcloud_img(enc)
        st.markdown(
            f'<img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px;">',
            unsafe_allow_html=True,
        )
        st.caption("Palabras más frecuentes en comentarios y sugerencias de los encuestados")

    # ── PRUEBA DE HIPÓTESIS ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Validación estadística – Prueba de Hipótesis</p>',
                unsafe_allow_html=True)

    _fc_h = forecast_kioskos(enc, stats)
    _h1 = test_h1_cantidad_kioskos(enc, _fc_h)
    _h2 = test_h2_aprobacion_kioskos(enc)

    col_h1r, col_h2r = st.columns(2)

    for _col_h, _res in [(col_h1r, _h1), (col_h2r, _h2)]:
        _bg  = _res["respuesta_color_bg"]
        _brd = _res["respuesta_color_brd"]
        _ico = _res["respuesta_icono"]
        _p_fmt = f"{_res['p_valor']:.2e}" if _res['p_valor'] < 0.001 else f"{_res['p_valor']:.4f}"
        with _col_h:
            st.markdown(f"""
            <div style="background:{_bg};border-radius:12px;padding:1.2rem;
                        border-left:6px solid {_brd};height:100%;">
                <div style="font-size:0.78rem;font-weight:600;color:#555;
                            text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.4rem;">
                    {_res['id']} — {_res['pregunta']}
                </div>
                <div style="font-size:1.05rem;font-weight:700;color:{_brd};margin-bottom:0.6rem;
                            line-height:1.4;">
                    {_ico} {_res['respuesta_directa']}
                </div>
                <div style="font-size:0.78rem;color:#666;">
                    {_res['test']} &nbsp;|&nbsp; Z = {_res['z_stat']} &nbsp;|&nbsp; p = {_p_fmt}
                </div>
            </div>
            """, unsafe_allow_html=True)

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.plotly_chart(fig_h1_distribucion(_h1), width="stretch")
    with col_g2:
        st.plotly_chart(fig_h2_distribucion(_h2), width="stretch")

    _insight(
        f"<b>H1 (50 kioskos SHOT):</b> {_h1['conclusion']} &nbsp;&nbsp; "
        f"<b>H2 (aprobación ciudadana):</b> {_h2['conclusion']}",
        _h1["rechaza_H0"] and _h2["rechaza_H0"],
    )

    # ── MODELO COMERCIAL ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown('<p class="section-header">Modelo Comercial – Kioskos propuestos (2026–2036)</p>',
                unsafe_allow_html=True)

    col_chart, col_info = st.columns([3, 2])
    with col_chart:
        st.plotly_chart(fig_kioskos_por_zona(_fc_h), width="stretch")

    with col_info:
        _fase1 = min(PLAN_FASES.keys())
        _fase4 = max(PLAN_FASES.keys())
        _tot_ini = sum(PLAN_FASES[_fase1]["kioskos"].values())
        _tot_fin = sum(PLAN_FASES[_fase4]["kioskos"].values())
        st.markdown(f"""
        <div style="background:white;border-radius:10px;padding:1.2rem;
                    box-shadow:0 2px 8px rgba(0,0,0,0.08);border-top:4px solid #2980b9;">
            <div style="font-weight:700;color:#1a3a5c;margin-bottom:0.8rem;">
                Plan de implementación
            </div>
            <table style="width:100%;font-size:0.85rem;border-collapse:collapse;">
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:0.3rem 0;color:#666;">Arranque {_fase1}</td>
                <td style="font-weight:700;color:#27ae60;text-align:right;">{_tot_ini} kioskos</td>
            </tr>
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:0.3rem 0;color:#666;">Proyección {_fase4}</td>
                <td style="font-weight:700;color:#e67e22;text-align:right;">{_tot_fin} kioskos</td>
            </tr>
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:0.3rem 0;color:#666;">Bulevar {_fase4}</td>
                <td style="font-weight:700;color:#2980b9;text-align:right;">{PLAN_FASES[_fase4]['kioskos']['Zona 1 – Bulevar Av. Amazonas']} kioskos</td>
            </tr>
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:0.3rem 0;color:#666;">Arena {_fase4}</td>
                <td style="font-weight:700;color:#e67e22;text-align:right;">{PLAN_FASES[_fase4]['kioskos']['Zona 2 – Arena de Espectáculos']} kioskos</td>
            </tr>
            <tr>
                <td style="padding:0.3rem 0;color:#666;">Canchas {_fase4}</td>
                <td style="font-weight:700;color:#27ae60;text-align:right;">{PLAN_FASES[_fase4]['kioskos']['Zona 3 – Canchas Deportivas y Pistas']} kioskos</td>
            </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Propuesta de implantación — Parque Bicentenario")
    st.image("src/img/implantación_parque.jpg", use_container_width=True)
    st.markdown("#### Sectores a intervenir")
    st.image("src/img/implantación_circulos.jpg", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Giros comerciales propuestos y respaldo de la encuesta</p>',
                unsafe_allow_html=True)

    col_giros, col_dem = st.columns([2, 3])
    with col_giros:
        for _giro_name, _giro_info in GIROS.items():
            st.markdown(f"""
            <div style="display:flex;align-items:center;background:white;border-radius:8px;
                        padding:0.55rem 0.8rem;margin-bottom:0.5rem;
                        box-shadow:0 1px 4px rgba(0,0,0,0.07);
                        border-left:4px solid {_giro_info['color']};">
                <span style="font-size:1.3rem;margin-right:0.6rem;">{_giro_info['icono']}</span>
                <div>
                    <div style="font-weight:700;color:#1a3a5c;font-size:0.88rem;">{_giro_name}</div>
                    <div style="font-size:0.75rem;color:#666;">{_giro_info['descripcion'][:60]}…</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    with col_dem:
        st.plotly_chart(fig_demanda_por_giro(enc), width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: ANÁLISIS UNIVARIADO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Análisis Univariado":
    _header("Análisis Univariado", "Distribución y frecuencias de cada variable de la encuesta")

    variable = st.selectbox("Seleccione la variable a analizar:", [
        "Edad",
        "Horario de visita",
        "Género",
        "Sector de residencia",
        "Acompañante",
        "Frecuencia de visita",
        "Días que visita (multi-opción)",
        "Motivo de visita",
        "¿Consumiría productos?",
        "Productos de interés",
        "Gasto dispuesto",
        "¿Considera adecuados los kioskos?",
        "Zona de acceso preferida",
        "¿Kioskos mejorarían la experiencia?",
        "Calificación actual de oferta",
        "Servicios prioritarios (multi-opción)",
    ])

    st.markdown("---")

    if variable == "Edad":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(hist_gaussian(enc["edad"], "Edad", "años"), width="stretch")
        with col2:
            st.markdown("#### Estadísticos descriptivos")
            st.dataframe(table_stats_continua(enc["edad"]), width="stretch", hide_index=True)
        st.markdown("#### Distribución por rangos de edad")
        col3, col4 = st.columns([2, 1])
        with col3:
            if "rango_edad" in enc.columns:
                st.plotly_chart(hist_rangos(enc["edad"], enc["rango_edad"], "Edad"),
                                width="stretch")
        with col4:
            if "rango_edad" in enc.columns:
                df_rango = enc["rango_edad"].value_counts().sort_index().reset_index()
                df_rango.columns = ["Rango", "Frecuencia"]
                df_rango["%"] = (df_rango["Frecuencia"] / df_rango["Frecuencia"].sum() * 100).round(1)
                st.dataframe(df_rango, width="stretch", hide_index=True)
        top_rango, pct_rango = _top_categoria(enc["rango_edad"]) if "rango_edad" in enc.columns else ("", 0)
        _insight(f"La edad promedio de los visitantes es de **{enc['edad'].mean():.0f} años**; "
                 f"el grupo predominante es **{top_rango}** ({pct_rango}%), un dato clave para "
                 f"definir el tipo de producto y rango de precio de los kioskos.")

    elif variable == "Horario de visita":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                hist_gaussian(enc["horario_visita"], "Horario de visita", "h"),
                width="stretch"
            )
        with col2:
            st.markdown("#### Estadísticos descriptivos")
            st.dataframe(table_stats_continua(enc["horario_visita"]), width="stretch",
                         hide_index=True)
        st.markdown("#### Distribución por franjas horarias")
        col3, col4 = st.columns([2, 1])
        with col3:
            if "rango_horario" in enc.columns:
                st.plotly_chart(
                    hist_rangos(enc["horario_visita"], enc["rango_horario"], "Horario"),
                    width="stretch"
                )
        with col4:
            if "rango_horario" in enc.columns:
                df_h = enc["rango_horario"].value_counts().sort_index().reset_index()
                df_h.columns = ["Franja", "Frecuencia"]
                df_h["%"] = (df_h["Frecuencia"] / df_h["Frecuencia"].sum() * 100).round(1)
                st.dataframe(df_h, width="stretch", hide_index=True)
        top_franja, pct_franja = _top_categoria(enc["rango_horario"]) if "rango_horario" in enc.columns else ("", 0)
        _insight(f"La franja de mayor afluencia es **{top_franja}** ({pct_franja}%), información "
                 f"clave para definir el horario de atención de los kioskos.")

        # --- Análisis cruzado: horario × tipo de día (grupos exclusivos) ---
        if "dias_visita" in enc.columns and "horario_visita" in enc.columns:
            st.markdown("#### Horario de visita según tipo de día (grupos exclusivos)")
            _FINDE_u  = {"sábado", "sabado", "domingo"}
            _SEMANA_u = {"lunes", "martes", "miércoles", "miercoles", "jueves", "viernes"}
            def _tipo_dia_excl(fila):
                if not isinstance(fila, str): return None
                dias = {p.strip().lower() for p in fila.split(",")}
                tiene_s = bool(dias & _SEMANA_u)
                tiene_f = bool(dias & _FINDE_u)
                if tiene_s and tiene_f: return "Ambos días"
                if tiene_s: return "Solo Lun–Vie"
                if tiene_f: return "Solo Sáb–Dom"
                return None
            _enc_hor = enc[["dias_visita", "horario_visita"]].copy()
            _enc_hor["tipo_dia"] = _enc_hor["dias_visita"].apply(_tipo_dia_excl)
            _enc_hor = _enc_hor.dropna(subset=["tipo_dia", "horario_visita"])
            _ORDER_TIPO = ["Solo Lun–Vie", "Ambos días", "Solo Sáb–Dom"]
            _COLORS_TIPO = {"Solo Lun–Vie": "#2980b9", "Ambos días": "#8e44ad", "Solo Sáb–Dom": "#e67e22"}
            import plotly.graph_objects as _go
            _fig_box = _go.Figure()
            for _label in _ORDER_TIPO:
                _vals = _enc_hor.loc[_enc_hor["tipo_dia"] == _label, "horario_visita"]
                if len(_vals) == 0:
                    continue
                _fig_box.add_trace(_go.Box(
                    y=_vals,
                    name=f"{_label} (n={len(_vals)})",
                    marker_color=_COLORS_TIPO[_label],
                    boxpoints="all",
                    jitter=0.35,
                    pointpos=0,
                    marker=dict(size=4, opacity=0.45),
                    line_width=2,
                ))
            _hr_ticks = list(range(4, 18))
            _fig_box.update_layout(
                yaxis_title="Hora de visita",
                yaxis=dict(
                    tickmode="array",
                    tickvals=_hr_ticks,
                    ticktext=[f"{h:02d}:00" for h in _hr_ticks],
                    range=[3, 18],
                ),
                showlegend=False,
                height=380,
                margin=dict(l=40, r=20, t=30, b=40),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            _fig_box.update_xaxes(showgrid=False)
            _fig_box.update_yaxes(gridcolor="#eeeeee")
            st.plotly_chart(_fig_box, width="stretch")
            _insight(
                "Los visitantes de **lunes a viernes** concentran su visita entre las **04:00 y las 09:00 h**, "
                "mientras que los de **sábado-domingo** llegan principalmente entre las **09:00 y las 12:00 h**.",
                significativo=None
            )

    elif variable == "Género":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(pie_categorica(enc["genero"], "Género"), width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["genero"]), width="stretch", hide_index=True)
        top_g, pct_g = _top_categoria(enc["genero"])
        equilibrio = "una muestra equilibrada entre géneros" if pct_g < 60 else f"predominio del género {top_g.lower()}"
        _insight(f"**{pct_g}%** de los encuestados es **{top_g}**, mostrando {equilibrio}.")

    elif variable == "Sector de residencia":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(bar_categorica(enc["sector_residencia"], "Sector de residencia",
                                           horizontal=True), width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["sector_residencia"]), width="stretch",
                         hide_index=True)
        top_sec, pct_sec = _top_categoria(enc["sector_residencia"])
        _insight(f"El sector de residencia más representado es **{top_sec}** ({pct_sec}%), útil "
                 f"para dirigir campañas de difusión hacia esa zona.")

    elif variable == "Acompañante":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(pie_categorica(enc["acompanante"], "¿Con quién visita el parque?"),
                            width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["acompanante"]), width="stretch", hide_index=True)
        top_ac, pct_ac = _top_categoria(enc["acompanante"])
        _insight(f"La mayoría visita el parque **{top_ac.lower()}** ({pct_ac}%), lo que sugiere "
                 f"diseñar la oferta de los kioskos pensando en ese tipo de grupo.")

    elif variable == "Frecuencia de visita":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                bar_categorica(enc["frecuencia_visita"], "Frecuencia de visita",
                               order=FRECUENCIA_ORDER),
                width="stretch"
            )
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["frecuencia_visita"], order=FRECUENCIA_ORDER),
                         width="stretch", hide_index=True)
        top_fr, pct_fr = _top_categoria(enc["frecuencia_visita"])
        _insight(f"**{pct_fr}%** de los encuestados visita el parque **{top_fr.lower()}**, "
                 f"un indicador del nivel de fidelización de la demanda potencial.")

    elif variable == "Días que visita (multi-opción)":
        st.info("⚠️ Selección múltiple: los porcentajes pueden superar el 100% ya que cada "
                "encuestado pudo seleccionar varios días.")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(bar_multiselect(enc, "dias_visita", "Días de visita al parque"),
                            width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            tabla_dias = table_multiselect(enc, "dias_visita")
            st.dataframe(tabla_dias, width="stretch", hide_index=True)
        top_dia = tabla_dias.iloc[0]
        _insight(f"**{top_dia['Opción']}** es el día más mencionado "
                 f"({top_dia['% de encuestados que lo mencionaron']}% de los encuestados), "
                 f"relevante para programar el abastecimiento de los kioskos.")

    elif variable == "Motivo de visita":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(bar_categorica(enc["motivo_visita"], "Motivo principal de visita",
                                           horizontal=True), width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["motivo_visita"]), width="stretch",
                         hide_index=True)
        top_mot, pct_mot = _top_categoria(enc["motivo_visita"])
        _insight(f"El motivo principal de visita es **{top_mot}** ({pct_mot}%), lo que debe "
                 f"orientar la oferta comercial de los kioskos.")

    elif variable == "¿Consumiría productos?":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(pie_categorica(enc["consumiria"], "¿Consumiría productos en el parque?"),
                            width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["consumiria"]), width="stretch", hide_index=True)
        _insight(f"**{_pct_si(enc['consumiria'])}%** de los encuestados consumiría productos o "
                 f"servicios dentro del parque, lo que valida la viabilidad comercial de los kioskos.")

    elif variable == "Productos de interés":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(bar_categorica(enc["productos_interes"], "Productos o servicios de interés",
                                           horizontal=True), width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["productos_interes"]), width="stretch",
                         hide_index=True)
        top_prod, pct_prod = _top_categoria(enc["productos_interes"])
        _insight(f"El producto o servicio más demandado es **{top_prod}** ({pct_prod}%); debe "
                 f"priorizarse en la mezcla comercial (giro) de los kioskos.")

    elif variable == "Gasto dispuesto":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                bar_categorica(enc["gasto_dispuesto"], "Gasto dispuesto a pagar", order=GASTO_ORDER),
                width="stretch"
            )
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["gasto_dispuesto"], order=GASTO_ORDER),
                         width="stretch", hide_index=True)
        top_gas, pct_gas = _top_categoria(enc["gasto_dispuesto"])
        _insight(f"El rango de gasto más común es **{top_gas}** ({pct_gas}%), referencia directa "
                 f"para fijar el ticket promedio por kiosko.")

    elif variable == "¿Considera adecuados los kioskos?":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(
                pie_categorica(enc["kiosko_adecuado"], "¿Considera adecuados los kioskos?"),
                width="stretch"
            )
        with col2:
            st.dataframe(table_frecuencia(enc["kiosko_adecuado"]), width="stretch",
                         hide_index=True)
        _insight(f"**{_pct_si(enc['kiosko_adecuado'])}%** de los encuestados considera adecuada "
                 f"la implementación de kioskos, lo que respalda la aceptación social del proyecto.")

    elif variable == "Zona de acceso preferida":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                bar_categorica(enc["zona_acceso"], "Zona de acceso preferida para kioskos",
                               horizontal=True),
                width="stretch"
            )
        with col2:
            st.dataframe(table_frecuencia(enc["zona_acceso"]), width="stretch", hide_index=True)
        top_zona, pct_zona = _top_categoria(enc["zona_acceso"])
        _insight(f"La zona de acceso preferida es **{top_zona}** ({pct_zona}%); debe priorizarse "
                 f"en la distribución espacial de los kioskos.")

    elif variable == "¿Kioskos mejorarían la experiencia?":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(
                pie_categorica(enc["kiosko_mejora_experiencia"],
                               "¿Kioskos mejorarían la experiencia?"),
                width="stretch"
            )
        with col2:
            st.dataframe(table_frecuencia(enc["kiosko_mejora_experiencia"]),
                         width="stretch", hide_index=True)
        _insight(f"**{_pct_si(enc['kiosko_mejora_experiencia'])}%** de los encuestados considera "
                 f"que los kioskos mejorarían su experiencia en el parque.")

    elif variable == "Calificación actual de oferta":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                bar_calificacion(enc["calificacion_oferta"], "Oferta actual de servicios en el parque"),
                width="stretch"
            )
        with col2:
            st.markdown("#### Estadísticos")
            st.dataframe(table_stats_continua(enc["calificacion_oferta"]), width="stretch",
                         hide_index=True)
        calif_prom = enc["calificacion_oferta"].dropna().mean()
        nivel = "baja" if calif_prom < 3 else ("regular" if calif_prom < 4 else "buena")
        _insight(f"La calificación promedio de la oferta actual es **{calif_prom:.1f}/5** "
                 f"(percepción **{nivel}**), lo que evidencia "
                 f"{'una oportunidad clara para nuevos kioskos' if calif_prom < 4 else 'un punto de partida favorable para ampliar la oferta'}.")

    elif variable == "Servicios prioritarios (multi-opción)":
        st.info("⚠️ Selección múltiple: cada encuestado pudo seleccionar varias opciones.")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                bar_multiselect(enc, "servicios_prioritarios",
                                "Servicios complementarios prioritarios"),
                width="stretch"
            )
        with col2:
            st.markdown("#### Tabla de menciones")
            tabla_serv = table_multiselect(enc, "servicios_prioritarios")
            st.dataframe(tabla_serv, width="stretch", hide_index=True)
        top_serv = tabla_serv.iloc[0]
        _insight(f"El servicio complementario más solicitado es **{top_serv['Opción']}** "
                 f"({top_serv['% de encuestados que lo mencionaron']}% de menciones).")


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: ANÁLISIS BIVARIADO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔗 Análisis Bivariado":
    _header("Análisis Bivariado", "Relaciones entre pares de variables")

    analisis = st.selectbox("Seleccione el análisis:", [
        "Consumo por género",
        "Gasto dispuesto por género",
        "Motivo de visita por frecuencia",
        "Aprobación de kioskos por edad (boxplot)",
        "Aprobación de kioskos por sector",
        "Gasto dispuesto por acompañante",
        "Calificación de oferta por frecuencia",
        "Horario de visita por género (boxplot)",
        "Consumo vs Gasto – tabla cruzada",
    ])

    st.markdown("---")

    if analisis == "Consumo por género":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                crosstab_stacked(enc, "genero", "consumiria", "Género", "¿Consumiría?"),
                width="stretch"
            )
        with col2:
            st.markdown("#### Tabla cruzada (frecuencias)")
            ct = pd.crosstab(enc["genero"], enc["consumiria"], margins=True)
            st.dataframe(ct, width="stretch")
            chi = chi2_test(enc, "genero", "consumiria")
            st.markdown(f"""
            **{chi['test']}:**
            - χ² = {chi['chi2']} | p = {chi['p_valor']} (α = 0.05)
            - Grados de libertad: {chi['grados_libertad']}
            """)
        _insight(chi["interpretacion"], chi["significativo"])

    elif analisis == "Gasto dispuesto por género":
        st.plotly_chart(consumo_vs_gasto(enc), width="stretch")
        st.markdown("#### Estadísticos descriptivos por género")
        gasto_map = {"Menos de $2": 1.5, "Entre $2 y $5": 3.5, "Entre $5 y $10": 7.5,
                     "Entre $10 y $20": 15.0, "Más de $20": 22.0}
        enc_g = enc.copy()
        enc_g["gasto_num"] = enc_g["gasto_dispuesto"].map(gasto_map)
        st.dataframe(stats_tabla_bivariada(enc_g, "gasto_num", "genero"),
                     width="stretch", hide_index=True)
        anova = anova_test(enc_g, "gasto_num", "genero")
        st.markdown(f"**{anova['test']}:** F = {anova['f']} | p = {anova['p_valor']} (α = 0.05)")
        _insight(anova["interpretacion"], anova["significativo"])

    elif analisis == "Motivo de visita por frecuencia":
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(
                crosstab_heatmap(enc, "frecuencia_visita", "motivo_visita",
                                 "Frecuencia de visita", "Motivo"),
                width="stretch"
            )
        with col2:
            chi = chi2_test(enc, "frecuencia_visita", "motivo_visita")
            st.markdown(f"""
            **{chi['test']}:**
            - χ² = {chi['chi2']}
            - p = {chi['p_valor']} (α = 0.05)
            """)
        _insight(chi["interpretacion"], chi["significativo"])

    elif analisis == "Aprobación de kioskos por edad (boxplot)":
        st.plotly_chart(
            boxplot_continua_vs_cat(enc, "edad", "kiosko_adecuado",
                                    "Edad (años)", "¿Kioskos adecuados?"),
            width="stretch"
        )
        st.dataframe(stats_tabla_bivariada(enc, "edad", "kiosko_adecuado"),
                     width="stretch", hide_index=True)
        anova = anova_test(enc, "edad", "kiosko_adecuado")
        st.markdown(f"**{anova['test']}:** F = {anova['f']} | p = {anova['p_valor']} (α = 0.05)")
        _insight(anova["interpretacion"], anova["significativo"])

    elif analisis == "Aprobación de kioskos por sector":
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(
                crosstab_stacked(enc, "sector_residencia", "kiosko_adecuado",
                                 "Sector de residencia", "¿Kioskos adecuados?"),
                width="stretch"
            )
        with col2:
            chi = chi2_test(enc, "sector_residencia", "kiosko_adecuado")
            st.markdown(f"""
            **{chi['test']}:**
            - χ² = {chi['chi2']}
            - p = {chi['p_valor']} (α = 0.05)
            """)
        _insight(chi["interpretacion"], chi["significativo"])

    elif analisis == "Gasto dispuesto por acompañante":
        st.plotly_chart(
            crosstab_grouped(enc, "acompanante", "gasto_dispuesto",
                             "Con quién visita", "Gasto dispuesto"),
            width="stretch"
        )
        chi = chi2_test(enc, "acompanante", "gasto_dispuesto")
        st.markdown(f"**{chi['test']}:** χ² = {chi['chi2']} | p = {chi['p_valor']} (α = 0.05)")
        _insight(chi["interpretacion"], chi["significativo"])

    elif analisis == "Calificación de oferta por frecuencia":
        st.plotly_chart(
            boxplot_continua_vs_cat(enc, "calificacion_oferta", "frecuencia_visita",
                                    "Calificación (1-5)", "Frecuencia de visita"),
            width="stretch"
        )
        st.dataframe(stats_tabla_bivariada(enc, "calificacion_oferta", "frecuencia_visita"),
                     width="stretch", hide_index=True)
        anova = anova_test(enc, "calificacion_oferta", "frecuencia_visita")
        st.markdown(f"**{anova['test']}:** F = {anova['f']} | p = {anova['p_valor']} (α = 0.05)")
        _insight(anova["interpretacion"], anova["significativo"])

    elif analisis == "Horario de visita por género (boxplot)":
        st.plotly_chart(
            boxplot_continua_vs_cat(enc, "horario_visita", "genero",
                                    "Horario de visita (h)", "Género"),
            width="stretch"
        )
        st.dataframe(stats_tabla_bivariada(enc, "horario_visita", "genero"),
                     width="stretch", hide_index=True)
        anova = anova_test(enc, "horario_visita", "genero")
        st.markdown(f"**{anova['test']}:** F = {anova['f']} | p = {anova['p_valor']} (α = 0.05)")
        _insight(anova["interpretacion"], anova["significativo"])

    elif analisis == "Consumo vs Gasto – tabla cruzada":
        ct = pd.crosstab(enc["consumiria"], enc["gasto_dispuesto"], margins=True)
        st.markdown("#### Tabla cruzada: ¿Consumiría? × Gasto dispuesto")
        st.dataframe(ct, width="stretch")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(
                crosstab_heatmap(enc.dropna(subset=["consumiria", "gasto_dispuesto"]),
                                 "consumiria", "gasto_dispuesto",
                                 "¿Consumiría?", "Gasto dispuesto"),
                width="stretch"
            )
        with col2:
            chi = chi2_test(enc, "consumiria", "gasto_dispuesto")
            st.markdown(f"""
            **{chi['test']}:**
            - χ² = {chi['chi2']}
            - p = {chi['p_valor']} (α = 0.05)
            """)
        _insight(chi["interpretacion"], chi["significativo"])


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: ANÁLISIS MULTIVARIADO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🌐 Análisis Multivariado":
    _header("Análisis Multivariado", "PCA, segmentación y correlaciones entre múltiples variables")

    COLS_MULTI = ["edad", "horario_visita", "calificacion_oferta"]
    COLS_LABELS = {
        "edad": "Edad",
        "horario_visita": "Horario",
        "calificacion_oferta": "Calificación",
        "genero": "Género",
        "frecuencia_visita": "Frecuencia",
        "acompanante": "Acompañante",
        "motivo_visita": "Motivo",
        "consumiria": "¿Consumiría?",
        "gasto_dispuesto": "Gasto",
        "kiosko_adecuado": "Aprob. kiosko",
    }

    COLS_ENCODE = [c for c in [
        "edad", "horario_visita", "calificacion_oferta",
        "genero", "frecuencia_visita", "acompanante",
        "motivo_visita", "consumiria", "gasto_dispuesto",
    ] if c in enc.columns]

    tab1, tab2, tab3, tab4 = st.tabs(["Correlaciones", "PCA – Biplot", "Segmentación K-Means",
                                       "K óptimo (Codo)"])

    with tab1:
        st.markdown("#### Matriz de correlación entre variables (codificadas)")
        st.plotly_chart(correlation_matrix(enc, COLS_ENCODE, COLS_LABELS),
                        width="stretch")
        st.caption("Las variables categóricas fueron codificadas ordinalmente. "
                   "Coeficiente de Pearson sobre valores codificados.")
        # Interpretación automática basada en los pares con mayor correlación
        _enc_corr = enc[COLS_ENCODE].dropna()
        _top = top_correlations(_enc_corr, COLS_ENCODE, COLS_LABELS, n=3)
        if _top:
            _parts = [f"<b>{la}</b> y <b>{lb}</b> (r = {r:+.2f})" for r, la, lb in _top]
            _dir = "positiva" if _top[0][0] > 0 else "negativa"
            _insight(
                f"La correlación más marcada se da entre {_parts[0]}, una relación {_dir}. "
                f"También destacan {_parts[1]} y {_parts[2]}. "
                f"Valores de r cercanos a ±1 indican fuerte relación lineal; "
                f"cercanos a 0 señalan variables prácticamente independientes. "
                f"Las variables categóricas se codificaron ordinalmente, por lo que las "
                f"correlaciones reflejan tendencias de orden, no magnitudes exactas."
            )

    with tab2:
        color_opt = st.selectbox("Colorear por:", ["genero", "consumiria", "frecuencia_visita",
                                                    "motivo_visita"], key="pca_color")
        extra_cols = [color_opt] if color_opt not in COLS_ENCODE else []
        enc_clean = enc[COLS_ENCODE + extra_cols].dropna()
        fig_pca, var_exp = pca_biplot(enc_clean, COLS_ENCODE, COLS_LABELS, color_col=color_opt)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig_pca, width="stretch")
        with col2:
            st.markdown("#### Varianza explicada")
            st.dataframe(var_exp, width="stretch", hide_index=True)
            st.plotly_chart(scree_plot(enc_clean, COLS_ENCODE), width="stretch")
        # Interpretación de PCs basada en las cargas más altas
        _loads = pca_loadings_df(enc_clean, COLS_ENCODE, COLS_LABELS, n_pcs=2)
        _pc1_top = _loads["PC1"].abs().nlargest(2).index.tolist()
        _pc2_top = _loads["PC2"].abs().nlargest(2).index.tolist()
        _pc1_pct = float(var_exp.loc[var_exp["Componente"] == "PC1",
                                     "Varianza explicada (%)"].iloc[0])
        _pc2_pct = float(var_exp.loc[var_exp["Componente"] == "PC2",
                                     "Varianza explicada (%)"].iloc[0])
        _cum_pct = float(var_exp.loc[var_exp["Componente"] == "PC2",
                                     "Varianza acumulada (%)"].iloc[0])
        _insight(
            f"<b>PC1</b> ({_pc1_pct}% de varianza) captura principalmente la variabilidad de "
            f"<b>{' y '.join(_pc1_top)}</b> — el eje que más diferencia a los encuestados. "
            f"<b>PC2</b> ({_pc2_pct}%) refleja principalmente <b>{' y '.join(_pc2_top)}</b>. "
            f"Juntos explican el <b>{_cum_pct}%</b> de la varianza total. "
            f"Los vectores rojos indican la dirección en que crece cada variable; "
            f"puntos agrupados y vectores alineados señalan perfiles similares de encuestados."
        )

    with tab3:
        n_k = st.slider("Número de segmentos (k):", min_value=2, max_value=6, value=3)
        enc_km = enc[COLS_ENCODE].dropna()
        fig_km, _km_labels, perfil = kmeans_clusters(enc_km, COLS_ENCODE, n_clusters=n_k,
                                                      color_labels=list(COLS_LABELS.values()))
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig_km, width="stretch")
        with col2:
            st.markdown("#### Perfil de cada segmento")
            st.dataframe(perfil.T, width="stretch")
        st.markdown("#### Tamaño de cada segmento")
        seg_counts = _km_labels.value_counts().sort_index().reset_index()
        seg_counts.columns = ["Segmento", "N"]
        seg_counts["Segmento"] = [f"Segmento {k+1}" for k in seg_counts["Segmento"]]
        seg_counts["%"] = (seg_counts["N"] / seg_counts["N"].sum() * 100).round(1)
        st.dataframe(seg_counts, width="stretch", hide_index=True)
        # Interpretación automática de cada segmento (solo columnas numéricas del perfil)
        _overall_m = enc_km.mean(numeric_only=True)
        _overall_s = enc_km.std(numeric_only=True).replace(0, 1)
        _k_counts = _km_labels.value_counts().sort_index()
        _bullets = []
        for _ki, _seg_name in enumerate(perfil.index):
            _row = perfil.loc[_seg_name]
            _n_seg = int(_k_counts.get(_ki, 0))
            _devs = []
            for _col in COLS_ENCODE:
                if _col not in perfil.columns:
                    continue
                _z = (_row[_col] - _overall_m[_col]) / _overall_s[_col]
                _lbl = COLS_LABELS.get(_col, _col)
                if _col == "edad":
                    _feat = f"edad promedio {_row[_col]:.0f} años"
                elif _col == "horario_visita":
                    _feat = f"horario de visita {_row[_col]:.0f}h"
                elif _col == "calificacion_oferta":
                    _feat = f"calificación de oferta {_row[_col]:.1f}/5"
                else:
                    _feat = f"{_lbl} {'↑' if _z > 0 else '↓'} respecto al promedio"
                _devs.append((abs(_z), _feat))
            _devs.sort(reverse=True)
            _top_feats = "; ".join(_f for _, _f in _devs[:3])
            _bullets.append(f"<b>{_seg_name}</b> (n={_n_seg}): {_top_feats}.")
        _insight(
            "Cada segmento agrupa encuestados con perfil similar. "
            "Características distintivas de cada uno:<br>"
            + "<br>".join(f"• {b}" for b in _bullets)
        )

    with tab4:
        enc_elbow = enc[COLS_ENCODE].dropna()
        st.plotly_chart(elbow_silhouette(enc_elbow, COLS_ENCODE, max_k=8),
                        width="stretch")
        _insight(
            "Para elegir el número óptimo de segmentos, observe dos criterios simultáneamente: "
            "(1) <b>Método del codo (inercia):</b> el k en el que la curva deja de bajar "
            "bruscamente — agregar más clusters ya no reduce significativamente la dispersión interna. "
            "(2) <b>Silhouette:</b> el k con el valor más alto indica segmentos más compactos y "
            "bien separados entre sí. Cuando ambos criterios coinciden, ese es el número recomendado. "
            "Para este perfil de visitantes, k = 3 suele equilibrar interpretabilidad y separación."
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: RECOMENDACIONES
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "💬 Recomendaciones":
    _header("Análisis de Recomendaciones", "Procesamiento de comentarios y sugerencias de los encuestados")

    if "comentarios" not in enc.columns or enc["comentarios"].notna().sum() < 3:
        st.warning("No hay suficientes comentarios válidos para el análisis.")
    else:
        result_df = analyze_comments(enc)
        n_com = len(result_df)

        col_k1, col_k2, col_k3 = st.columns(3)
        with col_k1:
            pos = (result_df["sentiment"] == "Positivo").sum()
            st.metric("Comentarios positivos", f"{pos} ({pos/n_com*100:.0f}%)")
        with col_k2:
            neg = (result_df["sentiment"] == "Negativo").sum()
            st.metric("Comentarios negativos", f"{neg} ({neg/n_com*100:.0f}%)")
        with col_k3:
            st.metric("Total comentarios analizados", n_com)

        st.markdown("---")
        tab_s1, tab_s2, tab_s3, tab_s4 = st.tabs(
            ["Sentimiento global", "Temas detectados", "Palabras clave", "Comentarios"]
        )

        with tab_s1:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(sentiment_pie(result_df), width="stretch")
            with col2:
                st.plotly_chart(sentiment_by_topic_heatmap(result_df), width="stretch")

        with tab_s2:
            st.plotly_chart(topics_bar(result_df), width="stretch")
            st.markdown("""
            <div class="insight-box">
            <b>Metodología:</b> Los temas se detectan mediante coincidencia de palabras clave
            en los comentarios. Un mismo comentario puede pertenecer a múltiples temas.
            </div>
            """, unsafe_allow_html=True)

        with tab_s3:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("#### Palabras más frecuentes")
                kw_df = top_keywords(enc, top_n=25)
                st.dataframe(kw_df, width="stretch", hide_index=True)
            with col2:
                st.markdown("#### Nube de palabras")
                img_b64 = wordcloud_img(enc)
                st.markdown(
                    f'<img src="data:image/png;base64,{img_b64}" style="width:100%;border-radius:8px;">',
                    unsafe_allow_html=True,
                )

        with tab_s4:
            filtro_sent = st.selectbox("Filtrar por sentimiento:",
                                       ["Todos", "Positivo", "Negativo", "Mixto", "Neutro"])
            df_show = result_df if filtro_sent == "Todos" else result_df[
                result_df["sentiment"] == filtro_sent
            ]
            for _, row in df_show.iterrows():
                badge_color = {
                    "Positivo": "#27ae60", "Negativo": "#e74c3c",
                    "Mixto": "#f39c12", "Neutro": "#95a5a6",
                }.get(row["sentiment"], "#bdc3c7")
                temas_str = ", ".join(row["temas"])
                st.markdown(f"""
                <div style="background:white;border-radius:8px;padding:0.8rem;margin-bottom:0.5rem;
                            box-shadow:0 1px 4px rgba(0,0,0,0.08);">
                    <span style="background:{badge_color};color:white;padding:2px 8px;
                                 border-radius:4px;font-size:0.8rem;">{row['sentiment']}</span>
                    <span style="color:#666;font-size:0.8rem;margin-left:0.5rem;">📌 {temas_str}</span>
                    <p style="margin:0.5rem 0 0;">{row['comentario']}</p>
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: PRUEBA DE HIPÓTESIS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📐 Prueba de Hipótesis":
    _header("Prueba de Hipótesis",
            "Validación estadística de las decisiones clave del modelo comercial")

    forecast = forecast_kioskos(enc, stats)

    st.markdown("""
    <div class="insight-box">
    Se aplican dos <b>pruebas Z de hipótesis</b> (α = 0.05) sobre los datos reales de la encuesta.
    Cada prueba sigue el marco clásico: H₀ (hipótesis nula) vs. H₁ (hipótesis alternativa),
    con valor Z, p-valor y decisión de aceptar o rechazar H₀.
    </div>
    """, unsafe_allow_html=True)

    # ── HIPÓTESIS 1 ──────────────────────────────────────────────────────────
    res_h1 = test_h1_cantidad_kioskos(enc, forecast)

    st.markdown("---")
    st.markdown('<p class="section-header">Hipótesis 1 – Cantidad de kioskos (Plan SHOT vs. modelo estadístico)</p>',
                unsafe_allow_html=True)

    # Tarjetas H0 / H1
    col_h0, col_h1 = st.columns(2)
    with col_h0:
        st.markdown(f"""
        <div style="background:#fef9e7;border-radius:10px;padding:1rem;
                    border-left:5px solid #f39c12;">
            <div style="font-weight:700;color:#d68910;font-size:0.9rem;margin-bottom:0.4rem;">
                H₀ (Hipótesis nula)
            </div>
            <div style="font-size:0.88rem;color:#333;">{res_h1['H0']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h1:
        st.markdown(f"""
        <div style="background:#eaf4fb;border-radius:10px;padding:1rem;
                    border-left:5px solid #2980b9;">
            <div style="font-weight:700;color:#1a5276;font-size:0.9rem;margin-bottom:0.4rem;">
                H₁ (Hipótesis alternativa)
            </div>
            <div style="font-size:0.88rem;color:#333;">{res_h1['H1_alt']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Métricas del test
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Prueba estadística", "Z-test (cola izq.)")
    with c2:
        st.metric("Z estadístico", f"{res_h1['z_stat']:.3f}")
    with c3:
        st.metric("Z crítico (α=0.05)", f"{res_h1['z_crit']:.3f}")
    with c4:
        st.metric("p-valor", "< 0.0001")

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        st.plotly_chart(fig_h1_distribucion(res_h1), width="stretch")
    with col_fig2:
        st.plotly_chart(fig_h1_intervalo_confianza(res_h1), width="stretch")

    # Decisión
    dec_color = "#d5f5e3" if res_h1["rechaza_H0"] else "#fde8e8"
    dec_border = "#27ae60" if res_h1["rechaza_H0"] else "#e74c3c"
    st.markdown(f"""
    <div style="background:{dec_color};border-radius:10px;padding:1rem 1.3rem;
                border-left:6px solid {dec_border};margin-top:0.5rem;">
        <div style="font-size:1.05rem;font-weight:700;color:{dec_border};">
            {res_h1['decision']}
        </div>
        <div style="font-size:0.88rem;color:#333;margin-top:0.5rem;">
            {res_h1['conclusion']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── HIPÓTESIS 2 ──────────────────────────────────────────────────────────
    res_h2 = test_h2_aprobacion_kioskos(enc)

    st.markdown("---")
    st.markdown('<p class="section-header">Hipótesis 2 – Aprobación ciudadana de los kioskos</p>',
                unsafe_allow_html=True)

    col_h0b, col_h1b = st.columns(2)
    with col_h0b:
        st.markdown(f"""
        <div style="background:#fef9e7;border-radius:10px;padding:1rem;
                    border-left:5px solid #f39c12;">
            <div style="font-weight:700;color:#d68910;font-size:0.9rem;margin-bottom:0.4rem;">
                H₀ (Hipótesis nula)
            </div>
            <div style="font-size:0.88rem;color:#333;">{res_h2['H0']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_h1b:
        st.markdown(f"""
        <div style="background:#eaf4fb;border-radius:10px;padding:1rem;
                    border-left:5px solid #2980b9;">
            <div style="font-weight:700;color:#1a5276;font-size:0.9rem;margin-bottom:0.4rem;">
                H₁ (Hipótesis alternativa)
            </div>
            <div style="font-size:0.88rem;color:#333;">{res_h2['H1_alt']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1b, c2b, c3b, c4b = st.columns(4)
    with c1b:
        st.metric("Prueba estadística", "Z-test (cola der.)")
    with c2b:
        st.metric("Z estadístico", f"{res_h2['z_stat']:.3f}")
    with c3b:
        st.metric("Z crítico (α=0.05)", f"{res_h2['z_crit']:.3f}")
    with c4b:
        p_fmt = f"{res_h2['p_valor']:.2e}" if res_h2['p_valor'] < 0.001 else f"{res_h2['p_valor']:.4f}"
        st.metric("p-valor", p_fmt)

    col_fig3, col_fig4 = st.columns(2)
    with col_fig3:
        st.plotly_chart(fig_h2_distribucion(res_h2), width="stretch")
    with col_fig4:
        st.plotly_chart(fig_h2_proporciones(res_h2), width="stretch")

    dec_color2 = "#d5f5e3" if res_h2["rechaza_H0"] else "#fde8e8"
    dec_border2 = "#27ae60" if res_h2["rechaza_H0"] else "#e74c3c"
    st.markdown(f"""
    <div style="background:{dec_color2};border-radius:10px;padding:1rem 1.3rem;
                border-left:6px solid {dec_border2};margin-top:0.5rem;">
        <div style="font-size:1.05rem;font-weight:700;color:{dec_border2};">
            {res_h2['decision']}
        </div>
        <div style="font-size:0.88rem;color:#333;margin-top:0.5rem;">
            {res_h2['conclusion']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    <b>Resumen ejecutivo:</b> ambas pruebas son concluyentes (p&lt;0.05).
    La demanda estadística <b>no respalda</b> el plan de 50 kioskos de la SHOT —
    el número técnicamente justificado al año 2036 es <b>25 kioskos</b>
    (10 Bulevar + 8 Arena + 7 Canchas), respetando restricciones físicas y comerciales.
    Al mismo tiempo, la ciudadanía <b>sí apoya mayoritariamente</b> la implementación
    de espacios comerciales en el parque, lo que valida el proyecto en su conjunto.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: MODELO COMERCIAL
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🏪 Modelo comercial":
    _header("Modelo Comercial – Kioskos Parque Bicentenario",
            "Modelo integrado: demanda de encuesta + análisis espacial por zona")

    forecast = forecast_kioskos(enc, stats)
    tasas = forecast["tasas"]

    # KPIs del modelo
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tasa de consumo", f"{tasas['tasa_consumo']*100:.1f}%",
                  help="% de encuestados que consumirían en el parque")
    with col2:
        st.metric("Gasto promedio dispuesto", f"${tasas['gasto_promedio_usd']}",
                  help="Gasto promedio por visita")
    with col3:
        st.metric("Kioskos totales (demanda)", forecast["total_kioskos_recomendados"],
                  help=f"Rango: {forecast['total_kioskos_min']}–{forecast['total_kioskos_max']}")
    with col4:
        st.metric("Aprobación de kioskos", f"{tasas['tasa_aprobacion_kiosko']*100:.1f}%",
                  help="% de encuestados que aprueban la implementación")

    st.markdown("---")

    tab_esp, tab_dem, tab_com, tab_proy, tab_lay = st.tabs([
        "📐 Análisis Espacial y Tipo de Kiosko",
        "📊 Modelo de Demanda (2025)",
        "🔄 Distribución Comercial y Giros",
        "📈 Proyección 2026–2036",
        "🗺️ Esquemas de Planta por Zona",
    ])

    # ── TAB 1: ANÁLISIS ESPACIAL ───────────────────────────────────────────────
    with tab_esp:
        st.markdown('<p class="section-header">Dimensiones físicas de las zonas</p>',
                    unsafe_allow_html=True)

        # Tarjetas de zona
        cols_z = st.columns(3)
        zone_keys = list(ZONE_DIMENSIONS.keys())
        for ci, zona_name in enumerate(zone_keys):
            dim = ZONE_DIMENSIONS[zona_name]
            area = dim["longitud_m"] * dim["ancho_m"]
            short = zona_name.split("–")[1].strip() if "–" in zona_name else zona_name
            with cols_z[ci]:
                st.markdown(f"""
                <div style="background:white;border-radius:10px;padding:1rem;
                            box-shadow:0 2px 8px rgba(0,0,0,0.08);
                            border-top:4px solid {dim['color']};">
                    <div style="font-weight:700;color:#1a3a5c;font-size:0.95rem;">{short}</div>
                    <div style="font-size:0.8rem;color:#666;margin:0.3rem 0;">{dim['tipo_zona']}</div>
                    <div style="font-size:1.4rem;font-weight:700;color:{dim['color']};">
                        {dim['longitud_m']:,} × {dim['ancho_m']} m
                    </div>
                    <div style="font-size:0.85rem;color:#444;">
                        Área total: <b>{area:,} m²</b><br>
                        Lados disponibles: <b>{dim['lados_disponibles']}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Viabilidad por tipo de kiosko</p>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>Estándar aplicado (INEN 2 314 – Espacios públicos Ecuador):</b>
        corredor peatonal mínimo 3 m · área de servicio frontal 1.5 m ·
        separación entre kioskos 5 m · margen lateral 0.5 m.
        <br><b>Zona 1 – Bulevar:</b> ancho peatonal 12 m (la calzada vehicular ocupa los restantes 8 m del total de 20 m).
        </div>
        """, unsafe_allow_html=True)

        # Tabla de viabilidad zona × tipo
        for zona_name in zone_keys:
            dim = ZONE_DIMENSIONS[zona_name]
            short = zona_name.split("–")[1].strip() if "–" in zona_name else zona_name
            st.markdown(f"#### 📍 {short}  ({dim['longitud_m']:,} m × {dim['ancho_m']} m)")
            fase1_año = min(PLAN_FASES.keys())
            n_fase1 = PLAN_FASES[fase1_año]["kioskos"].get(zona_name, 2)
            rows_fit = []
            for ksk_key, ksk in KIOSKO_TIPOS.items():
                fit  = _ancho_util_por_kiosko(zona_name, ksk_key)
                cap  = _max_kioskos_por_longitud(zona_name, ksk_key)
                rec  = recommend_kiosko_type(zona_name) == ksk_key
                rows_fit.append({
                    "Tipo de kiosko": ksk_key,
                    "Dimensión": f"{ksk['ancho_m']}×{ksk['fondo_m']} m",
                    "Área": f"{ksk['area_m2']} m²",
                    "Categoría": ksk["categoria"],
                    "¿Cabe en la zona?": fit["observacion"],
                    "Máx. físico práctico": cap["max_practico_total"],
                    "Modelo estadístico": forecast["zonas"][zona_name]["kioskos_recomendados"],
                    f"Arranque {fase1_año}": n_fase1,
                    "⭐ Recomendado": "✅ Sí" if rec else "",
                    "Usos sugeridos": ksk["usos_recomendados"],
                })
            st.dataframe(pd.DataFrame(rows_fit), width="stretch", hide_index=True)
            st.markdown("")

        # Resumen final integrado
        st.markdown('<p class="section-header">Resumen integrado – cantidad y tipo por zona</p>',
                    unsafe_allow_html=True)
        tabla_esp = tabla_espacial_resumen(forecast["zonas"])
        # Renombrar columna para que sea claro que es el plan de arranque
        fase1_año = min(PLAN_FASES.keys())
        tabla_esp_display = tabla_esp.rename(columns={"Por demanda": f"Modelo estadístico", "RECOMENDADO FINAL": f"✅ Arranque {fase1_año}"})
        st.dataframe(tabla_esp_display[[
            "Zona", "Longitud", "Ancho", "Lados",
            "Tipo recomendado", "Máx. físico práctico", "Modelo estadístico",
            f"✅ Arranque {fase1_año}", "Viabilidad espacial",
        ]], width="stretch", hide_index=True)

        # Resumen de tipos por zona
        st.markdown('<p class="section-header">Características de los tipos de kiosko</p>',
                    unsafe_allow_html=True)
        cols_k = st.columns(3)
        for ci, (ksk_key, ksk) in enumerate(KIOSKO_TIPOS.items()):
            with cols_k[ci]:
                st.markdown(f"""
                <div style="background:white;border-radius:10px;padding:1rem;
                            box-shadow:0 2px 8px rgba(0,0,0,0.08);
                            border-top:4px solid {ksk['color']};">
                    <div style="font-weight:700;color:#1a3a5c;">{ksk_key}</div>
                    <div style="font-size:1.6rem;font-weight:700;color:{ksk['color']};
                                margin:0.3rem 0;">{ksk['area_m2']} m²</div>
                    <div style="font-size:0.82rem;color:#444;margin-bottom:0.5rem;">
                        {ksk['descripcion']}
                    </div>
                    <div style="font-size:0.8rem;background:#f8f9fa;padding:0.4rem;
                                border-radius:5px;color:#555;">
                        <b>Usos:</b> {ksk['usos_recomendados']}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 3: DISTRIBUCIÓN COMERCIAL Y GIROS ─────────────────────────────────
    with tab_com:
        st.markdown('<p class="section-header">Distancia comercial entre kioskos</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        <b>Criterio técnico:</b> cada kiosko necesita un "territorio" propio para ser viable.
        Si dos kioskos del <b>mismo giro</b> están demasiado cerca, se canibalizan.
        Si son de <b>giros complementarios</b>, pueden estar más próximos.
        La rotación alterna los rubros para maximizar la oferta sin competencia interna.
        </div>
        """, unsafe_allow_html=True)

        # Tabla resumen de todas las zonas — usar valores Arranque 2026 (no modelo estadístico)
        _fase1_com = min(PLAN_FASES.keys())
        n_finals = {}
        for zona_name in ZONE_DIMENSIONS:
            n_finals[zona_name] = PLAN_FASES[_fase1_com]["kioskos"].get(zona_name, 2)

        tabla_com = tabla_distribucion_comercial(forecast["zonas"])
        st.dataframe(tabla_com, width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown('<p class="section-header">Diagramas de distribución por zona</p>',
                    unsafe_allow_html=True)

        for zona_name in ZONE_DIMENSIONS:
            n_k = n_finals[zona_name]
            dim = ZONE_DIMENSIONS[zona_name]
            analysis = commercial_distance_analysis(zona_name, n_k)
            short = zona_name.split("–")[1].strip() if "–" in zona_name else zona_name

            st.markdown(f"#### 📍 {short}  ({dim['longitud_m']:,} m × {dim['ancho_m']} m)")

            # Métricas de la zona
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric("Kioskos", n_k)
            with col_m2:
                d_real = min(d["distancia_m"] for d in analysis["distancias"]) if analysis["distancias"] else dim["longitud_m"]
                st.metric("Distancia mínima real", f"{d_real:,} m")
            with col_m3:
                tipo = ZONAS[zona_name]["tipo"]
                st.metric("Dist. mín. dif. giro", f"{COMMERCIAL_PARAMS['distancia_min_dif_giro_m'][tipo]} m")
            with col_m4:
                estado = "✅ Cumple" if analysis["todos_cumplen"] else "⚠️ Revisar"
                st.metric("Estado", estado)

            # Diagrama lineal de distribución
            st.plotly_chart(fig_distribucion_comercial(zona_name, n_k), width="stretch")

            # Detalle de pares de kioskos
            if analysis["distancias"]:
                rows_dist = []
                for d_info in analysis["distancias"]:
                    icono_a = GIROS[d_info["giro_a"]]["icono"]
                    icono_b = GIROS[d_info["giro_b"]]["icono"]
                    rows_dist.append({
                        "Par": d_info["par"],
                        "Giro K-anterior": f"{icono_a} {d_info['giro_a']}",
                        "Giro K-siguiente": f"{icono_b} {d_info['giro_b']}",
                        "Relación": "Mismo giro ⚠️" if d_info["mismo_giro"] else "Giros distintos ✅",
                        "Distancia real": f"{d_info['distancia_m']:,} m",
                        "Distancia mínima": f"{d_info['distancia_min_m']} m",
                        "Estado": "✅" if d_info["cumple"] else "❌",
                    })
                st.dataframe(pd.DataFrame(rows_dist), width="stretch", hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)

        # Ficha de cada giro
        st.markdown("---")
        st.markdown('<p class="section-header">Características de cada giro comercial</p>',
                    unsafe_allow_html=True)
        cols_g = st.columns(3)
        giro_items = list(GIROS.items())
        for gi, (giro_name, giro_info) in enumerate(giro_items):
            with cols_g[gi % 3]:
                st.markdown(f"""
                <div style="background:white;border-radius:10px;padding:1rem;
                            box-shadow:0 2px 8px rgba(0,0,0,0.08);
                            border-left:5px solid {giro_info['color']};
                            margin-bottom:0.8rem;">
                    <div style="font-size:1.4rem;">{giro_info['icono']}</div>
                    <div style="font-weight:700;color:#1a3a5c;margin:0.2rem 0;">{giro_name}</div>
                    <div style="font-size:0.82rem;color:#555;">{giro_info['descripcion']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Respaldo de la encuesta a los giros elegidos
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">¿Por qué estos giros? Respaldo con datos de la encuesta</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_demanda_por_giro(enc), width="stretch")
        df_giro_dem = demanda_por_giro(enc)
        top_giro_row = df_giro_dem.iloc[0]
        _insight(
            f"Los giros de la rotación comercial no son arbitrarios: cada uno responde a una opción "
            f"real elegida por los encuestados en '¿Qué productos o servicios consumiría?'. "
            f"El giro con mayor respaldo es <b>{top_giro_row['giro']}</b> "
            f"({top_giro_row['pct_encuestados']}% de los encuestados), seguido por el resto en el orden "
            f"mostrado en el gráfico. Esto confirma que la oferta planificada está alineada con la "
            f"demanda manifestada, no con una selección arbitraria de productos."
        )

        # Tabla de rotación por zona
        st.markdown("---")
        st.markdown('<p class="section-header">Rotación de giros por tipo de zona</p>',
                    unsafe_allow_html=True)
        tipo_nombres = {
            "comercial_alta_densidad": "Bulevar comercial (familias, paseos)",
            "eventos_espectaculos": "Arena de espectáculos (eventos)",
            "deportivo": "Zona deportiva (canchas, pistas)",
        }
        for tipo_key, rotation in GIRO_ROTATION.items():
            secuencia = " → ".join(f"{GIROS[g]['icono']} {g}" for g in rotation)
            st.markdown(f"""
            <div style="background:#f8f9fa;border-radius:8px;padding:0.8rem 1rem;margin-bottom:0.6rem;">
                <div style="font-weight:600;color:#1a3a5c;font-size:0.9rem;">{tipo_nombres[tipo_key]}</div>
                <div style="font-size:0.85rem;color:#444;margin-top:0.3rem;">{secuencia} (se repite)</div>
            </div>
            """, unsafe_allow_html=True)

    # ── TAB 4: PROYECCIÓN 2025–2035 ───────────────────────────────────────────
    with tab_proy:
        st.markdown('<p class="section-header">Contexto urbano: el nuevo hipercentro de Quito</p>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="insight-box">
        <b>¿Por qué proyectar a 2036?</b> El sector del Parque Bicentenario está en transformación
        urbana acelerada con datos reales: la población actual es de <b>142,034 habitantes</b>
        y se proyecta alcanzar <b>304,570 habitantes</b> en 10 años. Los motores de crecimiento son:
        la edificabilidad que pasará de <b>6 a 40 pisos</b> (rascacielos aprobados), y
        <b>2 nuevas estaciones de metro</b> dentro del área de influencia del parque
        (<i>Bicentenario</i> en el eje del bulevar y <i>Andalucía</i> a 1–2 cuadras al norte).
        Con El Labrador ya existente, el parque quedará entre <b>3 paradas de metro</b>,
        convirtiéndose en el nuevo <b>hipercentro de Quito</b>.
        </div>
        """, unsafe_allow_html=True)

        # KPIs del escenario 2036
        pob_2026 = PROYECCION_PARAMS["poblacion_sector_hab"][2026]
        pob_2036 = PROYECCION_PARAMS["poblacion_sector_hab"][2036]
        total_2036 = sum(PLAN_FASES[2036]["kioskos"].values())
        total_2026 = sum(PLAN_FASES[2026]["kioskos"].values())
        trafico_2036 = sum(PROYECCION_PARAMS["trafico_boulevard_dia"][2036].values())

        col_k1, col_k2, col_k3, col_k4 = st.columns(4)
        with col_k1:
            st.metric("Población actual (2026)",
                      f"{pob_2026:,} hab.",
                      delta=f"+{pob_2036 - pob_2026:,} proyectados a 2036")
        with col_k2:
            st.metric("Nuevas paradas metro", "2",
                      delta="Bicentenario + Andalucía", delta_color="normal")
        with col_k3:
            st.metric("Edificabilidad", "Hasta 40 pisos",
                      delta="Antes: 6 pisos", delta_color="normal")
        with col_k4:
            st.metric("Kioskos en 2036", total_2036,
                      delta=f"+{total_2036 - total_2026} vs. arranque 2026")

        st.markdown("---")

        # Tráfico peatonal proyectado
        st.markdown('<p class="section-header">Proyección de tráfico peatonal – Bulevar Av. Amazonas</p>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        El bulevar actuará como <b>eje de movilidad cotidiana</b>: deportistas y familias
        (alta conversión a compra), residentes y trabajadores en tránsito al metro (baja conversión).
        La demanda de kioskos supera con creces la oferta planeada — lo que garantiza viabilidad comercial.
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(fig_trafico_parque_total(stats), width="stretch")

        # Demanda vs. plan vs. límite urbano
        st.markdown('<p class="section-header">Demanda por kioskos vs. plan de implementación</p>',
                    unsafe_allow_html=True)

        from models.kiosko_model import _consumidores_dia_boulevard, _kioskos_por_demanda_pura
        ultimo_año = max(PROYECCION_PARAMS["trafico_boulevard_dia"].keys())
        k_demanda_pura_final = _kioskos_por_demanda_pura(_consumidores_dia_boulevard(ultimo_año))
        limite_u = ZONE_DIMENSIONS["Zona 1 – Bulevar Av. Amazonas"]["longitud_m"] \
                   // PROYECCION_PARAMS["metros_entre_kioskos_bulevar"]

        st.markdown(f"""
        <div class="insight-box">
        En 2036, la <b>demanda pura</b> justificaría <b>~{k_demanda_pura_final} kioskos</b> solo en el bulevar.
        El plan cap en <b>10 kioskos</b> (1 cada 200 m) no es por falta de demanda —
        es el <b>estándar urbano de densidad comercial en espacio público</b>:
        más kioskos fragmentan el corredor peatonal y reducen la calidad del paseo.
        Con demanda que supera la oferta, cada kiosko opera con <b>alta ocupación y rentabilidad garantizada</b>.
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(fig_demanda_vs_plan_parque(stats, enc), width="stretch")

        st.markdown("---")

        # Plan de fases
        st.markdown('<p class="section-header">Plan de implementación por fases</p>',
                    unsafe_allow_html=True)
        st.plotly_chart(fig_plan_fases(), width="stretch")

        # Tabla de fases
        df_fases = tabla_plan_fases()
        st.dataframe(df_fases[[
            "Año", "Fase", "Hito", "Población (hab.)",
            "Bulevar", "Arena", "Canchas", "Total", "Contexto"
        ]], width="stretch", hide_index=True)

        st.markdown("---")
        st.markdown('<p class="section-header">Comparación: propuesta referencial (50 kioskos) vs. plan técnico 2036</p>',
                    unsafe_allow_html=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("""
            <div style="background:#ffeaa7;border-radius:10px;padding:1.2rem;
                        border-left:5px solid #fdcb6e;">
                <div style="font-weight:700;font-size:1rem;color:#2d3436;">
                    ⚠️ Propuesta referencial (50 kioskos)
                </div>
                <ul style="color:#636e72;font-size:0.85rem;margin-top:0.5rem;">
                    <li>Bulevar: 10 kioskos ✅ – correcto</li>
                    <li>Otras 2 zonas: 40 kioskos combinados ❌</li>
                    <li>Arena (150 m): imposible físicamente con 20+ kioskos</li>
                    <li>Canchas (200 m): imposible físicamente con 20+ kioskos</li>
                    <li>No considera distancias comerciales mínimas ni giros</li>
                    <li>No contempla fases de implementación</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with col_p2:
            total_final = sum(PLAN_FASES[2036]["kioskos"].values())
            total_ini = sum(PLAN_FASES[2026]["kioskos"].values())
            st.markdown(f"""
            <div style="background:#d5f5e3;border-radius:10px;padding:1.2rem;
                        border-left:5px solid #27ae60;">
                <div style="font-weight:700;font-size:1rem;color:#1a3a5c;">
                    ✅ Plan técnico recomendado (2026–2036)
                </div>
                <ul style="color:#2c3e50;font-size:0.85rem;margin-top:0.5rem;">
                    <li>Arranque 2026: <b>{total_ini} kioskos</b> (Bulevar 4 + Arena 3 + Canchas 3)</li>
                    <li>Bulevar 2036: <b>10 kioskos</b> – 1 cada 200 m ✅</li>
                    <li>Arena 2036: <b>8 kioskos</b> – físicamente viable en 150 m ✅</li>
                    <li>Canchas 2036: <b>7 kioskos</b> – físicamente viable en 200 m ✅</li>
                    <li>Total 2036: <b>{total_final} kioskos</b></li>
                    <li>4 fases: 2026 → 2029 → 2033 → 2036</li>
                    <li>Giros alternados · Distancias comerciales garantizadas</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <br>
        <div class="insight-box">
        <b>Conclusión:</b> Los <b>10 kioskos en el bulevar</b> son válidos por dos criterios
        independientes: densidad urbana (1/200m) y demanda proyectada (que en 2036 justificaría
        más del cuádruple). Las zonas 2 y 3 están acotadas por su <b>longitud física</b>
        (150m y 200m); colocar 40 kioskos en ambas es inviable — el máximo técnico razonable
        para el horizonte 2036 es <b>8 en la Arena y 7 en Canchas</b>, respetando distancias
        comerciales y tipos de giro alternados. El total técnico para 2036 es
        <b>{total_final} kioskos</b>, no 50.
        </div>
        """, unsafe_allow_html=True)

    # ── TAB 2: MODELO DE DEMANDA ───────────────────────────────────────────────
    with tab_dem:
        st.dataframe(tabla_resumen_forecast(forecast, stats, enc), width="stretch", hide_index=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(fig_kioskos_por_zona(forecast), width="stretch")
        with col_g2:
            st.plotly_chart(fig_visitantes_vs_demanda(stats, enc), width="stretch")

        st.plotly_chart(fig_ingresos_potenciales(stats, enc), width="stretch")

        with st.expander("⚙️ Parámetros del modelo de demanda"):
            cap = forecast["capacidad_kiosko"]
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.markdown(f"""**Capacidad por kiosko:**
- Clientes/hora: 10–15
- Horas semana: 7h | Finde: 10h
- Ocupación objetivo: 60%""")
            with col_p2:
                st.markdown(f"""**Demanda (encuesta):**
- Tasa de consumo: {tasas['tasa_consumo']*100:.1f}%
- Gasto promedio: ${tasas['gasto_promedio_usd']}
- Factor no contabilizados: ×2.5""")
            with col_p3:
                st.markdown(f"""**Capacidad anual/kiosko:**
- {cap['clientes_anuales_por_kiosko']:,} clientes/año
- {cap['clientes_diario_pico']} clientes en hora pico""")

        with st.expander("📐 Metodología detallada – cómo se calcularon los números"):
            cap_met = forecast["capacidad_kiosko"]
            st.markdown(f"""
**El modelo estadístico de demanda sigue 5 pasos:**

**Paso 1 – Usuarios base de actividades registradas (estadísticas 2025)**
Se suman los registros de participantes de todas las actividades asignadas a cada zona.
Solo se incluyen datos oficiales del parque (juegos recreativos, zona canina, bailoterapia,
competencias deportivas, uso de pistas, etc.). Este conteo tiene **subregistro reconocido**.

**Paso 2 – Visitantes estimados reales**
`Usuarios base × 2.5 (factor no contabilizados) × factor de captación de zona`
El multiplicador ×2.5 corrige el subregistro: por cada participante registrado, se estima
que 1.5 más visitan sin quedar en registros. El factor de captación varía por zona:
- Bulevar: **{ZONAS['Zona 1 – Bulevar Av. Amazonas']['factor_captacion']*100:.0f}%** (familias + zona canina + juegos inclusivos)
- Arena: **{ZONAS['Zona 2 – Arena de Espectáculos']['factor_captacion']*100:.0f}%** (público de eventos + bailoterapia)
- Canchas: **{ZONAS['Zona 3 – Canchas Deportivas y Pistas']['factor_captacion']*100:.0f}%** (deportistas + pista de ciclismo)

**Paso 3 – Consumidores potenciales**
`Visitantes × {tasas['tasa_consumo']*100:.1f}%` = tasa de consumo medida en encuesta.
De los 206 encuestados, el {tasas['tasa_consumo']*100:.1f}% indicó que consumiría en un kiosko del parque.

**Paso 4 – Ingresos potenciales**
`Consumidores/año × ${tasas['gasto_promedio_usd']:.2f}` = gasto promedio dispuesto (encuesta).

**Paso 5 – Número de kioskos (modelo estadístico)**
`Consumidores/año ÷ {cap_met['clientes_anuales_por_kiosko']:,} clientes/kiosko/año`
Capacidad por kiosko: horario semana (lunes–viernes) + fines de semana, 12 clientes/hora máx,
ocupación objetivo 60%. Total: **{cap_met['clientes_anuales_por_kiosko']:,} clientes/kiosko/año**.

---

**¿Por qué el modelo estadístico da 2 para el Bulevar y el plan de arranque 2026 es 4?**

El modelo estadístico usa **solo las estadísticas de actividades registradas** del parque 2025:
juegos recreativos, zona canina, juegos inclusivos. Estas actividades no capturan el tráfico
total del bulevar como eje de circulación del parque.

El **arranque 2026 con 4 kioskos** incorpora contexto urbano que el modelo estadístico no mide:
- **142,034 habitantes** en el sector bicentenario (tráfico cotidiano no registrado en actividades)
- El bulevar es el **eje de circulación principal** del parque → flujo de paso constante
- **Cobertura mínima funcional:** 1 kiosko cada 500 m en 2,000 m de recorrido = 4 kioskos
- **Estándar de servicio:** menos de 4 kioskos en 2 km de bulevar deja tramos sin oferta comercial

Los 4 kioskos son el **mínimo operacional razonable para el contexto urbano**, no solo lo que
justifican las estadísticas de actividades específicas del parque.
            """)

    # ── TAB 3: ESQUEMAS DE PLANTA ──────────────────────────────────────────────
    with tab_lay:
        st.markdown('<p class="section-header">Simulador de distribución en planta</p>',
                    unsafe_allow_html=True)
        st.info("Seleccione zona, tipo de kiosko y cantidad para ver el esquema de planta.")

        col_sel1, col_sel2, col_sel3 = st.columns(3)
        with col_sel1:
            zona_sel = st.selectbox("Zona:", list(ZONE_DIMENSIONS.keys()),
                                    format_func=lambda z: z.split("–")[1].strip() if "–" in z else z)
        with col_sel2:
            # Mostrar todos los tipos — el modelo valida la viabilidad al final
            rec_tipo = recommend_kiosko_type(zona_sel)
            ksk_sel = st.selectbox(
                "Tipo de kiosko:",
                list(KIOSKO_TIPOS.keys()),
                index=list(KIOSKO_TIPOS.keys()).index(rec_tipo) if rec_tipo in KIOSKO_TIPOS else 0,
            )
        with col_sel3:
            _fase1_lay = min(PLAN_FASES.keys())
            n_default = PLAN_FASES[_fase1_lay]["kioskos"].get(zona_sel, 2)
            n_sel = st.slider("Número de kioskos:", min_value=1, max_value=120,
                              value=min(n_default, 120))

        fit_info = _ancho_util_por_kiosko(zona_sel, ksk_sel)
        cap_info = _max_kioskos_por_longitud(zona_sel, ksk_sel)
        ksk_info = KIOSKO_TIPOS[ksk_sel]
        dim_info = ZONE_DIMENSIONS[zona_sel]

        viab_ancho_icon = "✅" if fit_info["cabe"] else "❌"
        col_i1, col_i2, col_i3, col_i4 = st.columns(4)
        with col_i1:
            st.metric("Área kiosko", f"{ksk_info['area_m2']} m²")
        with col_i2:
            st.metric("Máx. físico práctico", cap_info["max_practico_total"])
        with col_i3:
            st.metric("Viabilidad ancho", viab_ancho_icon)
        with col_i4:
            st.metric("Lados posibles", fit_info["lados_posibles"])

        st.plotly_chart(fig_layout_zona(zona_sel, ksk_sel, n_sel), width="stretch")

        # ── Veredicto de factibilidad ──────────────────────────────────────────
        cabe_ancho    = fit_info["cabe"]
        cabe_cantidad = n_sel <= cap_info["max_practico_total"]
        analisis_dist = commercial_distance_analysis(zona_sel, n_sel)
        dist_ok       = analisis_dist["todos_cumplen"]
        factible      = cabe_ancho and cabe_cantidad and dist_ok

        v_icon  = "✅" if factible else "❌"
        v_color = "#d5f5e3" if factible else "#fde8e8"
        v_borde = "#27ae60" if factible else "#c0392b"
        v_label = "FACTIBLE" if factible else "NO FACTIBLE"
        v_text_color = "#1a6e3c" if factible else "#922b21"

        checks_html = "".join([
            f'<div style="margin:0.3rem 0;">{"✅" if cabe_ancho else "❌"} '
            f'<b>Viabilidad de ancho:</b> kiosko {ksk_sel} '
            f'{"cabe" if cabe_ancho else "NO cabe"} en los {dim_info["ancho_m"]} m de la zona '
            f'({fit_info["observacion"]})</div>',

            f'<div style="margin:0.3rem 0;">{"✅" if cabe_cantidad else "❌"} '
            f'<b>Viabilidad de cantidad:</b> {n_sel} kioskos '
            f'{"dentro del" if cabe_cantidad else "supera el"} máximo físico práctico '
            f'({cap_info["max_practico_total"]} máx. para este tipo)</div>',

            f'<div style="margin:0.3rem 0;">{"✅" if dist_ok else "❌"} '
            f'<b>Distancias comerciales:</b> '
            f'{"todos los pares cumplen la separación mínima por giro" if dist_ok else "hay pares de kioskos que NO cumplen la distancia mínima requerida"}</div>',
        ])

        st.markdown(f"""
        <div style="background:{v_color};border-radius:10px;padding:1.2rem;
                    border-left:6px solid {v_borde};margin-top:0.5rem;">
            <div style="font-size:1.15rem;font-weight:700;color:{v_text_color};">
                {v_icon} Veredicto: <b>{v_label}</b>
            </div>
            <div style="font-size:0.85rem;color:#333;margin-top:0.6rem;">
                {checks_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="insight-box" style="margin-top:0.8rem;">
        <b>Zona:</b> {dim_info['longitud_m']:,} m × {dim_info['ancho_m']} m ·
        <b>Kiosko:</b> {ksk_sel} · <b>Kioskos simulados:</b> {n_sel} ·
        <b>Tipo de zona:</b> {dim_info['tipo_zona']}
        </div>
        """, unsafe_allow_html=True)

# ── Footer institucional (se inyecta una vez, aplica a todas las páginas) ──────
_footer()
