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
    consumo_vs_gasto, chi2_test,
)
from analysis.multivariate import (
    pca_biplot, scree_plot, kmeans_clusters, elbow_silhouette,
    correlation_matrix,
)
from analysis.sentiment import (
    analyze_comments, sentiment_pie, topics_bar,
    wordcloud_img, top_keywords, sentiment_by_topic_heatmap,
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
    fig_trafico_proyeccion, fig_demanda_vs_plan,
    fig_plan_fases, tabla_plan_fases,
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
    st.markdown("""
    <style>
        .main .block-container { padding-bottom: 100px !important; }
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
        <img src="/app/static/logo_escudo.png"
             style="height:48px;display:block;margin:0 auto 0.3rem;"
             alt="EPMMOP">
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

    # Hallazgos clave del estudio
    st.markdown('<p class="section-header">Hallazgos clave del estudio</p>', unsafe_allow_html=True)
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("""
        <div class="insight-box">
        <b>✅ Alta demanda de servicios comerciales:</b> La mayoría de visitantes consumiría
        productos o servicios dentro del parque, especialmente alimentos saludables y bebidas.
        </div>
        <div class="insight-box">
        <b>📅 Pico de visitas en fin de semana:</b> Sábado y domingo concentran más del 60%
        del flujo semanal, determinando el dimensionamiento de kioskos.
        </div>
        <div class="insight-box">
        <b>🎯 Zonas de mayor accesibilidad:</b> Los visitantes identifican el bulevar de la
        Av. Amazonas y las zonas deportivas como los puntos de mayor facilidad de acceso.
        </div>
        """, unsafe_allow_html=True)
    with col_h2:
        st.markdown("""
        <div class="insight-box">
        <b>💰 Disposición de gasto moderada:</b> La franja de mayor aceptación es entre $2 y $10,
        orientando la oferta hacia productos de precio accesible y rápido consumo.
        </div>
        <div class="insight-box">
        <b>⚠️ Oferta actual insuficiente:</b> La calificación promedio de los servicios actuales
        indica espacio de mejora significativo que los kioskos pueden cubrir.
        </div>
        <div class="insight-box">
        <b>🌿 Perfil de visitante activo y familiar:</b> El usuario típico visita con familia,
        tiene motivación recreativa y deportiva, con edad media de {:.0f} años.
        </div>
        """.format(kpis["edad_promedio"]), unsafe_allow_html=True)

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

    elif variable == "Género":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(pie_categorica(enc["genero"], "Género"), width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["genero"]), width="stretch", hide_index=True)

    elif variable == "Sector de residencia":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(bar_categorica(enc["sector_residencia"], "Sector de residencia",
                                           horizontal=True), width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["sector_residencia"]), width="stretch",
                         hide_index=True)

    elif variable == "Acompañante":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(pie_categorica(enc["acompanante"], "¿Con quién visita el parque?"),
                            width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["acompanante"]), width="stretch", hide_index=True)

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

    elif variable == "Días que visita (multi-opción)":
        st.info("⚠️ Selección múltiple: los porcentajes pueden superar el 100% ya que cada "
                "encuestado pudo seleccionar varios días.")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(bar_multiselect(enc, "dias_visita", "Días de visita al parque"),
                            width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_multiselect(enc, "dias_visita"), width="stretch",
                         hide_index=True)

    elif variable == "Motivo de visita":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(bar_categorica(enc["motivo_visita"], "Motivo principal de visita",
                                           horizontal=True), width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["motivo_visita"]), width="stretch",
                         hide_index=True)

    elif variable == "¿Consumiría productos?":
        col1, col2 = st.columns([1, 1])
        with col1:
            st.plotly_chart(pie_categorica(enc["consumiria"], "¿Consumiría productos en el parque?"),
                            width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["consumiria"]), width="stretch", hide_index=True)

    elif variable == "Productos de interés":
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(bar_categorica(enc["productos_interes"], "Productos o servicios de interés",
                                           horizontal=True), width="stretch")
        with col2:
            st.markdown("#### Tabla de frecuencias")
            st.dataframe(table_frecuencia(enc["productos_interes"]), width="stretch",
                         hide_index=True)

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
            st.dataframe(table_multiselect(enc, "servicios_prioritarios"),
                         width="stretch", hide_index=True)


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
            **Prueba Chi-cuadrado:**
            - χ² = {chi['chi2']} | p = {chi['p_valor']}
            - Grados de libertad: {chi['grados_libertad']}
            - **{chi['interpretacion']}**
            """)

    elif analisis == "Gasto dispuesto por género":
        st.plotly_chart(consumo_vs_gasto(enc), width="stretch")
        st.markdown("#### Estadísticos descriptivos por género")
        gasto_map = {"Menos de $2": 1.5, "Entre $2 y $5": 3.5, "Entre $5 y $10": 7.5,
                     "Entre $10 y $20": 15.0, "Más de $20": 22.0}
        enc_g = enc.copy()
        enc_g["gasto_num"] = enc_g["gasto_dispuesto"].map(gasto_map)
        st.dataframe(stats_tabla_bivariada(enc_g, "gasto_num", "genero"),
                     width="stretch", hide_index=True)

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
            **Chi-cuadrado:**
            - χ² = {chi['chi2']}
            - p = {chi['p_valor']}
            - **{chi['interpretacion']}**
            """)

    elif analisis == "Aprobación de kioskos por edad (boxplot)":
        st.plotly_chart(
            boxplot_continua_vs_cat(enc, "edad", "kiosko_adecuado",
                                    "Edad (años)", "¿Kioskos adecuados?"),
            width="stretch"
        )
        st.dataframe(stats_tabla_bivariada(enc, "edad", "kiosko_adecuado"),
                     width="stretch", hide_index=True)

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
            **Chi-cuadrado:**
            - χ² = {chi['chi2']}
            - p = {chi['p_valor']}
            - **{chi['interpretacion']}**
            """)

    elif analisis == "Gasto dispuesto por acompañante":
        st.plotly_chart(
            crosstab_grouped(enc, "acompanante", "gasto_dispuesto",
                             "Con quién visita", "Gasto dispuesto"),
            width="stretch"
        )

    elif analisis == "Calificación de oferta por frecuencia":
        st.plotly_chart(
            boxplot_continua_vs_cat(enc, "calificacion_oferta", "frecuencia_visita",
                                    "Calificación (1-5)", "Frecuencia de visita"),
            width="stretch"
        )
        st.dataframe(stats_tabla_bivariada(enc, "calificacion_oferta", "frecuencia_visita"),
                     width="stretch", hide_index=True)

    elif analisis == "Horario de visita por género (boxplot)":
        st.plotly_chart(
            boxplot_continua_vs_cat(enc, "horario_visita", "genero",
                                    "Horario de visita (h)", "Género"),
            width="stretch"
        )
        st.dataframe(stats_tabla_bivariada(enc, "horario_visita", "genero"),
                     width="stretch", hide_index=True)

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
            **Chi-cuadrado:**
            - χ² = {chi['chi2']}
            - p = {chi['p_valor']}
            - **{chi['interpretacion']}**
            """)


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

    with tab2:
        color_opt = st.selectbox("Colorear por:", ["genero", "consumiria", "frecuencia_visita",
                                                    "motivo_visita"], key="pca_color")
        enc_clean = enc[COLS_ENCODE + [color_opt]].dropna()
        fig_pca, var_exp = pca_biplot(enc_clean, COLS_ENCODE, COLS_LABELS, color_col=color_opt)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig_pca, width="stretch")
        with col2:
            st.markdown("#### Varianza explicada")
            st.dataframe(var_exp, width="stretch", hide_index=True)
            st.plotly_chart(scree_plot(enc_clean, COLS_ENCODE), width="stretch")

    with tab3:
        n_k = st.slider("Número de segmentos (k):", min_value=2, max_value=6, value=3)
        enc_km = enc[COLS_ENCODE].dropna()
        fig_km, labels, perfil = kmeans_clusters(enc_km, COLS_ENCODE, n_clusters=n_k,
                                                  color_labels=list(COLS_LABELS.values()))
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig_km, width="stretch")
        with col2:
            st.markdown("#### Perfil de cada segmento")
            st.dataframe(perfil.T, width="stretch")
        st.markdown("#### Tamaño de cada segmento")
        seg_counts = labels.value_counts().sort_index().reset_index()
        seg_counts.columns = ["Segmento", "N"]
        seg_counts["Segmento"] = [f"Segmento {k+1}" for k in seg_counts["Segmento"]]
        seg_counts["%"] = (seg_counts["N"] / seg_counts["N"].sum() * 100).round(1)
        st.dataframe(seg_counts, width="stretch", hide_index=True)

    with tab4:
        enc_elbow = enc[COLS_ENCODE].dropna()
        st.plotly_chart(elbow_silhouette(enc_elbow, COLS_ENCODE, max_k=8),
                        width="stretch")
        st.caption("El punto de quiebre en la curva de inercia (codo) y el pico en Silhouette "
                   "indican el número óptimo de clusters.")


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
# SECCIÓN 6: MODELO COMERCIAL
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
        st.plotly_chart(fig_trafico_proyeccion(), width="stretch")

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

        st.plotly_chart(fig_demanda_vs_plan(), width="stretch")

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
        st.dataframe(tabla_resumen_forecast(forecast), width="stretch", hide_index=True)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.plotly_chart(fig_kioskos_por_zona(forecast), width="stretch")
        with col_g2:
            st.plotly_chart(fig_visitantes_vs_demanda(forecast), width="stretch")

        st.plotly_chart(fig_ingresos_potenciales(forecast), width="stretch")

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
