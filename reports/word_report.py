"""
Generador del informe Word del Estudio de Mercado — Kioskos Parque Bicentenario.
"""

import io
import os
from datetime import date

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


# ── helpers ──────────────────────────────────────────────────────────────────

def _fig_png(fig, width=800, height=420):
    """Export plotly figure to PNG bytes (requires kaleido)."""
    try:
        return fig.to_image(format="png", width=width, height=height, scale=2)
    except Exception:
        return None


def _h(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def _p(doc, text, size=11, bold=False, italic=False, color=None, align=None):
    para = doc.add_paragraph()
    run  = para.add_run(text)
    run.bold   = bold
    run.italic = italic
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    if align:
        para.alignment = align
    return para


def _insert_fig(doc, png_bytes, caption, width=6.0):
    if not png_bytes:
        return
    doc.add_picture(io.BytesIO(png_bytes), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].italic    = True
    doc.add_paragraph()


def _kpi_table(doc, rows):
    """Inserta tabla de dos columnas (etiqueta | valor)."""
    tbl = doc.add_table(rows=len(rows) + 1, cols=2)
    tbl.style = "Table Grid"
    for j, h in enumerate(["Indicador", "Valor"]):
        cell = tbl.rows[0].cells[j]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(10)
    for i, (lbl, val) in enumerate(rows, 1):
        tbl.rows[i].cells[0].text = lbl
        tbl.rows[i].cells[1].text = val
        for j in range(2):
            tbl.rows[i].cells[j].paragraphs[0].runs[0].font.size = Pt(10)
    doc.add_paragraph()


def _bullet(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Bullet")


def _numbered(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Number")


# ── report ───────────────────────────────────────────────────────────────────

def generate_word_report(enc, kpis, stats, fc):
    """
    Genera el informe Word completo y devuelve bytes listos para descarga.

    Parámetros
    ----------
    enc    : DataFrame de encuesta
    kpis   : dict de KPIs calculados por pipeline
    stats  : dict de estadísticas de visitas del parque
    fc     : resultado de forecast_kioskos()
    """
    from models.kiosko_model import (
        fig_kioskos_por_zona, fig_visitantes_vs_demanda,
        fig_trafico_parque_total, fig_demanda_vs_plan_parque,
        fig_ingresos_fases_zona, fig_ingresos_fases_por_kiosko,
        PLAN_FASES, PROYECCION_PARAMS,
    )
    from analysis.univariate import pie_categorica, bar_categorica
    from analysis.sentiment import analyze_comments, topics_bar

    # ── pre-compute ──────────────────────────────────────────────────────────
    n_enc        = kpis["n_encuestados"]
    vis_2025     = kpis["total_visitas_parque_2025"]
    vis_dia      = round(vis_2025 / 365)
    cons_pct     = kpis["consumiria_pct"]
    cons_dia     = round(vis_dia * cons_pct / 100)
    gasto        = kpis["gasto_promedio_usd"]
    calif        = kpis["calificacion_promedio"]
    aprueba      = kpis["aprueba_kiosko_pct"]
    mejora       = kpis["mejora_experiencia_pct"]
    edad         = kpis["edad_promedio"]
    pob_2026     = PROYECCION_PARAMS["poblacion_sector_hab"][2026]
    pob_2036     = PROYECCION_PARAMS["poblacion_sector_hab"][2036]

    result_df = None
    if "comentarios" in enc.columns or any("suger" in c.lower() for c in enc.columns):
        try:
            result_df = analyze_comments(enc)
        except Exception:
            pass

    # ── documento ────────────────────────────────────────────────────────────
    doc = Document()

    # Márgenes A4
    sec = doc.sections[0]
    sec.page_width    = Cm(21)
    sec.page_height   = Cm(29.7)
    sec.left_margin   = Cm(3)
    sec.right_margin  = Cm(2.5)
    sec.top_margin    = Cm(2.5)
    sec.bottom_margin = Cm(2.5)

    # ━━━ PORTADA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    for _ in range(5):
        doc.add_paragraph()

    t = doc.add_heading("ESTUDIO DE MERCADO", 0)
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER

    s = doc.add_heading("Kioskos Comerciales — Parque Bicentenario", 1)
    s.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph()

    org = doc.add_paragraph("Empresa Pública Metropolitana de Movilidad y Obras Públicas")
    org.alignment = WD_ALIGN_PARAGRAPH.CENTER
    org.runs[0].bold = True
    org.runs[0].font.size = Pt(12)

    sub = doc.add_paragraph("EPMMOP — Promoción de Servicios")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].font.size = Pt(11)

    doc.add_paragraph()

    fecha = doc.add_paragraph(f"Quito, {date.today().strftime('%B de %Y').capitalize()}")
    fecha.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fecha.runs[0].font.size = Pt(11)

    doc.add_page_break()

    # ━━━ RESUMEN EJECUTIVO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "RESUMEN EJECUTIVO", 1)

    _p(doc, (
        f"El presente estudio evalúa la viabilidad de implementar kioskos comerciales en el Parque "
        f"Bicentenario de Quito, a partir de una encuesta directa a {n_enc} visitantes y el análisis "
        f"de {vis_2025:,} registros de visita del año 2025. Los resultados revelan una demanda "
        f"potencial sólida: el {cons_pct}% de los visitantes consumiría productos o servicios "
        f"comerciales dentro del parque, equivalente a {cons_dia} personas por día. "
        f"La zona de intervención propuesta es el Bulevar de las Canchas (160 m lineales, hasta 50 "
        f"kioskos). El plan contempla arrancar con 10 kioskos en 2026 y escalar a 25 al 2036. "
        f"El Área de Intervención Valorativa (AIVA) del sector Bicentenario se ubica en $267/m², "
        f"con proyección de crecimiento hacia los $1,895/m² del sector La Carolina."
    ))

    doc.add_paragraph()
    _h(doc, "Indicadores clave del estudio", 2)
    _kpi_table(doc, [
        ("Visitas registradas 2025",            f"{vis_2025:,}"),
        ("Visitantes diarios promedio",         f"{vis_dia:,}"),
        ("Encuestados",                         f"{n_enc}"),
        ("Consumidores potenciales por día",    f"{cons_dia} ({cons_pct}%)"),
        ("Gasto promedio dispuesto",            f"${gasto}"),
        ("Aprobación de kioskos",               f"{aprueba}%"),
        ("Mejora percibida en experiencia",     f"{mejora}%"),
        ("Calificación actual del parque",      f"{calif}/5"),
        ("Edad promedio del visitante",         f"{edad} años"),
        ("AIVA sector Bicentenario",            "$267/m²"),
        ("AIVA referencia La Carolina",         "$1,895/m²"),
        ("Kioskos propuestos (inicio / meta)",  "10 (2026) → 25 (2036)"),
    ])

    doc.add_page_break()

    # ━━━ 1. INTRODUCCIÓN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "1. INTRODUCCIÓN", 1)

    _h(doc, "1.1 Antecedentes", 2)
    _p(doc, (
        "El Parque Bicentenario ocupa los terrenos del antiguo Aeropuerto Mariscal Sucre de Quito. "
        "Con aproximadamente 50 hectáreas de área verde, es el parque urbano más grande de la ciudad "
        "y uno de los más significativos de la región andina. Administrado por la EPMMOP, recibe "
        f"más de {vis_2025:,} visitantes al año y constituye el principal espacio de esparcimiento, "
        "deporte y cultura del norte de Quito."
    ))

    _h(doc, "1.2 Problemática", 2)
    _p(doc, (
        "El parque carece de una oferta comercial estructurada. La ausencia de servicios de "
        "alimentación, bebidas y productos básicos genera una demanda insatisfecha que ha sido "
        "identificada y cuantificada en este estudio. Esta brecha representa una oportunidad "
        "económica para el municipio, los operadores locales y los propietarios de inmuebles "
        "del sector."
    ))

    _h(doc, "1.3 Justificación", 2)
    _p(doc, (
        "La implementación de kioskos modulares en el Bulevar de las Canchas permite una "
        "intervención reversible, escalable y coherente con los lineamientos de uso del espacio "
        "público del Municipio del Distrito Metropolitano de Quito. El modelo se sustenta en "
        "evidencia directa de demanda ciudadana y en el análisis del entorno urbano y comercial "
        "del sector."
    ))

    doc.add_page_break()

    # ━━━ 2. OBJETIVOS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "2. OBJETIVOS", 1)

    _h(doc, "2.1 Objetivo general", 2)
    _p(doc, (
        "Determinar la viabilidad comercial y operativa de implementar kioskos de venta de "
        "productos y servicios en el Parque Bicentenario de Quito, a través del análisis de "
        "la demanda, el perfil del visitante y las condiciones del entorno urbano."
    ))

    _h(doc, "2.2 Objetivos específicos", 2)
    _bullet(doc, [
        "Caracterizar el perfil sociodemográfico del visitante del Parque Bicentenario.",
        "Cuantificar la demanda potencial de productos y servicios comerciales dentro del parque.",
        "Identificar los productos y servicios con mayor aceptación entre los visitantes.",
        "Establecer un modelo de negocio viable y escalable para la operación de kioskos.",
        "Proyectar los ingresos esperados a 10 años bajo el plan de implementación por fases.",
        "Evaluar el impacto en la valorización del suelo (AIVA) del sector Bicentenario.",
    ])

    doc.add_page_break()

    # ━━━ 3. METODOLOGÍA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "3. METODOLOGÍA", 1)
    _p(doc, (
        "El estudio combina métodos cuantitativos y cualitativos para obtener una visión integral "
        "del mercado potencial. La investigación se realizó en dos etapas: encuesta de campo "
        "a visitantes del parque y análisis estadístico de los registros históricos de visita."
    ))

    _h(doc, "3.1 Encuesta de campo", 2)
    _p(doc, (
        f"Se aplicaron {n_enc} encuestas estructuradas a visitantes del Parque Bicentenario "
        f"mediante muestreo aleatorio sistemático. El instrumento incluyó variables "
        "sociodemográficas (edad, género, sector de residencia), patrones de visita (frecuencia, "
        "días, horarios, motivo), preferencias de consumo y percepción sobre la oferta actual "
        "y potencial de servicios."
    ))
    _kpi_table(doc, [
        ("Tamaño de muestra",       f"{n_enc} encuestados"),
        ("Herramienta",             "Google Forms"),
        ("Tipo de muestreo",        "Aleatorio sistemático"),
        ("Variables analizadas",    "17 preguntas"),
        ("Análisis estadístico",    "Python — pandas, scipy, plotly"),
    ])

    _h(doc, "3.2 Análisis de registros de visita", 2)
    _p(doc, (
        f"Se analizaron {vis_2025:,} registros de visita al Parque Bicentenario correspondientes "
        "al año 2025, proporcionados por la EPMMOP. El análisis permitió identificar patrones "
        "temporales de afluencia, tendencias estacionales, distribución mensual y la demanda "
        "diaria promedio como base para la proyección comercial."
    ))

    doc.add_page_break()

    # ━━━ 4. PERFIL SOCIODEMOGRÁFICO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "4. PERFIL SOCIODEMOGRÁFICO DEL VISITANTE", 1)
    _p(doc, (
        f"El visitante típico del Parque Bicentenario es residente del norte de Quito (78.2%), "
        f"con una edad promedio de {edad} años. El domingo es el día de mayor afluencia (67.5%) "
        "en el horario de 09:00 a 12:00 h, mientras que en días de semana el pico es de 04:00 "
        "a 09:00 h (deportistas mañaneros). El grupo de visita predominante es la familia (45.6%)."
    ))

    _kpi_table(doc, [
        ("Visitantes totales 2025",              f"{vis_2025:,}"),
        ("Visitantes diarios promedio",          f"{vis_dia:,}"),
        ("Edad promedio",                        f"{edad} años"),
        ("Día más visitado",                     "Domingo (67.5%)"),
        ("Sector de residencia mayoritario",     "Norte de Quito (78.2%)"),
        ("Grupo de visita predominante",         "Familia (45.6%)"),
        ("Motivo de visita principal",           "Recreación (29.1%)"),
        ("Servicio complementario más pedido",   "Baños (76.2%)"),
    ])

    # Gráficos demográficos
    if "genero" in enc.columns:
        fig_gen = pie_categorica(enc["genero"], "Distribución por género")
        _insert_fig(doc, _fig_png(fig_gen, 520, 360),
                    "Gráfico 1. Distribución por género de los visitantes encuestados", 4.5)

    if "edad" in enc.columns:
        edad_bins = pd.cut(
            enc["edad"].dropna(),
            bins=[0, 20, 30, 40, 50, 60, 120],
            labels=["< 20", "20–29", "30–39", "40–49", "50–59", "60+"],
        )
        fig_edad = bar_categorica(
            edad_bins.astype(str), "Distribución por grupo de edad",
            order=["< 20", "20–29", "30–39", "40–49", "50–59", "60+"],
            horizontal=True,
        )
        _insert_fig(doc, _fig_png(fig_edad, 580, 360),
                    "Gráfico 2. Distribución de visitantes por grupos de edad", 5.0)

    if "motivo_visita" in enc.columns:
        fig_mot = pie_categorica(enc["motivo_visita"], "Motivo principal de visita")
        _insert_fig(doc, _fig_png(fig_mot, 520, 360),
                    "Gráfico 3. Motivo principal de visita al Parque Bicentenario", 4.5)

    doc.add_page_break()

    # ━━━ 5. ANÁLISIS DE DEMANDA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "5. ANÁLISIS DE DEMANDA", 1)

    _h(doc, "5.1 Afluencia al parque", 2)
    _p(doc, (
        f"El Parque Bicentenario registró {vis_2025:,} visitas durante 2025, con un promedio de "
        f"{vis_dia:,} visitantes diarios. La mayor afluencia se concentra los domingos. "
        "La tendencia de visitas muestra estacionalidad marcada en los meses de vacaciones "
        "escolares (julio, agosto) y en el primer trimestre del año."
    ))

    fig_traf = fig_trafico_parque_total(stats)
    _insert_fig(doc, _fig_png(fig_traf, 800, 430),
                "Gráfico 4. Tráfico de visitantes al Parque Bicentenario — 2025", 6.0)

    _h(doc, "5.2 Demanda potencial de servicios comerciales", 2)
    _p(doc, (
        f"El {cons_pct}% de los encuestados ({cons_dia} personas/día) afirmó que consumiría "
        f"productos o servicios si estuvieran disponibles en el parque. El gasto promedio "
        f"dispuesto es de ${gasto} por visita. El {aprueba}% aprueba la implementación de "
        f"kioskos y el {mejora}% considera que mejoraría su experiencia."
    ))

    fig_dem = fig_demanda_vs_plan_parque(stats, enc)
    _insert_fig(doc, _fig_png(fig_dem, 800, 430),
                "Gráfico 5. Demanda potencial de consumidores vs. plan de kioskos", 6.0)

    _h(doc, "5.3 Productos y servicios más demandados", 2)
    _p(doc, (
        "Los snacks y bebidas encabezan la demanda (38.0%), seguidos de bebidas calientes, "
        "comida rápida y helados. Entre los servicios complementarios, los baños son la "
        "prioridad ciudadana más urgente (76.2%), seguida de iluminación y zonas de descanso."
    ))

    if result_df is not None:
        try:
            fig_top = topics_bar(result_df)
            _insert_fig(doc, _fig_png(fig_top, 800, 430),
                        "Gráfico 6. Temas más solicitados por los visitantes (análisis de comentarios)", 6.0)
        except Exception:
            pass

    doc.add_page_break()

    # ━━━ 6. MODELO COMERCIAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "6. MODELO COMERCIAL", 1)

    _h(doc, "6.1 Zona de intervención", 2)
    _p(doc, (
        "La zona seleccionada es el Bulevar de las Canchas (Zona 1), corredor peatonal de "
        "160 m lineales que atraviesa el área de canchas deportivas del parque. Esta ubicación "
        "maximiza la exposición a los visitantes activos y aprovecha el flujo natural de "
        "tránsito peatonal que conecta los accesos norte y sur del parque."
    ))
    _kpi_table(doc, [
        ("Zona",                     "Zona 1 – Bulevar de las Canchas"),
        ("Longitud total",           "160 m lineales"),
        ("Número de secciones",      "4"),
        ("Capacidad máxima",         "50 kioskos"),
        ("Dimensión estándar",       "2.5 m × 2.0 m por kiosko"),
        ("Ancho del pasillo",        "6.0 m"),
    ])

    # Render / imagen de zona
    zona_img = "src/img/render_bicentenario_bulevar.png"
    if os.path.exists(zona_img):
        doc.add_picture(zona_img, width=Inches(6.0))
        cap = doc.add_paragraph("Figura 1. Zona de intervención — Bulevar de las Canchas")
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.runs[0].font.size = Pt(9)
        cap.runs[0].italic = True
        doc.add_paragraph()

    prop_img = "src/img/implantación_parque.jpg"
    if os.path.exists(prop_img):
        doc.add_picture(prop_img, width=Inches(6.0))
        cap2 = doc.add_paragraph("Figura 2. Render de la propuesta — Parque Bicentenario")
        cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap2.runs[0].font.size = Pt(9)
        cap2.runs[0].italic = True
        doc.add_paragraph()

    _h(doc, "6.2 Giros comerciales propuestos", 2)
    _p(doc, (
        "Los giros se asignan por rotación para garantizar diversidad de oferta y evitar "
        "competencia directa entre módulos adyacentes. La rotación incluye: Bebidas, "
        "Comida rápida, Helados y café, Snacks saludables, Artículos para mascotas y "
        "souvenirs, y Deportivo."
    ))

    _h(doc, "6.3 Plan de implementación por fases", 2)
    _p(doc, (
        "El proyecto se despliega en 4 fases a lo largo de 10 años, condicionado a hitos "
        "de desarrollo urbano (apertura del Metro, densificación del sector). Cada fase "
        "incrementa la oferta de kioskos en función del crecimiento de la demanda."
    ))

    fase_rows = []
    for anio, params in sorted(PLAN_FASES.items()):
        zona_kio = list(params["kioskos"].values())
        total_kio = sum(zona_kio)
        fase_rows.append((
            params["fase"],
            str(anio),
            params["hito"],
            str(total_kio),
            f"{params['poblacion_hab']:,} hab.",
        ))

    tbl_fases = doc.add_table(rows=len(fase_rows) + 1, cols=5)
    tbl_fases.style = "Table Grid"
    for j, h in enumerate(["Fase", "Año", "Hito", "Kioskos", "Población sector"]):
        cell = tbl_fases.rows[0].cells[j]
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.size = Pt(9)
    for i, row in enumerate(fase_rows, 1):
        for j, val in enumerate(row):
            tbl_fases.rows[i].cells[j].text = val
            tbl_fases.rows[i].cells[j].paragraphs[0].runs[0].font.size = Pt(9)
    doc.add_paragraph()

    fig_kio = fig_kioskos_por_zona(fc)
    _insert_fig(doc, _fig_png(fig_kio, 800, 430),
                "Gráfico 7. Proyección de kioskos 2026–2036", 6.0)

    doc.add_page_break()

    # ━━━ 7. PROYECCIÓN FINANCIERA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "7. PROYECCIÓN FINANCIERA", 1)
    _p(doc, (
        f"La proyección se basa en el crecimiento esperado de la población del sector "
        f"(de {pob_2026:,} en 2026 a {pob_2036:,} en 2036), el aumento progresivo de visitantes "
        "y el incremento gradual de kioskos activos. Los modelos de ingresos incorporan "
        "el ticket promedio declarado por los encuestados y escenarios conservadores de "
        "ocupación diaria."
    ))

    _h(doc, "7.1 Proyección de visitantes y demanda", 2)
    fig_vis = fig_visitantes_vs_demanda(stats, enc)
    _insert_fig(doc, _fig_png(fig_vis, 800, 430),
                "Gráfico 8. Visitantes del parque vs. consumidores potenciales por fase", 6.0)

    _h(doc, "7.2 Ingresos proyectados", 2)
    _p(doc, (
        "Se presentan dos perspectivas de ingresos: la proyección consolidada por zona "
        "y la proyección individual por kiosko, lo que permite evaluar la rentabilidad "
        "tanto del proyecto global como de cada operador."
    ))

    fig_ing_zona = fig_ingresos_fases_zona(stats, enc)
    _insert_fig(doc, _fig_png(fig_ing_zona, 800, 430),
                "Gráfico 9. Proyección de ingresos totales de la zona 2026–2036", 6.0)

    fig_ing_kio = fig_ingresos_fases_por_kiosko(stats, enc)
    _insert_fig(doc, _fig_png(fig_ing_kio, 800, 430),
                "Gráfico 10. Proyección de ingresos promedio por kiosko 2026–2036", 6.0)

    _h(doc, "7.3 Impacto en valorización del suelo (AIVA)", 2)
    _p(doc, (
        "El Área de Intervención Valorativa del sector Bicentenario se ubica en $267/m², "
        "frente a los $1,895/m² del sector La Carolina. La consolidación del parque como "
        "polo comercial y recreativo proyecta un crecimiento sostenido del AIVA, beneficiando "
        "directamente a propietarios de inmuebles del sector al incrementar el valor de venta "
        "y arrendamiento de sus propiedades conforme el proyecto avance."
    ))
    _kpi_table(doc, [
        ("AIVA actual Bicentenario",  "$267/m²"),
        ("AIVA referencia La Carolina", "$1,895/m²"),
        ("Potencial de crecimiento",  "~7x respecto al valor actual"),
    ])

    doc.add_page_break()

    # ━━━ 8. CONCLUSIONES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "8. CONCLUSIONES Y RECOMENDACIONES", 1)

    _h(doc, "8.1 Conclusiones", 2)
    _bullet(doc, [
        f"Existe demanda potencial sólida y verificada: el {cons_pct}% de los visitantes "
        f"({cons_dia}/día) consumiría en el parque si hubiera oferta disponible.",
        f"El parque recibe {vis_2025:,} visitas al año ({vis_dia:,}/día), con pico los domingos "
        f"y mayor concentración en el Bulevar de las Canchas.",
        "Los snacks, bebidas y comida rápida encabezan la preferencia de consumo; "
        "los baños son el servicio complementario más urgente (76.2%).",
        f"El modelo de kioskos por fases (10 → 25 unidades, 2026–2036) es técnicamente "
        "viable y alineado con el crecimiento urbano del sector.",
        f"La ciudadanía tiene actitud favorable: {aprueba}% aprueba los kioskos y "
        f"{mejora}% cree que mejorarían su experiencia en el parque.",
        "El AIVA del sector ($267/m²) tiene alto potencial de valorización hacia los niveles "
        "del sector La Carolina ($1,895/m²), generando un beneficio colateral para propietarios.",
        "El acceso al Metro (estaciones Bicentenario y Andalucía, desde 2029) transformará "
        "el corredor en un eje de movilidad urbana que multiplicará la afluencia.",
    ])

    _h(doc, "8.2 Recomendaciones", 2)
    _bullet(doc, [
        "Iniciar la Fase 1 en 2026 con 10 kioskos de giros diversificados para validar el "
        "modelo comercial antes de escalar.",
        "Priorizar la instalación de baños públicos como servicio complementario inmediato, "
        "dado el 76.2% de demanda identificada.",
        "Establecer contratos de concesión con cláusulas de revisión anual para ajustar giros "
        "y tarifas según la evolución real de la demanda.",
        "Implementar un sistema de monitoreo de ventas para optimizar el mix comercial y "
        "tomar decisiones basadas en datos en tiempo real.",
        "Comunicar activamente el impacto del proyecto en el AIVA del sector a propietarios "
        "e inversores para generar alianzas y apoyo ciudadano.",
        "Evaluar la extensión del modelo a otras zonas del parque en la Fase 4 (2036) "
        "si los resultados de las fases previas superan las proyecciones.",
    ])

    doc.add_page_break()

    # ━━━ ANEXOS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "ANEXOS", 1)

    _h(doc, "Anexo A — Instrumento de encuesta", 2)
    _p(doc, f"Variables incluidas en el formulario de encuesta ({n_enc} respuestas recolectadas):")
    _numbered(doc, [
        "Edad",
        "Género",
        "¿En qué sector reside?",
        "¿Con quién visita generalmente el Parque Bicentenario?",
        "¿Con qué frecuencia visita el Parque Bicentenario?",
        "¿Qué días usualmente visita el parque?",
        "¿En qué horario usualmente visita el parque?",
        "¿Cuál es el principal motivo de su visita?",
        "¿Consumiría productos o servicios dentro del parque?",
        "¿Qué productos o servicios consumiría dentro del parque?",
        "¿Cuánto estaría dispuesto a gastar en la compra de estos productos o servicios?",
        "¿Considera adecuada la implementación de kioskos comerciales dentro del parque?",
        "¿En qué zonas accedería con mayor facilidad para compra o adquisición de productos?",
        "¿Considera que los kioskos mejorarían su experiencia en el parque?",
        "¿Cómo califica actualmente la oferta de servicios dentro del parque?",
        "¿Qué servicios complementarios considera prioritarios?",
        "Comentarios y sugerencias",
    ])

    _h(doc, "Anexo B — Fases del plan de implementación", 2)
    for anio, params in sorted(PLAN_FASES.items()):
        _p(doc, params["fase"], bold=True, size=10)
        _p(doc, params["contexto"], size=10)
        doc.add_paragraph()

    # ── guardar ──────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()
