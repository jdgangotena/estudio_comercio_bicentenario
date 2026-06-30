"""
Generador del informe Word del Estudio de Mercado — Kioskos Parque Bicentenario.

Recibe un dict `charts` con las imágenes ya renderizadas como bytes PNG.
Esto evita llamar a kaleido dentro del contexto de Streamlit.
"""

import io
import os
from datetime import date

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH


# ── helpers ───────────────────────────────────────────────────────────────────

def _h(doc, text, level=1):
    p = doc.add_heading(text, level=level)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    return p


def _p(doc, text, size=11, bold=False):
    para = doc.add_paragraph()
    run  = para.add_run(text)
    run.bold = bold
    run.font.size = Pt(size)
    return para


def _insert_img(doc, png_bytes, caption, width=6.0):
    """Inserta imagen PNG (bytes) con pie de figura."""
    if not png_bytes:
        return
    doc.add_picture(io.BytesIO(png_bytes), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.runs[0].font.size = Pt(9)
    cap.runs[0].italic    = True
    doc.add_paragraph()


def _kpi_table(doc, rows):
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


# ── chart generator con matplotlib (sin kaleido, funciona en Streamlit) ───────

def build_charts(enc, kpis, stats, fc):
    """
    Genera todas las imágenes con matplotlib puro (sin kaleido/subprocesos).
    Retorna dict: nombre → bytes PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    from models.kiosko_model import PLAN_FASES, COMMERCIAL_PARAMS

    BLUE   = "#1a3a5c"
    ORANGE = "#e67e22"
    GREEN  = "#27ae60"
    COLS   = [BLUE, ORANGE, GREEN, "#8e44ad", "#e74c3c", "#16a085", "#f39c12"]

    def _save(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        buf.seek(0)
        data = buf.read()
        plt.close(fig)
        return data

    def _pie(series, title, w=5, h=4):
        vc = series.dropna().str.strip().value_counts()
        fig, ax = plt.subplots(figsize=(w, h))
        wedges, texts, autotexts = ax.pie(
            vc.values, labels=vc.index, autopct="%1.1f%%",
            colors=COLS[:len(vc)], startangle=90,
            wedgeprops=dict(linewidth=1, edgecolor="white"),
        )
        for t in autotexts:
            t.set_fontsize(9)
        ax.set_title(title, fontsize=12, fontweight="bold", color=BLUE, pad=12)
        return _save(fig)

    def _hbar(labels, values, title, w=6, h=4):
        fig, ax = plt.subplots(figsize=(w, h))
        bars = ax.barh(labels, values, color=BLUE, alpha=0.85, height=0.55)
        ax.set_xlabel("Número de encuestados", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold", color=BLUE, pad=10)
        ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + max(values)*0.01,
                    bar.get_y() + bar.get_height()/2,
                    str(int(val)), va="center", fontsize=9)
        ax.set_xlim(0, max(values) * 1.18)
        fig.tight_layout()
        return _save(fig)

    charts = {}

    # ── 1. Género ──────────────────────────────────────────────────────────
    try:
        if "genero" in enc.columns:
            charts["genero"] = _pie(enc["genero"], "Distribución por género")
    except Exception:
        charts["genero"] = None

    # ── 2. Edad ───────────────────────────────────────────────────────────
    try:
        if "edad" in enc.columns:
            orden = ["< 20", "20–29", "30–39", "40–49", "50–59", "60+"]
            bins  = pd.cut(enc["edad"].dropna(),
                           bins=[0,20,30,40,50,60,120], labels=orden)
            vc = bins.value_counts().reindex(orden).fillna(0)
            charts["edad"] = _hbar(vc.index.tolist(), vc.values.tolist(),
                                   "Distribución por grupo de edad")
    except Exception:
        charts["edad"] = None

    # ── 3. Motivo de visita ───────────────────────────────────────────────
    try:
        if "motivo_visita" in enc.columns:
            charts["motivo"] = _pie(enc["motivo_visita"], "Motivo principal de visita")
    except Exception:
        charts["motivo"] = None

    # ── 4. Tráfico mensual 2025 ───────────────────────────────────────────
    try:
        meses_dfs = [
            df[["mes", "num_mes", "total"]].copy()
            for df in stats.values()
            if "total" in df.columns and "mes" in df.columns
        ]
        agg = (pd.concat(meses_dfs)
               .groupby(["mes", "num_mes"])["total"].sum()
               .reset_index().sort_values("num_mes"))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(agg["mes"], agg["total"], color=BLUE, linewidth=2.2,
                marker="o", markersize=5)
        ax.fill_between(range(len(agg)), agg["total"].values,
                        alpha=0.12, color=BLUE)
        media = agg["total"].mean()
        ax.axhline(media, color="#e74c3c", linewidth=1.4, linestyle="--",
                   label=f"Media: {int(media):,}")
        ax.legend(fontsize=9)
        ax.set_xticks(range(len(agg)))
        ax.set_xticklabels(agg["mes"], rotation=35, ha="right", fontsize=8)
        ax.set_xlabel("Mes", fontsize=9)
        ax.set_ylabel("Visitas", fontsize=9)
        ax.set_title("Tráfico de visitantes al Parque Bicentenario — 2025",
                     fontsize=12, fontweight="bold", color=BLUE)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        charts["trafico"] = _save(fig)
    except Exception:
        charts["trafico"] = None

    # ── 5. Demanda vs plan de kioskos ─────────────────────────────────────
    try:
        cons_pct = kpis["consumiria_pct"] / 100
        vis_dia  = round(kpis["total_visitas_parque_2025"] / 365)
        fases = sorted(PLAN_FASES.items())
        anios = [str(a) for a, _ in fases]
        kios  = [sum(p["kioskos"].values()) for _, p in fases]
        pobs  = [p["poblacion_hab"] for _, p in fases]
        demanda = [round(pop * 0.008 * cons_pct) for pop in pobs]

        x = np.arange(len(anios))
        w = 0.38
        fig, ax = plt.subplots(figsize=(7, 4))
        b1 = ax.bar(x - w/2, demanda, w, label="Consumidores potenciales/día",
                    color=BLUE, alpha=0.85)
        b2 = ax.bar(x + w/2, [k * 30 for k in kios], w,
                    label="Capacidad diaria (kioskos × 30 tx)", color=ORANGE, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fase\n{a}" for a in anios], fontsize=9)
        ax.set_ylabel("Personas / transacciones diarias", fontsize=9)
        ax.set_title("Demanda potencial vs. capacidad del plan de kioskos",
                     fontsize=11, fontweight="bold", color=BLUE)
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        charts["demanda"] = _save(fig)
    except Exception:
        charts["demanda"] = None

    # ── 6. Visitantes vs consumidores por fase ────────────────────────────
    try:
        fases = sorted(PLAN_FASES.items())
        anios = [str(a) for a, _ in fases]
        vis   = [round(p["poblacion_hab"] * 0.008) for _, p in fases]
        cons  = [round(v * kpis["consumiria_pct"] / 100) for v in vis]
        x = np.arange(len(anios))
        w = 0.38
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(x - w/2, vis,  w, label="Visitantes/día estimados", color=BLUE, alpha=0.85)
        ax.bar(x + w/2, cons, w, label=f"Consumidores ({kpis['consumiria_pct']}%)", color=GREEN, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fase\n{a}" for a in anios], fontsize=9)
        ax.set_ylabel("Personas por día", fontsize=9)
        ax.set_title("Proyección de visitantes vs. consumidores por fase",
                     fontsize=11, fontweight="bold", color=BLUE)
        ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        charts["visitantes_vs"] = _save(fig)
    except Exception:
        charts["visitantes_vs"] = None

    # ── 7. Kioskos por fase ───────────────────────────────────────────────
    try:
        fases = sorted(PLAN_FASES.items())
        anios = [str(a) for a, _ in fases]
        kios  = [sum(p["kioskos"].values()) for _, p in fases]
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(anios, kios, color=BLUE, alpha=0.85, width=0.5)
        for bar, val in zip(bars, kios):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", fontweight="bold", fontsize=11)
        ax.set_ylabel("Número de kioskos", fontsize=9)
        ax.set_title("Proyección de kioskos — Plan por fases 2026–2036",
                     fontsize=11, fontweight="bold", color=BLUE)
        ax.set_ylim(0, max(kios) * 1.2)
        ax.spines[["top","right"]].set_visible(False)
        fig.tight_layout()
        charts["kioskos"] = _save(fig)
    except Exception:
        charts["kioskos"] = None

    # ── 8. Ingresos por zona ──────────────────────────────────────────────
    try:
        gasto = kpis["gasto_promedio_usd"]
        fases = sorted(PLAN_FASES.items())
        anios = [str(a) for a, _ in fases]
        ingresos_zona = []
        for a, p in fases:
            kios_n = sum(p["kioskos"].values())
            vis_d  = round(p["poblacion_hab"] * 0.008)
            cons_d = round(vis_d * kpis["consumiria_pct"] / 100)
            ing    = round(cons_d * gasto * 365 / 1000)
            ingresos_zona.append(ing)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(anios, ingresos_zona, color=ORANGE, alpha=0.85, width=0.5)
        for bar, val in zip(bars, ingresos_zona):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ingresos_zona)*0.01,
                    f"${val:,}k", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Ingresos potenciales (miles USD/año)", fontsize=9)
        ax.set_title("Proyección de ingresos totales de la zona — 2026–2036",
                     fontsize=11, fontweight="bold", color=BLUE)
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}k"))
        fig.tight_layout()
        charts["ing_zona"] = _save(fig)
    except Exception:
        charts["ing_zona"] = None

    # ── 9. Ingresos por kiosko ────────────────────────────────────────────
    try:
        gasto = kpis["gasto_promedio_usd"]
        fases = sorted(PLAN_FASES.items())
        anios = [str(a) for a, _ in fases]
        ing_kio = []
        for a, p in fases:
            kios_n = sum(p["kioskos"].values())
            vis_d  = round(p["poblacion_hab"] * 0.008)
            cons_d = round(vis_d * kpis["consumiria_pct"] / 100)
            ing    = round(cons_d * gasto * 365 / kios_n / 1000, 1) if kios_n else 0
            ing_kio.append(ing)
        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(anios, ing_kio, color=GREEN, alpha=0.85, width=0.5)
        for bar, val in zip(bars, ing_kio):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(ing_kio)*0.01,
                    f"${val:,}k", ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_ylabel("Ingresos por kiosko (miles USD/año)", fontsize=9)
        ax.set_title("Proyección de ingresos promedio por kiosko — 2026–2036",
                     fontsize=11, fontweight="bold", color=BLUE)
        ax.spines[["top","right"]].set_visible(False)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${int(x):,}k"))
        fig.tight_layout()
        charts["ing_kio"] = _save(fig)
    except Exception:
        charts["ing_kio"] = None

    # ── 10. Distribución de kioskos en el bulevar ────────────────────────
    try:
        from models.kiosko_model import GIRO_ROTATION, GIROS
        rotation = GIRO_ROTATION["comercial_alta_densidad"]
        n = 10
        giros = rotation[:n]
        colores_giro = {
            "Bebidas":                            "#2196F3",
            "Comida rápida":                      "#FF5722",
            "Helados y café":                     "#9C27B0",
            "Snacks saludables":                  "#4CAF50",
            "Artículos para mascotas y souvenirs":"#FF9800",
            "Deportivo":                          "#F44336",
        }
        colors = [colores_giro.get(g, BLUE) for g in giros]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        ax.axis("off")

        for i, (giro, col) in enumerate(zip(giros, colors)):
            # Rectángulo del kiosko
            rect = plt.Rectangle((i + 0.1, 0.15), 0.8, 0.55,
                                  color=col, alpha=0.88, zorder=3)
            ax.add_patch(rect)
            # Número
            ax.text(i + 0.5, 0.42, f"K{i+1}",
                    ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white", zorder=4)
            # Etiqueta giro (debajo)
            label = giro.replace(" y ", "\ny ").replace(" para ", "\npara ")
            ax.text(i + 0.5, 0.08, label,
                    ha="center", va="top", fontsize=6, color="#333", zorder=4)

        ax.set_title("Distribución comercial propuesta — Bulevar de las Canchas (10 kioskos, Fase 1)",
                     fontsize=10, fontweight="bold", color=BLUE, pad=6)
        ax.set_facecolor("#f0f0f0")
        fig.patch.set_facecolor("white")
        fig.tight_layout()
        charts["layout"] = _save(fig)
    except Exception:
        charts["layout"] = None

    # ── 11. Temas más solicitados (análisis de comentarios) ───────────────
    try:
        from analysis.sentiment import analyze_comments
        result_df = analyze_comments(enc)
        col_t = next((c for c in ["temas", "topic", "tema"] if c in result_df.columns), None)
        if result_df is not None and not result_df.empty and col_t:
            serie = result_df[col_t].dropna()
            # Si los valores son listas, explotar primero
            if serie.apply(lambda x: isinstance(x, list)).any():
                serie = serie.explode()
            tc = serie[serie != "Otro"].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(7, 4))
            bars = ax.barh(tc.index[::-1], tc.values[::-1], color=BLUE, alpha=0.85, height=0.55)
            ax.set_xlabel("Número de menciones", fontsize=9)
            ax.set_title("Temas más solicitados por los visitantes",
                         fontsize=11, fontweight="bold", color=BLUE)
            ax.spines[["top","right"]].set_visible(False)
            for bar, val in zip(bars, tc.values[::-1]):
                ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                        str(int(val)), va="center", fontsize=9)
            fig.tight_layout()
            charts["topics"] = _save(fig)
        else:
            charts["topics"] = None
    except Exception:
        charts["topics"] = None

    return charts


# ── generador principal ────────────────────────────────────────────────────────

def generate_word_report(enc, kpis, stats, fc, charts=None):
    """
    Genera el informe Word completo y devuelve bytes.

    Parámetros
    ----------
    enc    : DataFrame de encuesta limpia
    kpis   : dict de KPIs (de get_kpi_summary)
    stats  : dict de estadísticas de visitas
    fc     : resultado de forecast_kioskos()
    charts : dict nombre→bytes PNG (de build_charts). Si es None se intenta
             generar internamente (puede fallar en contexto Streamlit).
    """
    from models.kiosko_model import PLAN_FASES, PROYECCION_PARAMS

    if charts is None:
        charts = build_charts(enc, kpis, stats, fc)

    # ── variables ────────────────────────────────────────────────────────────
    n_enc    = kpis["n_encuestados"]
    vis_2025 = kpis["total_visitas_parque_2025"]
    vis_dia  = round(vis_2025 / 365)
    cons_pct = kpis["consumiria_pct"]
    cons_dia = round(vis_dia * cons_pct / 100)
    gasto    = kpis["gasto_promedio_usd"]
    calif    = kpis["calificacion_promedio"]
    aprueba  = kpis["aprueba_kiosko_pct"]
    mejora   = kpis["mejora_experiencia_pct"]
    edad     = kpis["edad_promedio"]
    pob_2026 = PROYECCION_PARAMS["poblacion_sector_hab"][2026]
    pob_2036 = PROYECCION_PARAMS["poblacion_sector_hab"][2036]

    # ── documento ────────────────────────────────────────────────────────────
    doc = Document()
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

    for txt, sz, bold in [
        ("Empresa Pública Metropolitana de Movilidad y Obras Públicas", 12, True),
        ("EPMMOP — Promoción de Servicios", 11, False),
    ]:
        p = doc.add_paragraph(txt)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.runs[0].bold = bold
        p.runs[0].font.size = Pt(sz)

    doc.add_paragraph()
    pf = doc.add_paragraph(f"Quito, {date.today().strftime('%B de %Y').capitalize()}")
    pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf.runs[0].font.size = Pt(11)
    doc.add_page_break()

    # ━━━ RESUMEN EJECUTIVO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "RESUMEN EJECUTIVO", 1)
    _p(doc, (
        f"El presente estudio evalúa la viabilidad de implementar kioskos comerciales en el "
        f"Parque Bicentenario de Quito, con base en una encuesta a {n_enc} visitantes y el "
        f"análisis de {vis_2025:,} registros de visita del año 2025. El {cons_pct}% de los "
        f"encuestados consumiría productos o servicios dentro del parque ({cons_dia} personas/día). "
        f"La zona de intervención propuesta es el Bulevar de las Canchas (160 m, hasta 50 kioskos), "
        f"con un plan de implementación gradual de 10 kioskos en 2026 hasta 25 en 2036. "
        f"El AIVA del sector Bicentenario se ubica en $267/m², con proyección hacia los "
        f"$1,895/m² del sector La Carolina."
    ))
    doc.add_paragraph()
    _h(doc, "Indicadores clave", 2)
    _kpi_table(doc, [
        ("Visitas registradas 2025",           f"{vis_2025:,}"),
        ("Visitantes diarios promedio",        f"{vis_dia:,}"),
        ("Encuestados",                        f"{n_enc}"),
        ("Consumidores potenciales por día",   f"{cons_dia} ({cons_pct}%)"),
        ("Gasto promedio dispuesto",           f"${gasto}"),
        ("Aprobación de kioskos",              f"{aprueba}%"),
        ("Mejora percibida en experiencia",    f"{mejora}%"),
        ("Calificación actual del parque",     f"{calif}/5"),
        ("Edad promedio del visitante",        f"{edad} años"),
        ("AIVA sector Bicentenario",           "$267/m²"),
        ("AIVA referencia La Carolina",        "$1,895/m²"),
        ("Kioskos propuestos inicio / meta",   "10 (2026) → 25 (2036)"),
    ])
    doc.add_page_break()

    # ━━━ 1. INTRODUCCIÓN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "1. INTRODUCCIÓN", 1)
    _h(doc, "1.1 Antecedentes", 2)
    _p(doc, (
        "El Parque Bicentenario ocupa los terrenos del antiguo Aeropuerto Mariscal Sucre de Quito. "
        "Con aproximadamente 50 hectáreas, es el parque urbano más grande de la ciudad y uno de "
        "los más significativos de la región andina. Administrado por la EPMMOP, recibe más de "
        f"{vis_2025:,} visitantes al año y es el principal espacio de esparcimiento, deporte y "
        "cultura del norte de Quito."
    ))
    _h(doc, "1.2 Problemática", 2)
    _p(doc, (
        "El parque carece de una oferta comercial estructurada. La ausencia de servicios de "
        "alimentación y productos básicos genera una demanda insatisfecha identificada y "
        "cuantificada en este estudio. Esta brecha representa una oportunidad para el municipio, "
        "los operadores locales y los propietarios de inmuebles del sector."
    ))
    _h(doc, "1.3 Justificación", 2)
    _p(doc, (
        "La implementación de kioskos modulares en el Bulevar de las Canchas permite una "
        "intervención reversible, escalable y coherente con los lineamientos de uso del espacio "
        "público del Municipio del Distrito Metropolitano de Quito, sustentada en evidencia "
        "directa de demanda ciudadana."
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
        "El estudio combina métodos cuantitativos y cualitativos para obtener una visión "
        "integral del mercado potencial, en dos etapas: encuesta de campo y análisis de "
        "registros históricos de visita."
    ))
    _h(doc, "3.1 Encuesta de campo", 2)
    _p(doc, (
        f"Se aplicaron {n_enc} encuestas estructuradas a visitantes del Parque Bicentenario "
        "mediante muestreo aleatorio sistemático, con 17 variables que cubren perfil "
        "sociodemográfico, patrones de visita, preferencias de consumo y percepción de servicios."
    ))
    _kpi_table(doc, [
        ("Tamaño de muestra",     f"{n_enc} encuestados"),
        ("Herramienta",           "Google Forms"),
        ("Tipo de muestreo",      "Aleatorio sistemático"),
        ("Variables analizadas",  "17 preguntas"),
        ("Análisis estadístico",  "Python — pandas, scipy, plotly"),
    ])
    _h(doc, "3.2 Análisis de registros de visita", 2)
    _p(doc, (
        f"Se analizaron {vis_2025:,} registros de visita al Parque Bicentenario del año 2025, "
        "proporcionados por la EPMMOP, para identificar patrones de afluencia, tendencias "
        "estacionales y la demanda diaria promedio."
    ))
    doc.add_page_break()

    # ━━━ 4. PERFIL SOCIODEMOGRÁFICO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "4. PERFIL SOCIODEMOGRÁFICO DEL VISITANTE", 1)
    _p(doc, (
        f"El visitante típico es residente del norte de Quito (78.2%), con edad promedio de "
        f"{edad} años. El domingo es el día de mayor afluencia (67.5%), con pico en el horario "
        "de 09:00–12:00 h en fines de semana y 04:00–09:00 h en días laborables. "
        "El grupo de visita predominante es la familia (45.6%) y el motivo principal es la recreación."
    ))
    _kpi_table(doc, [
        ("Visitantes totales 2025",             f"{vis_2025:,}"),
        ("Visitantes diarios promedio",         f"{vis_dia:,}"),
        ("Edad promedio",                       f"{edad} años"),
        ("Día más visitado",                    "Domingo (67.5%)"),
        ("Sector de residencia mayoritario",    "Norte de Quito (78.2%)"),
        ("Grupo de visita predominante",        "Familia (45.6%)"),
        ("Motivo de visita principal",          "Recreación (29.1%)"),
        ("Servicio complementario más pedido",  "Baños (76.2%)"),
    ])

    _insert_img(doc, charts.get("genero"),
                "Gráfico 1. Distribución por género de los visitantes encuestados", 4.5)
    _insert_img(doc, charts.get("edad"),
                "Gráfico 2. Distribución de visitantes por grupos de edad", 5.0)
    _insert_img(doc, charts.get("motivo"),
                "Gráfico 3. Motivo principal de visita al Parque Bicentenario", 4.5)
    doc.add_page_break()

    # ━━━ 5. ANÁLISIS DE DEMANDA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "5. ANÁLISIS DE DEMANDA", 1)

    _h(doc, "5.1 Afluencia al parque", 2)
    _p(doc, (
        f"El Parque Bicentenario registró {vis_2025:,} visitas durante 2025, con promedio de "
        f"{vis_dia:,} visitantes/día. La mayor afluencia se concentra los domingos. La tendencia "
        "muestra estacionalidad marcada en julio–agosto (vacaciones escolares) y en el primer "
        "trimestre del año."
    ))
    _insert_img(doc, charts.get("trafico"),
                "Gráfico 4. Tráfico de visitantes al Parque Bicentenario — 2025", 6.0)

    _h(doc, "5.2 Demanda potencial de servicios comerciales", 2)
    _p(doc, (
        f"El {cons_pct}% de los encuestados ({cons_dia} personas/día) afirmó que consumiría "
        f"productos o servicios si estuvieran disponibles. El gasto promedio dispuesto es "
        f"${gasto} por visita. El {aprueba}% aprueba la implementación de kioskos y el "
        f"{mejora}% considera que mejoraría su experiencia."
    ))
    _insert_img(doc, charts.get("demanda"),
                "Gráfico 5. Demanda potencial de consumidores vs. plan de kioskos", 6.0)

    _h(doc, "5.3 Productos y servicios más demandados", 2)
    _p(doc, (
        "Los snacks y bebidas encabezan la preferencia (38.0%), seguidos de bebidas calientes, "
        "comida rápida y helados. Entre los servicios complementarios, los baños son la "
        "prioridad más urgente (76.2%), seguida de iluminación y zonas de descanso."
    ))
    _insert_img(doc, charts.get("topics"),
                "Gráfico 6. Temas más solicitados por los visitantes (análisis de comentarios)", 6.0)
    doc.add_page_break()

    # ━━━ 6. MODELO COMERCIAL ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "6. MODELO COMERCIAL", 1)

    _h(doc, "6.1 Zona de intervención", 2)
    _p(doc, (
        "La zona seleccionada es el Bulevar de las Canchas (Zona 1), corredor peatonal de "
        "160 m que atraviesa el área de canchas deportivas. Esta ubicación maximiza la exposición "
        "a visitantes activos y aprovecha el flujo natural de tránsito peatonal."
    ))
    _kpi_table(doc, [
        ("Zona",                  "Zona 1 – Bulevar de las Canchas"),
        ("Longitud total",        "160 m lineales"),
        ("Número de secciones",   "4"),
        ("Capacidad máxima",      "50 kioskos"),
        ("Dimensión por kiosko",  "2.5 m × 2.0 m"),
        ("Ancho del pasillo",     "6.0 m"),
    ])

    # Imágenes del parque
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
        "Los giros se asignan por rotación garantizando diversidad de oferta y evitando "
        "competencia directa entre módulos adyacentes. La Fase 1 arranca con 10 kioskos "
        "distribuidos en 6 giros distintos:"
    ))

    from models.kiosko_model import GIROS, GIRO_ROTATION
    giro_rows = [
        (g, GIROS[g]["descripcion"])
        for g in GIROS
    ]
    tbl_g = doc.add_table(rows=len(giro_rows) + 1, cols=2)
    tbl_g.style = "Table Grid"
    for j, h in enumerate(["Giro comercial", "Productos / servicios"]):
        c = tbl_g.rows[0].cells[j]
        c.text = h
        c.paragraphs[0].runs[0].bold = True
        c.paragraphs[0].runs[0].font.size = Pt(10)
    for i, (giro, desc) in enumerate(giro_rows, 1):
        tbl_g.rows[i].cells[0].text = giro
        tbl_g.rows[i].cells[1].text = desc
        for j in range(2):
            tbl_g.rows[i].cells[j].paragraphs[0].runs[0].font.size = Pt(10)
    doc.add_paragraph()

    _p(doc, "Distribución de kioskos por rotación — Fase 1 (10 kioskos):", bold=True)
    rotation = GIRO_ROTATION["comercial_alta_densidad"]
    tbl_r = doc.add_table(rows=2, cols=10)
    tbl_r.style = "Table Grid"
    for i, giro in enumerate(rotation[:10]):
        tbl_r.rows[0].cells[i].text = f"K{i+1}"
        tbl_r.rows[0].cells[i].paragraphs[0].runs[0].bold = True
        tbl_r.rows[0].cells[i].paragraphs[0].runs[0].font.size = Pt(8)
        tbl_r.rows[1].cells[i].text = giro
        tbl_r.rows[1].cells[i].paragraphs[0].runs[0].font.size = Pt(8)
    doc.add_paragraph()

    _insert_img(doc, charts.get("layout"),
                "Figura 3. Distribución comercial propuesta — Bulevar de las Canchas (Fase 1, 10 kioskos)", 6.0)

    _h(doc, "6.3 Plan de implementación por fases", 2)
    _p(doc, (
        "El proyecto se despliega en 4 fases a 10 años, condicionado a hitos de desarrollo "
        "urbano (metro, densificación). Cada fase incrementa la oferta según el crecimiento "
        "de la demanda."
    ))
    fase_rows = []
    for anio, params in sorted(PLAN_FASES.items()):
        total_kio = sum(params["kioskos"].values())
        fase_rows.append((params["fase"], str(anio), params["hito"],
                          str(total_kio), f"{params['poblacion_hab']:,} hab."))

    tbl_f = doc.add_table(rows=len(fase_rows) + 1, cols=5)
    tbl_f.style = "Table Grid"
    for j, h in enumerate(["Fase", "Año", "Hito", "Kioskos", "Población"]):
        c = tbl_f.rows[0].cells[j]
        c.text = h
        c.paragraphs[0].runs[0].bold = True
        c.paragraphs[0].runs[0].font.size = Pt(9)
    for i, row in enumerate(fase_rows, 1):
        for j, val in enumerate(row):
            tbl_f.rows[i].cells[j].text = val
            tbl_f.rows[i].cells[j].paragraphs[0].runs[0].font.size = Pt(9)
    doc.add_paragraph()

    _insert_img(doc, charts.get("kioskos"),
                "Gráfico 7. Proyección de kioskos 2026–2036", 6.0)
    doc.add_page_break()

    # ━━━ 7. PROYECCIÓN FINANCIERA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "7. PROYECCIÓN FINANCIERA", 1)
    _p(doc, (
        f"La proyección financiera se basa en el crecimiento esperado de la población del sector "
        f"(de {pob_2026:,} en 2026 a {pob_2036:,} en 2036), el aumento progresivo de visitantes "
        "y el incremento gradual de kioskos activos, con escenarios conservadores de ocupación."
    ))

    _h(doc, "7.1 Proyección de visitantes y demanda", 2)
    _insert_img(doc, charts.get("visitantes_vs"),
                "Gráfico 8. Visitantes del parque vs. consumidores potenciales por fase", 6.0)

    _h(doc, "7.2 Ingresos proyectados", 2)
    _p(doc, (
        "Se presentan dos perspectivas: ingresos consolidados por zona e ingresos por kiosko "
        "individual, para evaluar la rentabilidad del proyecto global y de cada operador."
    ))
    _insert_img(doc, charts.get("ing_zona"),
                "Gráfico 9. Proyección de ingresos totales de la zona 2026–2036", 6.0)
    _insert_img(doc, charts.get("ing_kio"),
                "Gráfico 10. Proyección de ingresos promedio por kiosko 2026–2036", 6.0)

    _h(doc, "7.3 Impacto en valorización del suelo (AIVA)", 2)
    _p(doc, (
        "El AIVA del sector Bicentenario se ubica en $267/m², frente a los $1,895/m² de La "
        "Carolina. La consolidación del parque como polo comercial proyecta un crecimiento "
        "sostenido del AIVA, beneficiando directamente a propietarios de inmuebles del sector."
    ))
    _kpi_table(doc, [
        ("AIVA actual Bicentenario",    "$267/m²"),
        ("AIVA referencia La Carolina", "$1,895/m²"),
        ("Potencial de crecimiento",    "~7x respecto al valor actual"),
    ])
    doc.add_page_break()

    # ━━━ 8. CONCLUSIONES Y RECOMENDACIONES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "8. CONCLUSIONES Y RECOMENDACIONES", 1)
    _h(doc, "8.1 Conclusiones", 2)
    _bullet(doc, [
        f"Existe demanda potencial verificada: el {cons_pct}% de visitantes ({cons_dia}/día) consumiría en el parque.",
        f"El parque recibe {vis_2025:,} visitas al año ({vis_dia:,}/día), con pico los domingos.",
        "Snacks, bebidas y comida rápida encabezan la preferencia; baños son el servicio más urgente (76.2%).",
        "El modelo de kioskos por fases (10 → 25, 2026–2036) es técnicamente viable y alineado al crecimiento urbano.",
        f"La ciudadanía tiene actitud favorable: {aprueba}% aprueba los kioskos, {mejora}% cree que mejorarían su experiencia.",
        "El AIVA del sector ($267/m²) tiene alto potencial de valorización hacia los niveles de La Carolina ($1,895/m²).",
        "La apertura del Metro (2029) transformará el corredor en eje de movilidad urbana y multiplicará la afluencia.",
    ])
    _h(doc, "8.2 Recomendaciones", 2)
    _bullet(doc, [
        "Iniciar la Fase 1 en 2026 con 10 kioskos diversificados para validar el modelo antes de escalar.",
        "Priorizar instalación de baños públicos como servicio complementario inmediato (76.2% de demanda).",
        "Establecer contratos de concesión con revisión anual para ajustar giros y tarifas según demanda real.",
        "Implementar monitoreo de ventas para optimizar el mix comercial con datos en tiempo real.",
        "Comunicar el impacto en el AIVA del sector a propietarios e inversores para generar alianzas ciudadanas.",
        "Evaluar extensión del modelo a otras zonas del parque en la Fase 4 (2036) si resultados superan proyecciones.",
    ])
    doc.add_page_break()

    # ━━━ ANEXOS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    _h(doc, "ANEXOS", 1)
    _h(doc, "Anexo A — Instrumento de encuesta", 2)
    _p(doc, f"Variables del formulario aplicado ({n_enc} respuestas):")
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
        "¿Cuánto estaría dispuesto a gastar?",
        "¿Considera adecuada la implementación de kioskos comerciales?",
        "¿En qué zonas accedería con mayor facilidad para adquirir productos?",
        "¿Considera que los kioskos mejorarían su experiencia?",
        "¿Cómo califica actualmente la oferta de servicios dentro del parque?",
        "¿Qué servicios complementarios considera prioritarios?",
        "Comentarios y sugerencias",
    ])
    _h(doc, "Anexo B — Descripción de fases del plan", 2)
    for anio, params in sorted(PLAN_FASES.items()):
        _p(doc, params["fase"], bold=True, size=10)
        _p(doc, params["contexto"], size=10)
        doc.add_paragraph()

    # ── guardar ───────────────────────────────────────────────────────────────
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()
