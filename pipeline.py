"""
Pipeline orquestador: carga, limpia y prepara todos los datos para la app.
Usa caché de Streamlit para no recargar en cada interacción.
"""

import sys
from pathlib import Path

# Asegurar que el directorio raíz esté en el path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "analysis"))
sys.path.insert(0, str(Path(__file__).parent / "models"))

import streamlit as st
import pandas as pd
from data_loader import load_all
from data_cleaner import get_clean_data


@st.cache_data(show_spinner="Cargando y procesando datos...")
def run_pipeline() -> dict:
    """
    Ejecuta el pipeline completo y retorna un dict con todos los artefactos.
    """
    encuesta_raw, estadisticas = load_all()
    encuesta = get_clean_data(encuesta_raw)

    return {
        "encuesta": encuesta,
        "estadisticas": estadisticas,
        "n_encuestados": len(encuesta),
        "n_con_comentarios": int(encuesta["comentarios"].notna().sum()),
    }


def get_kpi_summary(ctx: dict) -> dict:
    """KPIs principales para el dashboard de resumen."""
    enc = ctx["encuesta"]
    total = len(enc)

    consumiria_pct = 0
    if "consumiria" in enc.columns:
        consumiria_pct = round(
            enc["consumiria"].str.strip().str.lower().isin(["sí", "si"]).sum() / total * 100, 1
        )

    aprueba_kiosko_pct = 0
    if "kiosko_adecuado" in enc.columns:
        aprueba_kiosko_pct = round(
            enc["kiosko_adecuado"].str.strip().str.lower().isin(["sí", "si"]).sum() / total * 100, 1
        )

    mejora_exp_pct = 0
    if "kiosko_mejora_experiencia" in enc.columns:
        mejora_exp_pct = round(
            enc["kiosko_mejora_experiencia"].str.strip().str.lower().isin(["sí", "si"]).sum() / total * 100, 1
        )

    calif_prom = 0
    if "calificacion_oferta" in enc.columns:
        calif_prom = round(enc["calificacion_oferta"].dropna().mean(), 2)

    edad_prom = 0
    if "edad" in enc.columns:
        edad_prom = round(enc["edad"].dropna().mean(), 1)

    gasto_map = {
        "Menos de $2": 1.5,
        "Entre $2 y $5": 3.5,
        "Entre $5 y $10": 7.5,
        "Entre $10 y $20": 15.0,
        "Más de $20": 22.0,
    }
    gasto_prom = 0
    if "gasto_dispuesto" in enc.columns:
        gastos = enc["gasto_dispuesto"].map(gasto_map).dropna()
        gasto_prom = round(gastos.mean(), 2) if len(gastos) > 0 else 0

    # Total visitas anuales parque
    total_visitas = sum(
        df["total"].sum()
        for df in ctx["estadisticas"].values()
        if "total" in df.columns
    )

    return {
        "n_encuestados": total,
        "consumiria_pct": consumiria_pct,
        "aprueba_kiosko_pct": aprueba_kiosko_pct,
        "mejora_experiencia_pct": mejora_exp_pct,
        "calificacion_promedio": calif_prom,
        "edad_promedio": edad_prom,
        "gasto_promedio_usd": gasto_prom,
        "total_visitas_parque_2025": int(total_visitas),
    }
