"""
Limpieza y transformación del dataset de encuestas.
"""

import pandas as pd
import numpy as np
import re

MULTI_SELECT_COLS = ["dias_visita", "servicios_prioritarios"]

GASTO_ORDER = ["Menos de $2", "Entre $2 y $5", "Entre $5 y $10", "Entre $10 y $20", "Más de $20"]
FRECUENCIA_ORDER = ["Diariamente", "Varias veces a la semana", "Una vez a la semana",
                    "Varias veces al mes", "Ocasionalmente"]
CALIFICACION_LABELS = {1: "Muy mala", 2: "Mala", 3: "Regular", 4: "Buena", 5: "Muy buena"}

EDAD_BINS = [0, 17, 25, 35, 45, 55, 65, 120]
EDAD_LABELS = ["<18", "18-25", "26-35", "36-45", "46-55", "56-65", "65+"]

HORARIO_BINS = [5, 7, 9, 11, 13, 15, 17, 19, 21, 24]
HORARIO_LABELS = ["06-07h", "08-09h", "10-11h", "12-13h", "14-15h", "16-17h", "18-19h", "20-21h", "22h+"]


def _normalize_str(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip())


def clean_encuesta(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza y normalización al DataFrame de encuestas."""
    df = df.copy()

    # Normalizar strings
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].apply(lambda x: _normalize_str(x) if pd.notna(x) else x)

    # Normalizar género
    if "genero" in df.columns:
        df["genero"] = df["genero"].str.strip().str.title()

    # Normalizar columna consumiria
    if "consumiria" in df.columns:
        df["consumiria"] = df["consumiria"].str.strip().str.capitalize()
        df["consumiria"] = df["consumiria"].replace({"Si": "Sí"})

    # Convertir calificacion a int
    if "calificacion_oferta" in df.columns:
        df["calificacion_oferta"] = pd.to_numeric(df["calificacion_oferta"], errors="coerce")

    # Crear rangos de edad
    if "edad" in df.columns:
        df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
        df["rango_edad"] = pd.cut(df["edad"], bins=EDAD_BINS, labels=EDAD_LABELS, right=True)

    # Crear rangos de horario
    if "horario_visita" in df.columns:
        df["horario_visita"] = pd.to_numeric(df["horario_visita"], errors="coerce")
        df["rango_horario"] = pd.cut(
            df["horario_visita"], bins=HORARIO_BINS, labels=HORARIO_LABELS, right=False
        )

    # Limpiar comentarios nulos o triviales
    if "comentarios" in df.columns:
        mask_trivial = df["comentarios"].str.lower().isin(
            ["ninguno", "ninguna", "n/a", "na", "no", "nada", "sin comentarios", "nan"]
        )
        df.loc[mask_trivial, "comentarios"] = np.nan

    return df


def explode_multiselect(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Para una columna de selección múltiple (valores separados por comas),
    retorna una Serie con todos los valores individuales expandidos.
    """
    exploded = (
        df[column]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .replace("", np.nan)
        .dropna()
    )
    return exploded


def count_multiselect(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Cuenta frecuencias de cada opción en una columna de selección múltiple.
    Las respuestas pueden superar el 100% al ser multi-opción.
    """
    total_respondents = df[column].dropna().shape[0]
    exploded = explode_multiselect(df, column)
    counts = exploded.value_counts().reset_index()
    counts.columns = ["opcion", "frecuencia"]
    counts["pct_respuestas"] = (counts["frecuencia"] / total_respondents * 100).round(1)
    return counts


def get_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo de limpieza."""
    return clean_encuesta(df)
