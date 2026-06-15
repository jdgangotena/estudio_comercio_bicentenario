"""
Carga de datos desde los archivos Excel del proyecto.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "src" / "data"
ENCUESTA_FILE = DATA_DIR / "Encuesta kioskos bicentenario.xlsx"
ESTADISTICAS_FILE = DATA_DIR / "ESTADISTICAS PARQUE BICENTENARIO.xlsx"

MONTHS_ES = ["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO",
             "JULIO", "AGOSTO", "SEPTIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"]

MONTH_MAP = {m: i + 1 for i, m in enumerate(MONTHS_ES)}

COLUMN_ALIASES = {
    "Marca temporal": "timestamp",
    "Edad": "edad",
    "Género ": "genero",
    "¿En qué sector reside? ": "sector_residencia",
    "¿Con quién visita generalmente el Parque Bicentenario? ": "acompanante",
    "¿Con qué frecuencia visita el Parque Bicentenario? ": "frecuencia_visita",
    "¿Qué días usualmente  visita  el parque?  ": "dias_visita",
    "¿En qué horario usualmente visita el parque? ": "horario_visita",
    "¿Cuál es el principal motivo de su visita? ": "motivo_visita",
    "¿Consumiría productos o servicios dentro del parque? ": "consumiria",
    "¿Qué productos o servicios consumiría dentro del parque? ": "productos_interes",
    "¿Cuánto estaría dispuesto a gastar en la compra de estos productos o servicios? ": "gasto_dispuesto",
    "¿Considera adecuada la implementación de kioskos comerciales dentro del parque? ": "kiosko_adecuado",
    "¿En qué zonas accedería con mayor facilidad para compra o adquisición de productos o servicios (kioskos)? ": "zona_acceso",
    "¿Considera que la implementación de nuevos espacios comerciales (kioskos) mejoraría su experiencia en el parque? ": "kiosko_mejora_experiencia",
    "¿Cómo califica actualmente la oferta de servicios dentro del parque? ": "calificacion_oferta",
    "¿Qué servicios complementarios considera prioritarios para implementar en el parque?\n(Puede seleccionar varias opciones) ": "servicios_prioritarios",
    "Comentarios y Sugerencias ": "comentarios",
}


def load_encuesta() -> pd.DataFrame:
    """Carga la encuesta y renombra columnas a nombres cortos."""
    df = pd.read_excel(ENCUESTA_FILE)
    # Renombrar columnas que coincidan exactamente
    rename_map = {}
    for orig, alias in COLUMN_ALIASES.items():
        if orig in df.columns:
            rename_map[orig] = alias
    df = df.rename(columns=rename_map)
    return df


def _parse_stats_section(raw: pd.DataFrame, start_row: int) -> pd.DataFrame:
    """
    Extrae los datos mensuales de una sección del archivo de estadísticas.
    Estructura esperada: pares (fila_encabezado_mes, fila_datos).
    Retorna un DataFrame con columnas: mes, num_mes, lunes..domingo, total.
    """
    rows = []
    i = start_row + 2  # saltar nombre y ubicación
    while i < len(raw) and i < start_row + 27:
        month_val = raw.iloc[i, 3]
        if pd.notna(month_val) and str(month_val).strip().upper() in MONTH_MAP:
            mes = str(month_val).strip().upper()
            data_row = raw.iloc[i + 1]
            vals = {
                "mes": mes,
                "num_mes": MONTH_MAP[mes],
                "lunes": _safe_num(data_row.iloc[4]),
                "martes": _safe_num(data_row.iloc[5]),
                "miercoles": _safe_num(data_row.iloc[6]),
                "jueves": _safe_num(data_row.iloc[7]),
                "viernes": _safe_num(data_row.iloc[8]),
                "sabado": _safe_num(data_row.iloc[9]),
                "domingo": _safe_num(data_row.iloc[10]),
                "total": _safe_num(data_row.iloc[11]),
            }
            rows.append(vals)
            i += 2
        else:
            i += 1
    return pd.DataFrame(rows)


def _safe_num(val):
    try:
        v = float(val)
        return v if not np.isnan(v) else 0
    except Exception:
        return 0


def load_estadisticas() -> dict[str, pd.DataFrame]:
    """
    Parsea el archivo de estadísticas y retorna un dict con DataFrames
    mensuales por cada zona/actividad.
    """
    raw = pd.read_excel(ESTADISTICAS_FILE, header=None)

    sections = {}
    for i, row in raw.iterrows():
        val = row[1]
        if pd.notna(val) and str(val).strip() and "PARQUE" not in str(val):
            # Normalizar nombre: strip y colapsar espacios internos
            name = " ".join(str(val).split())
            df_section = _parse_stats_section(raw, i)
            if not df_section.empty:
                sections[name] = df_section

    return sections


def load_all() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Carga y retorna ambos conjuntos de datos."""
    encuesta = load_encuesta()
    estadisticas = load_estadisticas()
    return encuesta, estadisticas
