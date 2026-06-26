"""
Modelo de pronóstico y planificación espacial de kioskos comerciales.

Metodología combinada:
  A) DEMANDA: visitantes × tasa de consumo ÷ capacidad por kiosko
  B) ESPACIAL: dimensiones físicas × estándares de circulación peatonal
  C) RESULTADO: reconciliación demanda–espacio + tipo de kiosko recomendado
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PALETTE = px.colors.qualitative.Safe

# ── Dimensiones físicas de cada zona ──────────────────────────────────────────
ZONE_DIMENSIONS = {
    "Zona 1 – Bulevar de las Canchas": {
        "longitud_m": 2000,
        "ancho_m": 12,        # ancho del bulevar peatonal (la calzada vehicular ocupa los 20m totales)
        "lados_disponibles": 1,           # kioscos en un solo lado del bulevar peatonal
        "tipo_zona": "Bulevar peatonal",
        "color": "#2980b9",
    },
}

# ── Tipos de kiosko disponibles ────────────────────────────────────────────────
KIOSKO_TIPOS = {
    "3×2 m  (6 m²) – Pequeño": {
        "ancho_m": 3,       # dimensión paralela al camino (a lo largo)
        "fondo_m": 2,       # dimensión perpendicular al camino (hacia adentro)
        "area_m2": 6,
        "categoria": "Pequeño",
        "descripcion": "Ideal para servicio rápido: bebidas, snacks, helados. "
                       "Perfecto para zonas deportivas y camineras estrechas.",
        "usos_recomendados": "Bebidas hidratantes, snacks, helados, café rápido",
        "color": "#3498db",
    },
    "3×3 m  (9 m²) – Mediano": {
        "ancho_m": 3,
        "fondo_m": 3,
        "area_m2": 9,
        "categoria": "Mediano",
        "descripcion": "Versátil para gastronomía ligera y productos saludables. "
                       "Equilibrio entre capacidad y huella en el espacio público.",
        "usos_recomendados": "Comida rápida saludable, jugos naturales, sánduches, snacks",
        "color": "#27ae60",
    },
    "6×4 m  (24 m²) – Grande": {
        "ancho_m": 6,
        "fondo_m": 4,
        "area_m2": 24,
        "categoria": "Grande",
        "descripcion": "Para oferta gastronómica amplia, con área de preparación y vitrina. "
                       "Requiere espacio suficiente; solo viable en zona de bulevar.",
        "usos_recomendados": "Restaurante de parque, cafetería, artesanías, productos locales",
        "color": "#e74c3c",
    },
}

# ── Estándares de circulación peatonal (INEN 2 314, espacios públicos Ecuador) ─
SPATIAL_PARAMS = {
    # Corredor peatonal mínimo libre de obstáculos
    "circulacion_minima_m": 3.0,       # por lado en zonas estrechas
    "circulacion_bulevar_central_m": 6.0,  # corredor central en bulevares ≥15m
    # Área de servicio frente al kiosko (clientes en fila/espera)
    "servicio_frontal_m": 1.5,
    # Separación entre kioskos contiguos (acceso lateral, emergencias)
    "separacion_entre_kioskos_m": 5.0,
    # Separación mínima entre grupos/clusters de kioskos
    "separacion_entre_grupos_m": 30.0,
    # Margen lateral mínimo desde borde de zona
    "margen_lateral_m": 0.5,
}

# ── Parámetros de distancia comercial (viabilidad de mercado) ─────────────────
COMMERCIAL_PARAMS = {
    # Distancia MÍNIMA entre kioskos del MISMO giro (para no canibalizarse)
    "distancia_min_mismo_giro_m": {
        "comercial_alta_densidad": 250,   # bulevar largo: cada kiosko igual necesita su territorio
        "eventos_espectaculos": 50,        # zona corta: imposible separar mucho
        "deportivo": 120,                  # caminera: usuarios pasan en movimiento
    },
    # Distancia MÍNIMA entre kioskos de DISTINTO giro (complementariedad)
    "distancia_min_dif_giro_m": {
        "comercial_alta_densidad": 50,
        "eventos_espectaculos": 15,
        "deportivo": 25,
    },
    # Radio de captación por kiosko (cuánto metro a cada lado atrae clientes)
    "radio_captacion_m": {
        "comercial_alta_densidad": 150,
        "eventos_espectaculos": 40,
        "deportivo": 80,
    },
}

# ── Tipos de giro (rubro comercial) ──────────────────────────────────────────
GIROS = {
    "Bebidas":              {"icono": "🥤", "color": "#2980b9",
                             "descripcion": "Agua, jugos naturales, bebidas hidratantes, batidos"},
    "Comida rápida":        {"icono": "🍔", "color": "#e67e22",
                             "descripcion": "Sánduches, almuerzos ejecutivos, wraps, ensaladas"},
    "Snacks saludables":    {"icono": "🥗", "color": "#27ae60",
                             "descripcion": "Frutas, yogur, granola, snacks naturales, barras energéticas"},
    "Helados y café":       {"icono": "🍦", "color": "#8e44ad",
                             "descripcion": "Helados artesanales, café, infusiones, postres ligeros"},
    "Artesanías":           {"icono": "🎨", "color": "#c0392b",
                             "descripcion": "Artesanías locales, souvenirs del parque, productos culturales"},
    "Deportivo":            {"icono": "🏃", "color": "#16a085",
                             "descripcion": "Accesorios deportivos, protector solar, kits de hidratación"},
}

# Relación entre cada giro y las opciones de la encuesta
# ("¿Qué productos o servicios consumiría dentro del parque?")
# que lo respaldan — permite sustentar la elección de giros con datos reales.
GIRO_PRODUCTO_MAP = {
    "Bebidas":           ["Snacks y bebidas"],
    "Comida rápida":     ["Comida rápida"],
    "Snacks saludables": ["Productos saludables"],
    "Helados y café":    ["Heladería", "Cafeterías"],
    "Artesanías":        ["Artesanías"],
    "Deportivo":         ["Servicios deportivos", "Alquiler de bicicletas/scooters"],
}

# Secuencia de giros recomendada por tipo de zona (se repite cíclicamente)
GIRO_ROTATION = {
    "comercial_alta_densidad": [
        "Bebidas", "Comida rápida", "Helados y café",
        "Snacks saludables", "Artesanías",
    ],
    "eventos_espectaculos": [
        "Bebidas", "Comida rápida", "Deportivo", "Snacks saludables",
    ],
    "deportivo": [
        "Bebidas", "Snacks saludables", "Deportivo",
    ],
}

# ── Proyección demográfica y de tráfico 2025–2035 ─────────────────────────────
PROYECCION_PARAMS = {
    # Crecimiento poblacional del sector Parque Bicentenario
    # Fuente: datos oficiales 142,034 hab. actuales → 304,570 hab. proyectados.
    # Impulsado por: leyes de densificación (edificabilidad 6→40 pisos),
    # 2 nuevas estaciones de metro (Bicentenario + Andalucía) y rascacielos aprobados.
    "poblacion_sector_hab": {
        2026: 142_034,
        2029: 185_000,
        2033: 248_000,
        2036: 304_570,
    },
    # Metro Quito: 2 nuevas estaciones en el área del parque
    # · Bicentenario: dentro del eje del bulevar
    # · Andalucía: 1-2 cuadras al norte, dentro de la longitud del parque (~2029)
    # Ambas amplían el radio de influencia y el flujo peatonal diario.
    "metro_pasajeros_dia": {
        2026: 0,        # estaciones Bicentenario y Andalucía aún en construcción
        2029: 23_000,   # apertura de ambas: ~15k Bicentenario + ~8k Andalucía
        2033: 32_000,
        2036: 40_000,   # ambas maduras + crecimiento poblacional
    },
    # Tráfico peatonal diario proyectado EN EL BULEVAR por categoría
    # (desglosado para transparencia del modelo)
    "trafico_boulevard_dia": {
        2026: {
            "Deportistas / Salud":      400,
            "Familias / Paseos":        250,
            "Residentes del sector":  2_000,   # 142k hab × ~1.4% usa el bulevar a diario
            "Metro (El Labrador)":      300,
        },
        2029: {
            "Deportistas / Salud":      700,
            "Familias / Paseos":        500,
            "Residentes del sector":  6_000,   # ~185k hab × ~3.2%
            "Metro (Bicentenario)":  15_000,
            "Metro (Andalucía)":      8_000,
        },
        2033: {
            "Deportistas / Salud":    1_200,
            "Familias / Paseos":      1_500,
            "Residentes del sector": 15_000,   # ~248k × ~6%
            "Metro (Bicentenario)":  20_000,
            "Metro (Andalucía)":     12_000,
        },
        2036: {
            "Deportistas / Salud":    2_000,
            "Familias / Paseos":      3_000,
            "Residentes del sector": 25_000,   # ~305k × ~8%
            "Metro (Bicentenario)":  25_000,
            "Metro (Andalucía)":     15_000,
        },
    },
    # Tasa de conversión a compradores por perfil de usuario
    "tasa_conversion_por_categoria": {
        "Deportistas / Salud":   0.45,   # alta: encuesta 52%; hidratación y snacks
        "Familias / Paseos":     0.50,   # alta: paseos de ocio con consumo
        "Residentes del sector": 0.03,   # baja: van a trabajo/metro, paso rápido
        "Metro (El Labrador)":   0.02,
        "Metro (Bicentenario)":  0.02,
        "Metro (Andalucía)":     0.02,
    },
    # Estándar urbano de densidad para boulevard peatonal comercial
    "metros_entre_kioskos_bulevar": 200,   # 1 kiosko c/200m → 10 en 2,000m
    "metros_entre_kioskos_deportivo": 65,
    "metros_entre_kioskos_eventos": 50,
}

# Plan de implementación por fases (2026–2036)
PLAN_FASES = {
    2026: {
        "fase": "Fase 1 – Implementación inicial",
        "contexto": (
            "Apertura del proyecto. Bulevar activo con 142,034 hab. en el sector. "
            "10 kioskos en el Bulevar de las Canchas (1 c/200m) cubren la demanda "
            "actual y establecen el corredor comercial del parque."
        ),
        "poblacion_hab": 142_034,
        "hito": "🟢 Arranque del proyecto",
        "kioskos": {
            "Zona 1 – Bulevar de las Canchas": 10,
        },
    },
    2029: {
        "fase": "Fase 2 – Apertura metro Bicentenario + Andalucía",
        "contexto": (
            "Apertura de las 2 nuevas estaciones de metro. El flujo de ~23,000 usuarios "
            "metro/día transforma el bulevar en corredor de movilidad urbana. "
            "Se amplía a 16 kioskos (1 c/125m)."
        ),
        "poblacion_hab": 185_000,
        "hito": "🚇 Metro: estaciones Bicentenario + Andalucía",
        "kioskos": {
            "Zona 1 – Bulevar de las Canchas": 16,
        },
    },
    2033: {
        "fase": "Fase 3 – Consolidación del nuevo polo urbano",
        "contexto": (
            "Rascacielos en funcionamiento. Edificabilidad hasta 40 pisos genera alta "
            "densidad residencial y oficinas. ~248,000 hab. 20 kioskos en el bulevar "
            "(1 c/100m). Demanda supera ampliamente la oferta planificada."
        ),
        "poblacion_hab": 248_000,
        "hito": "🏙️ Rascacielos y densificación urbana",
        "kioskos": {
            "Zona 1 – Bulevar de las Canchas": 20,
        },
    },
    2036: {
        "fase": "Fase 4 – Nuevo hipercentro de Quito",
        "contexto": (
            "304,570 hab. en el sector. Metro maduro con 40,000 usuarios/día combinados "
            "(Bicentenario + Andalucía). Bulevar de las Canchas como eje comercial maduro en La "
            "Carolina. 25 kioskos (1 c/80m) – máxima densidad comercial sustentable."
        ),
        "poblacion_hab": 304_570,
        "hito": "🏆 Nuevo hipercentro de Quito",
        "kioskos": {
            "Zona 1 – Bulevar de las Canchas": 25,
        },
    },
}

# ── Factor de visitantes no contabilizados ─────────────────────────────────────
FACTOR_VISITANTES_NO_CONTABILIZADOS = 2.5

ZONAS = {
    "Zona 1 – Bulevar de las Canchas": {
        "descripcion": "Bulevar peatonal de las Canchas del Parque Bicentenario. Zona de paseo familiar, "
                       "recreación y deporte. Concentra toda la demanda del parque: familias, "
                       "deportistas, usuarios de canchas/pistas y visitantes de actividades "
                       "especiales. Eje comercial principal del Parque Bicentenario.",
        "fuentes_estadisticas": [
            "JUEGOS RECREATIVOS",
            "ZONA CANINA",
            "2025 MINI CITI JUEGOS INCLUSIVOS",
            "2025 JUEGOS INCLUSIVOS",
            "2025 COMPETENCIAS DEPORTIVAS 3 CARRERAS AL MES",
            "2025 BAILOTERAPIA",
            "USO DE LA PISTA DE CICLISMO",
            "2025 PISTA MULTIUSO",
        ],
        "factor_captacion": 0.65,
        "tipo": "comercial_alta_densidad",
        "color": "#2980b9",
    },
}

KIOSKO_PARAMS = {
    "min_atendidos_hora": 10,
    "max_atendidos_hora": 15,
    "horas_semana": 7,
    "horas_finde": 10,
    "dias_semana_por_semana": 5,
    "dias_finde_por_semana": 2,
    "semanas_anio": 52,
    "factor_peak": 0.30,
    "ocupacion_objetivo": 0.60,
}


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS ESPACIAL
# ══════════════════════════════════════════════════════════════════════════════

def _ancho_util_por_kiosko(zona_name: str, kiosko_key: str) -> dict:
    """
    Determina si un tipo de kiosko CABE en la zona considerando circulación.
    Retorna dict con análisis de viabilidad.
    """
    dim = ZONE_DIMENSIONS[zona_name]
    ksk = KIOSKO_TIPOS[kiosko_key]
    sp = SPATIAL_PARAMS
    ancho = dim["ancho_m"]
    lados = dim["lados_disponibles"]
    fondo = ksk["fondo_m"]

    if lados == 2:
        # Bulevar: kioscos en ambos lados + corredor central
        # Cada lado: margen + kiosko + servicio
        espacio_cada_lado = (ancho - sp["circulacion_bulevar_central_m"]) / 2
        espacio_util_lado = espacio_cada_lado - sp["margen_lateral_m"]
        necesario = fondo + sp["servicio_frontal_m"]
        cabe_un_lado = necesario <= espacio_util_lado
        # ¿Caben en AMBOS lados simultáneamente?
        total_necesario = (necesario * 2) + sp["circulacion_bulevar_central_m"]
        cabe = total_necesario <= ancho
        espacio_libre = ancho - total_necesario
    else:
        # Pasaje/caminera: kioscos en UN solo lado + corredor peatonal
        necesario = fondo + sp["servicio_frontal_m"]
        espacio_libre = ancho - sp["circulacion_minima_m"] - necesario
        cabe = espacio_libre >= -0.3   # tolerancia de 30cm
        cabe_un_lado = cabe

    return {
        "cabe": cabe,
        "espacio_libre_m": round(espacio_libre, 2),
        "lados_posibles": (2 if (lados == 2 and cabe) else (1 if cabe_un_lado else 0)),
        "observacion": _observacion_cabe(cabe, espacio_libre, lados),
    }


def _observacion_cabe(cabe: bool, libre: float, lados: int) -> str:
    if not cabe:
        return "❌ No cabe – espacio insuficiente"
    if libre < 0.5:
        return "⚠️ Cabe con ajuste – espacio muy justo"
    if lados == 2:
        return f"✅ Cabe en ambos lados ({libre:.1f}m libre por lado)"
    return f"✅ Cabe cómodamente ({libre:.1f}m libre)"


def _max_kioskos_por_longitud(zona_name: str, kiosko_key: str) -> dict:
    """
    Calcula cuántos kioskos caben físicamente a lo largo de la zona.
    """
    dim = ZONE_DIMENSIONS[zona_name]
    ksk = KIOSKO_TIPOS[kiosko_key]
    sp = SPATIAL_PARAMS
    L = dim["longitud_m"]
    ancho_ksk = ksk["ancho_m"]
    sep = sp["separacion_entre_kioskos_m"]

    # Máximo absoluto (sin respetar clusters)
    max_absoluto = max(1, int(L / (ancho_ksk + sep)))

    # Máximo práctico: clusters de 2-4 kioskos separados 30m
    # Tamaño cluster: 3 kioskos × ancho + 2 separaciones internas
    cluster_longitud = 3 * ancho_ksk + 2 * sep
    n_clusters = max(1, int(L / (cluster_longitud + sp["separacion_entre_grupos_m"])))
    max_practico = n_clusters * 3   # 3 kioskos por cluster

    lados = _ancho_util_por_kiosko(zona_name, kiosko_key)["lados_posibles"]
    max_practico_total = max_practico * lados

    return {
        "max_absoluto_un_lado": max_absoluto,
        "max_practico_un_lado": max(1, n_clusters * 3),
        "n_clusters": n_clusters,
        "max_practico_total": max(1, max_practico_total),
        "longitud_m": L,
    }


def spatial_analysis_all() -> pd.DataFrame:
    """
    Tabla completa de análisis espacial: zona × tipo de kiosko.
    """
    rows = []
    for zona_name in ZONE_DIMENSIONS:
        dim = ZONE_DIMENSIONS[zona_name]
        for ksk_key, ksk in KIOSKO_TIPOS.items():
            fit = _ancho_util_por_kiosko(zona_name, ksk_key)
            cap = _max_kioskos_por_longitud(zona_name, ksk_key)
            rows.append({
                "Zona": zona_name.split("–")[1].strip() if "–" in zona_name else zona_name,
                "Tipo de kiosko": ksk_key,
                "Área (m²)": ksk["area_m2"],
                f"Ancho zona ({dim['ancho_m']}m)": fit["observacion"],
                "Máx. físico (práctico)": cap["max_practico_total"],
                "Lados posibles": cap["max_practico_total"] // max(1, cap["max_practico_un_lado"]),
                "Cabe": fit["cabe"],
            })
    return pd.DataFrame(rows)


def recommend_kiosko_type(zona_name: str) -> str:
    """
    Recomienda el tipo de kiosko más adecuado para cada zona
    basado en espacio disponible y perfil de usuario.
    """
    dim = ZONE_DIMENSIONS[zona_name]
    ancho = dim["ancho_m"]

    if ancho >= 15:
        # Bulevar ancho: puede albergar el modelo grande o mediano
        return "3×3 m  (9 m²) – Mediano"  # versátil, atractivo, viable
    else:
        # Zonas estrechas (8m): solo pequeño o mediano (6×4 no cabe con servicio)
        if "Deport" in zona_name or "Pista" in zona_name or "Cancha" in zona_name:
            return "3×2 m  (6 m²) – Pequeño"   # deportivo: servicio rápido
        else:
            return "3×3 m  (9 m²) – Mediano"   # arena: algo más de oferta


def fig_layout_zona(zona_name: str, kiosko_key: str,
                    n_kioskos_recomendados: int) -> go.Figure:
    """
    Diagrama esquemático 2D de planta de la zona con kioskos posicionados.
    """
    dim = ZONE_DIMENSIONS[zona_name]
    ksk = KIOSKO_TIPOS[kiosko_key]
    sp = SPATIAL_PARAMS
    L = dim["longitud_m"]
    W = dim["ancho_m"]
    lados = dim["lados_disponibles"]
    fondo = ksk["fondo_m"]
    ancho_k = ksk["ancho_m"]
    color_zona = dim["color"]
    color_ksk = ksk["color"]

    fig = go.Figure()

    # Fondo: zona completa
    fig.add_shape(type="rect", x0=0, y0=0, x1=L, y1=W,
                  fillcolor=color_zona, opacity=0.12,
                  line=dict(color=color_zona, width=2))

    # Corredor peatonal
    if lados == 2:
        cen = W / 2
        half = sp["circulacion_bulevar_central_m"] / 2
        fig.add_shape(type="rect", x0=0, y0=cen - half, x1=L, y1=cen + half,
                      fillcolor="#ecf0f1", opacity=0.6,
                      line=dict(color="#bdc3c7", width=1))
        fig.add_annotation(x=L * 0.5, y=cen, text="CORREDOR PEATONAL",
                            showarrow=False, font=dict(size=9, color="#7f8c8d"))
    else:
        fig.add_shape(type="rect", x0=0, y0=0, x1=L, y1=sp["circulacion_minima_m"],
                      fillcolor="#ecf0f1", opacity=0.6,
                      line=dict(color="#bdc3c7", width=1))
        fig.add_annotation(x=L * 0.5, y=sp["circulacion_minima_m"] / 2,
                            text="CORREDOR PEATONAL", showarrow=False,
                            font=dict(size=9, color="#7f8c8d"))

    # Distribuir kioskos a lo largo
    sep = sp["separacion_entre_kioskos_m"]

    # Calcular posiciones X de kioskos
    if lados == 2:
        n_por_lado = max(1, (n_kioskos_recomendados + 1) // 2)
    else:
        n_por_lado = n_kioskos_recomendados

    # Calcular espaciado uniforme
    espacio_total = L - 2 * sep   # margen inicial y final
    if n_por_lado > 1:
        paso = espacio_total / (n_por_lado - 1) if n_por_lado > 1 else espacio_total
        paso = max(paso, ancho_k + sep)
    else:
        paso = espacio_total
    x_posiciones = [sep + i * paso for i in range(n_por_lado)]

    # Posiciones Y según lados
    if lados == 2:
        cen = W / 2
        half_corr = sp["circulacion_bulevar_central_m"] / 2
        y_posiciones_list = [
            (cen + half_corr + sp["servicio_frontal_m"],          # lado superior (y0)
             cen + half_corr + sp["servicio_frontal_m"] + fondo),  # lado superior (y1)
            (cen - half_corr - sp["servicio_frontal_m"] - fondo,  # lado inferior (y0)
             cen - half_corr - sp["servicio_frontal_m"]),          # lado inferior (y1)
        ]
    else:
        y0_k = W - fondo - sp["margen_lateral_m"]
        y1_k = W - sp["margen_lateral_m"]
        y_posiciones_list = [(y0_k, y1_k)]

    # Giros asignados a cada kiosko
    giros_asignados = assign_giros(zona_name, n_kioskos_recomendados)

    k_num = 0
    for yi, (y0, y1) in enumerate(y_posiciones_list):
        n_este_lado = n_por_lado if yi == 0 else (n_kioskos_recomendados - n_por_lado)
        if n_este_lado <= 0:
            continue
        xs = [sep + i * paso for i in range(n_este_lado)]
        for xi, x0 in enumerate(xs):
            x1 = x0 + ancho_k
            giro = giros_asignados[k_num] if k_num < len(giros_asignados) else "Bebidas"
            icono = GIROS[giro]["icono"]
            g_color = GIROS[giro]["color"]
            k_num += 1
            fig.add_shape(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                          fillcolor=g_color, opacity=0.88,
                          line=dict(color="white", width=1.5))
            # Número de kiosko
            fig.add_annotation(
                x=(x0 + x1) / 2, y=(y0 * 0.65 + y1 * 0.35),
                text=f"<b>K{k_num}</b>", showarrow=False,
                font=dict(size=7, color="white", family="Arial Bold"),
            )
            # Icono del giro
            fig.add_annotation(
                x=(x0 + x1) / 2, y=(y0 * 0.35 + y1 * 0.65),
                text=icono, showarrow=False,
                font=dict(size=9),
            )

    # Dimensiones
    fig.add_annotation(x=L / 2, y=-W * 0.12, text=f"← {L:,.0f} m →",
                       showarrow=False, font=dict(size=10, color="#555"))
    fig.add_annotation(x=-L * 0.03, y=W / 2, text=f"← {W} m →",
                       showarrow=False, font=dict(size=10, color="#555"),
                       textangle=-90)

    short_name = zona_name.split("–")[1].strip() if "–" in zona_name else zona_name
    fig.update_layout(
        title=dict(
            text=f"Esquema de planta – {short_name} | {kiosko_key} | {k_num} kioskos",
            font=dict(size=14, family="Arial"), x=0.02,
        ),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[-L * 0.06, L * 1.04], showgrid=False,
                   showticklabels=False, zeroline=False),
        yaxis=dict(range=[-W * 0.25, W * 1.15], showgrid=False,
                   showticklabels=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        margin=dict(l=40, r=20, t=60, b=40),
        height=320,
    )
    return fig


def tabla_espacial_resumen(forecast_zonas: dict) -> pd.DataFrame:
    """
    Tabla de síntesis: dimensiones + kiosko recomendado + cantidad + observaciones.
    """
    rows = []
    zona_names = list(ZONE_DIMENSIONS.keys())
    forecast_keys = list(forecast_zonas.keys())

    for i, zona_name in enumerate(zona_names):
        dim = ZONE_DIMENSIONS[zona_name]
        ksk_rec = recommend_kiosko_type(zona_name)
        ksk = KIOSKO_TIPOS[ksk_rec]
        fit = _ancho_util_por_kiosko(zona_name, ksk_rec)
        cap = _max_kioskos_por_longitud(zona_name, ksk_rec)

        n_demanda = forecast_zonas[forecast_keys[i]]["kioskos_recomendados"] if i < len(forecast_keys) else 1
        # Recomendación inicial = Fase 1 del plan de implementación (encaja demanda actual + contexto urbano)
        fase1_año = min(PLAN_FASES.keys())
        n_final = PLAN_FASES[fase1_año]["kioskos"].get(zona_name, max(n_demanda, 2))

        rows.append({
            "Zona": zona_name,
            "Longitud": f"{dim['longitud_m']:,} m",
            "Ancho": f"{dim['ancho_m']} m",
            "Área total": f"{dim['longitud_m'] * dim['ancho_m']:,} m²",
            "Lados": dim["lados_disponibles"],
            "Tipo recomendado": ksk_rec,
            "Área kiosko": f"{ksk['area_m2']} m²",
            "Máx. físico práctico": cap["max_practico_total"],
            "Por demanda": n_demanda,
            "RECOMENDADO FINAL": n_final,
            "Viabilidad espacial": fit["observacion"],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# PROYECCIÓN 2025–2035
# ══════════════════════════════════════════════════════════════════════════════

def _consumidores_dia_boulevard(año: int) -> int:
    """Consumidores diarios potenciales en el bulevar para un año dado."""
    trafico = PROYECCION_PARAMS["trafico_boulevard_dia"].get(año, {})
    tasas = PROYECCION_PARAMS["tasa_conversion_por_categoria"]
    return round(sum(trafico.get(cat, 0) * tasas.get(cat, 0.05) for cat in trafico))


def _kioskos_por_demanda_pura(consumidores_dia: int) -> int:
    """Cuántos kioskos necesitaría la demanda pura (sin cap urbano)."""
    p = KIOSKO_PARAMS
    cap_dia = p["horas_finde"] * p["max_atendidos_hora"] * p["ocupacion_objetivo"]
    return max(1, round(consumidores_dia / cap_dia))


def fig_trafico_parque_total(estadisticas: dict) -> go.Figure:
    """
    Visitantes diarios al Parque Bicentenario 2025→2036.
    Arranca desde el promedio real 2025 y crece con la población del sector.
    """
    total_anual_2025 = sum(
        df["total"].sum() for df in estadisticas.values() if "total" in df.columns
    )
    diario_2025 = max(1, round(total_anual_2025 / 365))

    pop      = PROYECCION_PARAMS["poblacion_sector_hab"]
    pop_base = pop[min(pop.keys())]   # 142,034 hab (2026)

    años = [2025] + sorted(pop.keys())

    vis = {2025: diario_2025}
    for a in sorted(pop.keys()):
        vis[a] = round(diario_2025 * pop[a] / pop_base)

    años_str = [str(a) for a in años]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="Visitantes parque",
        x=años_str, y=[vis[a] for a in años],
        mode="lines+markers",
        fill="tozeroy",
        fillcolor="rgba(41,128,185,0.35)",
        line=dict(color="#2980b9", width=2.5),
        marker=dict(size=8),
    ))
    for a in años:
        fig.add_annotation(
            x=str(a), y=vis[a],
            text=f"<b>{vis[a]:,}/día</b>",
            showarrow=False, yanchor="bottom",
            font=dict(size=10, color="#2c3e50"),
            yshift=8,
        )
    fig.add_vline(x="2025", line=dict(color="#27ae60", dash="dot", width=1.5))
    fig.add_annotation(
        x="2025", y=1, xref="x", yref="paper",
        text="Dato real 2025",
        showarrow=False, xanchor="left", yanchor="bottom",
        font=dict(color="#27ae60", size=10),
    )
    fig.update_layout(
        title=dict(text="Visitantes diarios – Parque Bicentenario (2025–2036)",
                   font=dict(size=15), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        xaxis=dict(title="Año", showgrid=False),
        yaxis=dict(title="Personas / día", gridcolor="#f0f0f0"),
        showlegend=False,
        margin=dict(l=50, r=20, t=80, b=40),
    )
    return fig


def fig_demanda_vs_plan_parque(estadisticas: dict, df_encuesta: pd.DataFrame) -> go.Figure:
    """
    Consumidores/día del parque vs. plan de kioskos (10→25).
    Usa exactamente la misma base de visitantes que fig_trafico_parque_total:
    dato real 2025 creciendo con población, sin metro.
    """
    rates = _conversion_rates(df_encuesta)

    total_anual_2025 = sum(
        df["total"].sum() for df in estadisticas.values() if "total" in df.columns
    )
    diario_2025 = max(1, round(total_anual_2025 / 365))

    pop      = PROYECCION_PARAMS["poblacion_sector_hab"]
    pop_base = pop[min(pop.keys())]

    años = sorted(PLAN_FASES.keys())   # 2026, 2029, 2033, 2036

    vis = {a: round(diario_2025 * pop.get(a, pop_base) / pop_base) for a in años}

    consumidores = {a: round(vis[a] * rates["tasa_consumo"]) for a in años}
    k_plan       = {a: sum(PLAN_FASES[a]["kioskos"].values())       for a in años}

    años_str = [str(a) for a in años]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Consumidores/día (parque completo)",
        x=años_str, y=[consumidores[a] for a in años],
        marker_color="#3498db", opacity=0.75, yaxis="y",
        text=[f"{consumidores[a]:,}" for a in años],
        textposition="outside", textfont=dict(size=10),
    ))
    fig.add_trace(go.Scatter(
        name="Kioskos – plan recomendado (3 zonas)",
        x=años_str, y=[k_plan[a] for a in años],
        mode="lines+markers+text",
        text=[str(k_plan[a]) for a in años],
        textposition="top center",
        textfont=dict(size=11, color="#27ae60"),
        marker=dict(color="#27ae60", size=10, symbol="diamond"),
        line=dict(color="#27ae60", width=2.5, dash="dash"),
        yaxis="y2",
    ))
    fig.update_layout(
        title=dict(text="Demanda vs. plan de kioskos – Parque Bicentenario (todas las zonas)",
                   font=dict(size=15), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Año", showgrid=False),
        yaxis=dict(title="Consumidores/día", gridcolor="#f0f0f0", side="left"),
        yaxis2=dict(title="N° de kioskos", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Arial", size=12),
        margin=dict(l=60, r=70, t=80, b=40),
    )
    return fig


def fig_plan_fases() -> go.Figure:
    """Barras apiladas: kioskos totales por zona en cada fase."""
    años = sorted(PLAN_FASES.keys())
    años_str = [str(a) for a in años]
    zonas = list(ZONE_DIMENSIONS.keys())
    short_z = [z.split("–")[1].strip() if "–" in z else z for z in zonas]
    colores = [ZONE_DIMENSIONS[z]["color"] for z in zonas]

    fig = go.Figure()
    for zi, zona_name in enumerate(zonas):
        vals = [PLAN_FASES[a]["kioskos"][zona_name] for a in años]
        fig.add_trace(go.Bar(
            name=short_z[zi],
            x=años_str, y=vals,
            marker_color=colores[zi],
            text=vals, textposition="inside",
            textfont=dict(color="white", size=11),
        ))
    # Totales sobre las barras
    for año in años:
        total = sum(PLAN_FASES[año]["kioskos"].values())
        fig.add_annotation(x=str(año), y=total + 0.3,
                           text=f"<b>{total} total</b>",
                           showarrow=False, font=dict(size=11, color="#2c3e50"))
    # Hitos dinámicos (tomados directamente de PLAN_FASES para evitar años hardcodeados)
    for año in años:
        hito_txt = PLAN_FASES[año].get("hito", "")
        if hito_txt:
            # Solo el emoji (primer carácter)
            icono = hito_txt.split()[0]
            total = sum(PLAN_FASES[año]["kioskos"].values())
            fig.add_annotation(x=str(año), y=total + 1.3,
                               text=icono, showarrow=False, font=dict(size=16))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Plan de implementación por fases (2025–2035)",
                   font=dict(size=15), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Año / Fase", showgrid=False),
        yaxis=dict(title="Número de kioskos", gridcolor="#f0f0f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        font=dict(family="Arial", size=12),
        margin=dict(l=40, r=20, t=90, b=40),
    )
    return fig


def tabla_plan_fases() -> pd.DataFrame:
    """Tabla del plan de fases con contexto y distribución por zona."""
    rows = []
    for año, fd in sorted(PLAN_FASES.items()):
        k = fd["kioskos"]
        rows.append({
            "Año": año,
            "Fase": fd["fase"],
            "Hito": fd["hito"],
            "Población (hab.)": f"{fd['poblacion_hab']:,}",
            "Bulevar de las Canchas": k.get("Zona 1 – Bulevar de las Canchas", 0),
            "Total": sum(k.values()),
            "Contexto": fd["contexto"],
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS COMERCIAL: DISTANCIAS Y ROTACIÓN DE GIROS
# ══════════════════════════════════════════════════════════════════════════════

def assign_giros(zona_name: str, n_kioskos: int) -> list:
    """Asigna un giro comercial a cada kiosko siguiendo la rotación de la zona."""
    tipo = ZONAS[zona_name]["tipo"]
    rotation = GIRO_ROTATION[tipo]
    return [rotation[i % len(rotation)] for i in range(n_kioskos)]


def demanda_por_giro(df_encuesta: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula, para cada giro comercial, el % de encuestados que eligió en
    '¿Qué productos o servicios consumiría dentro del parque?' alguna opción
    asociada a ese giro (ver GIRO_PRODUCTO_MAP). Sustenta con datos reales
    la elección de giros en la rotación comercial.
    """
    serie = df_encuesta["productos_interes"].dropna() if "productos_interes" in df_encuesta.columns else pd.Series(dtype=str)
    total = len(serie)
    counts = serie.value_counts()
    rows = []
    for giro, productos in GIRO_PRODUCTO_MAP.items():
        n = sum(counts.get(p, 0) for p in productos)
        pct = round(n / total * 100, 1) if total else 0.0
        rows.append({
            "giro": giro,
            "productos_encuesta": ", ".join(productos),
            "n_encuestados": int(n),
            "pct_encuestados": pct,
        })
    df = pd.DataFrame(rows).sort_values("pct_encuestados", ascending=False).reset_index(drop=True)
    return df


def fig_demanda_por_giro(df_encuesta: pd.DataFrame) -> go.Figure:
    """Barras: % de encuestados que respaldan con su elección cada giro comercial."""
    df = demanda_por_giro(df_encuesta)
    colors = [GIROS[g]["color"] for g in df["giro"]]
    labels = [f"{GIROS[g]['icono']} {g}" for g in df["giro"]]

    fig = go.Figure(go.Bar(
        x=labels, y=df["pct_encuestados"],
        marker_color=colors,
        text=[f"{p}%" for p in df["pct_encuestados"]],
        textposition="outside",
        hovertext=df["productos_encuesta"],
    ))
    fig.update_layout(
        title=dict(text="Respaldo de la encuesta a cada giro comercial",
                   font=dict(size=15), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=13),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="% de encuestados interesados", gridcolor="#f0f0f0",
                   range=[0, df["pct_encuestados"].max() * 1.25 if len(df) else 1]),
        margin=dict(l=40, r=20, t=60, b=60),
    )
    return fig


def commercial_distance_analysis(zona_name: str, n_kioskos: int) -> dict:
    """
    Calcula las posiciones óptimas de los kioskos a lo largo de la zona,
    verifica que se cumplen las distancias mínimas comerciales y asigna giros.
    """
    dim = ZONE_DIMENSIONS[zona_name]
    tipo = ZONAS[zona_name]["tipo"]
    L = dim["longitud_m"]
    giros = assign_giros(zona_name, n_kioskos)

    # Margen en extremos: 5% de la longitud, mínimo 20m, máximo 50m
    margen = min(50.0, max(20.0, L * 0.05))
    long_util = L - 2 * margen

    if n_kioskos <= 0:
        return {"zona": zona_name, "n_kioskos": 0, "giros": [], "posiciones_m": [],
                "distancias": [], "todos_cumplen": True}
    elif n_kioskos == 1:
        posiciones = [L / 2]
    else:
        posiciones = [margen + i * long_util / (n_kioskos - 1) for i in range(n_kioskos)]

    # Análisis de distancia entre pares consecutivos
    distancias = []
    for i in range(len(posiciones) - 1):
        d = round(posiciones[i + 1] - posiciones[i])
        mismo_giro = giros[i] == giros[i + 1]
        min_req = (COMMERCIAL_PARAMS["distancia_min_mismo_giro_m"][tipo]
                   if mismo_giro
                   else COMMERCIAL_PARAMS["distancia_min_dif_giro_m"][tipo])
        distancias.append({
            "par": f"K{i + 1} → K{i + 2}",
            "giro_a": giros[i],
            "giro_b": giros[i + 1],
            "distancia_m": d,
            "distancia_min_m": min_req,
            "mismo_giro": mismo_giro,
            "cumple": d >= min_req,
        })

    return {
        "zona": zona_name,
        "n_kioskos": n_kioskos,
        "longitud_m": L,
        "tipo": tipo,
        "giros": giros,
        "posiciones_m": [round(p) for p in posiciones],
        "distancias": distancias,
        "radio_captacion_m": COMMERCIAL_PARAMS["radio_captacion_m"][tipo],
        "todos_cumplen": all(d["cumple"] for d in distancias) if distancias else True,
    }


def fig_distribucion_comercial(zona_name: str, n_kioskos: int) -> go.Figure:
    """
    Diagrama lineal de distribución comercial: posiciones, distancias y giros.
    """
    analysis = commercial_distance_analysis(zona_name, n_kioskos)
    L = analysis["longitud_m"]
    posiciones = analysis["posiciones_m"]
    giros = analysis["giros"]
    radio = analysis["radio_captacion_m"]

    fig = go.Figure()

    # Fondo de la zona (caminera/bulevar)
    fig.add_shape(type="rect", x0=0, y0=0.25, x1=L, y1=0.75,
                  fillcolor="#dfe6e9", opacity=0.6,
                  line=dict(color="#b2bec3", width=1.5))

    # Radio de captación (área de influencia de cada kiosko)
    for pos, giro in zip(posiciones, giros):
        color = GIROS[giro]["color"]
        r = min(radio, L * 0.45)
        fig.add_shape(type="rect",
                      x0=max(0, pos - r), y0=0.08, x1=min(L, pos + r), y1=0.92,
                      fillcolor=color, opacity=0.08, line=dict(width=0))

    # Cotas de distancia entre kioskos
    for d_info in analysis["distancias"]:
        idx = int(d_info["par"][1]) - 1
        pos_a = posiciones[idx]
        pos_b = posiciones[idx + 1]
        mid = (pos_a + pos_b) / 2
        ok = d_info["cumple"]
        color_ok = "#27ae60" if ok else "#e74c3c"

        # Flecha doble de distancia
        fig.add_annotation(
            ax=pos_a, ay=-0.12, x=pos_b, y=-0.12,
            xref="x", yref="y", axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor=color_ok, showarrow=True,
        )
        # Texto de distancia + estado
        fig.add_annotation(
            x=mid, y=-0.27,
            text=(f"<b>{d_info['distancia_m']:,} m</b> {'✅' if ok else '⚠️'}<br>"
                  f"<span style='font-size:8px'>mín. {d_info['distancia_min_m']} m "
                  f"({'mismo' if d_info['mismo_giro'] else 'dif.'} giro)</span>"),
            showarrow=False, font=dict(size=9, color=color_ok), align="center",
        )

    # Kioskos (rectángulos + etiquetas)
    kw = max(L * 0.014, 1.0)
    for i, (pos, giro) in enumerate(zip(posiciones, giros)):
        color = GIROS[giro]["color"]
        icono = GIROS[giro]["icono"]
        fig.add_shape(type="rect",
                      x0=pos - kw, y0=0.15, x1=pos + kw, y1=0.85,
                      fillcolor=color, opacity=0.92,
                      line=dict(color="white", width=1.5))
        fig.add_annotation(x=pos, y=0.5,
                           text=f"<b>K{i + 1}</b>",
                           showarrow=False,
                           font=dict(size=8, color="white", family="Arial Bold"))
        # Etiqueta superior con giro
        fig.add_annotation(
            x=pos, y=1.08,
            text=f"{icono} <b>K{i + 1}</b><br>{giro}",
            showarrow=True, arrowhead=2, arrowcolor=color,
            arrowwidth=1.5, ay=-28,
            font=dict(size=9, color=color), align="center",
        )

    # Dimensión total en la parte inferior
    fig.add_annotation(
        x=L / 2, y=-0.52,
        text=f"← Longitud total de la zona: {L:,} m →",
        showarrow=False, font=dict(size=10, color="#636e72"),
    )

    short = zona_name.split("–")[1].strip() if "–" in zona_name else zona_name
    fig.update_layout(
        title=dict(text=f"Distribución comercial – {short}", font=dict(size=14), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[-L * 0.05, L * 1.05], showgrid=False,
                   showticklabels=False, zeroline=False),
        yaxis=dict(range=[-0.7, 1.55], showgrid=False,
                   showticklabels=False, zeroline=False),
        margin=dict(l=20, r=20, t=55, b=55),
        height=310,
    )
    return fig


def tabla_distribucion_comercial(forecast_zonas: dict) -> pd.DataFrame:
    """Tabla resumen de distribución de giros y cumplimiento de distancias."""
    rows = []
    zona_names = list(ZONE_DIMENSIONS.keys())
    forecast_keys = list(forecast_zonas.keys())

    fase1_año = min(PLAN_FASES.keys())
    for i, zona_name in enumerate(zona_names):
        n = PLAN_FASES[fase1_año]["kioskos"].get(zona_name, 2)
        analysis = commercial_distance_analysis(zona_name, n)
        tipo = ZONAS[zona_name]["tipo"]

        giro_seq = " → ".join(
            f"{GIROS[g]['icono']} {g}" for g in analysis["giros"]
        )
        if analysis["distancias"]:
            dist_real = min(d["distancia_m"] for d in analysis["distancias"])
            todos_ok = analysis["todos_cumplen"]
        else:
            dist_real = analysis["longitud_m"]
            todos_ok = True

        rows.append({
            "Zona": zona_name.split("–")[1].strip() if "–" in zona_name else zona_name,
            "Kioskos": n,
            "Rotación de giros": giro_seq,
            "Dist. mínima real (m)": dist_real,
            "Dist. mín. requerida (dif. giro)": COMMERCIAL_PARAMS["distancia_min_dif_giro_m"][tipo],
            "Dist. mín. requerida (mismo giro)": COMMERCIAL_PARAMS["distancia_min_mismo_giro_m"][tipo],
            "Radio captación (m)": analysis["radio_captacion_m"],
            "Cumplimiento": "✅ Correcto" if todos_ok else "⚠️ Revisar",
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MODELO DE DEMANDA (sin cambios)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_zone_annual_visitors(estadisticas: dict, zone_sources: list,
                                   factor: float) -> dict:
    totals = []
    for source in zone_sources:
        if source in estadisticas:
            df = estadisticas[source]
            if "total" in df.columns:
                annual = df["total"].sum()
                if annual > 0:
                    totals.append(annual)
    if not totals:
        return {"anual": 0, "mensual_prom": 0, "diario_prom": 0, "base_actividades": 0}
    base_actividades = sum(totals)
    trafico_zona = base_actividades * FACTOR_VISITANTES_NO_CONTABILIZADOS
    anual = round(trafico_zona * factor)
    return {
        "anual": anual,
        "mensual_prom": round(anual / 12),
        "diario_prom": round(anual / 365),
        "base_actividades": int(base_actividades),
    }


def _conversion_rates(df_encuesta: pd.DataFrame) -> dict:
    def _tasa_si(col: str) -> float:
        """% de 'Sí' sobre quienes respondieron esa columna (excluye NaN/vacíos)."""
        if col not in df_encuesta.columns:
            return 0.0
        s = df_encuesta[col].dropna()
        s = s[s.str.strip() != ""]
        n = len(s)
        return s.str.strip().str.lower().isin(["sí", "si", "yes"]).sum() / n if n else 0.0

    consumiria_pct = _tasa_si("consumiria")
    kiosko_ok_pct  = _tasa_si("kiosko_adecuado")
    mejora_pct     = _tasa_si("kiosko_mejora_experiencia")

    gasto_map = {
        "Menos de $2": 1.5, "Entre $2 y $5": 3.5, "Entre $5 y $10": 7.5,
        "Entre $10 y $20": 15.0, "Más de $20": 22.0,
    }
    gasto_prom = 3.5
    if "gasto_dispuesto" in df_encuesta.columns:
        gastos = df_encuesta["gasto_dispuesto"].map(gasto_map).dropna()
        gasto_prom = gastos.mean() if len(gastos) > 0 else 3.5
    return {
        "tasa_consumo": round(consumiria_pct, 3),
        "tasa_aprobacion_kiosko": round(kiosko_ok_pct, 3),
        "gasto_promedio_usd": round(gasto_prom, 2),
        "tasa_mejora_experiencia": round(mejora_pct, 3),
    }


def _kiosko_capacity_annual() -> dict:
    p = KIOSKO_PARAMS
    dias_semana_anio = p["dias_semana_por_semana"] * p["semanas_anio"]
    dias_finde_anio = p["dias_finde_por_semana"] * p["semanas_anio"]
    cap_semana = p["horas_semana"] * p["max_atendidos_hora"] * p["ocupacion_objetivo"]
    cap_finde = p["horas_finde"] * p["max_atendidos_hora"] * p["ocupacion_objetivo"]
    anual = round(cap_semana * dias_semana_anio + cap_finde * dias_finde_anio)
    peak_diario = round(p["horas_finde"] * p["factor_peak"] * p["max_atendidos_hora"])
    return {"clientes_anuales_por_kiosko": anual, "clientes_diario_pico": peak_diario}


def forecast_kioskos(df_encuesta: pd.DataFrame, estadisticas: dict) -> dict:
    rates = _conversion_rates(df_encuesta)
    cap = _kiosko_capacity_annual()
    results = {}
    for zona_name, zona_cfg in ZONAS.items():
        visitors = _estimate_zone_annual_visitors(
            estadisticas, zona_cfg["fuentes_estadisticas"], zona_cfg["factor_captacion"]
        )
        demanda_anual = round(visitors["anual"] * rates["tasa_consumo"])
        demanda_diaria_prom = round(visitors["diario_prom"] * rates["tasa_consumo"])
        n_opt = demanda_anual / cap["clientes_anuales_por_kiosko"] if cap["clientes_anuales_por_kiosko"] > 0 else 0
        n_min = max(1, round(n_opt * 0.8))
        n_max = round(n_opt * 1.2) + 1
        n_recomendado = max(1, round(n_opt))
        results[zona_name] = {
            "descripcion": zona_cfg["descripcion"],
            "tipo": zona_cfg["tipo"],
            "base_usuarios_actividades": visitors.get("base_actividades", 0),
            "visitantes_anuales_estimados": visitors["anual"],
            "visitantes_diarios_promedio": visitors["diario_prom"],
            "demanda_anual_consumidores": demanda_anual,
            "demanda_diaria_promedio": demanda_diaria_prom,
            "kioskos_min": n_min,
            "kioskos_recomendados": n_recomendado,
            "kioskos_max": n_max,
            "ingresos_potenciales_anuales_usd": round(demanda_anual * rates["gasto_promedio_usd"]),
        }
    return {
        "zonas": results,
        "tasas": rates,
        "capacidad_kiosko": cap,
        "total_kioskos_min": sum(z["kioskos_min"] for z in results.values()),
        "total_kioskos_recomendados": sum(z["kioskos_recomendados"] for z in results.values()),
        "total_kioskos_max": sum(z["kioskos_max"] for z in results.values()),
    }


# ── Gráficos de demanda ────────────────────────────────────────────────────────

def fig_kioskos_por_zona(forecast: dict) -> go.Figure:
    zonas = list(forecast["zonas"].keys())
    short = [z.split("–")[1].strip() if "–" in z else z for z in zonas]
    fase1_año = min(PLAN_FASES.keys())
    fase4_año = max(PLAN_FASES.keys())
    arranque   = [PLAN_FASES[fase1_año]["kioskos"].get(z, 0) for z in zonas]
    proyeccion = [PLAN_FASES[fase4_año]["kioskos"].get(z, 0) for z in zonas]
    total_ini = sum(arranque)
    total_fin = sum(proyeccion)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"Arranque {fase1_año}  (total: {total_ini})",
        x=short, y=arranque,
        marker_color="#27ae60",
        text=arranque, textposition="outside", textfont=dict(size=13, color="#1a3a5c"),
    ))
    fig.add_trace(go.Bar(
        name=f"Proyección {fase4_año}  (total: {total_fin})",
        x=short, y=proyeccion,
        marker_color="#e67e22", opacity=0.85,
        text=proyeccion, textposition="outside", textfont=dict(size=13, color="#6e2e00"),
    ))
    fig.update_layout(
        barmode="group",
        title=dict(text=f"Kioskos por zona: arranque {fase1_año} vs. proyección {fase4_año}",
                   font=dict(size=15), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=13),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Número de kioskos", gridcolor="#f0f0f0", range=[0, max(proyeccion) * 1.25]),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=65, b=100),
    )
    return fig


def fig_visitantes_vs_demanda(estadisticas: dict, df_encuesta: pd.DataFrame) -> go.Figure:
    """
    Visitantes y consumidores por zona/día.
    Usa la misma base que fig_trafico_parque_total: total_anual/365 = dato real,
    distribuido proporcionalmente entre zonas según el peso de sus fuentes.
    """
    rates = _conversion_rates(df_encuesta)

    total_anual = sum(
        df["total"].sum() for df in estadisticas.values() if "total" in df.columns
    )
    diario_total = max(1, round(total_anual / 365))

    # Peso bruto de cada zona (suma de sus fuentes en estadisticas)
    zona_names = list(ZONAS.keys())
    pesos = {}
    for zona_name, zona_cfg in ZONAS.items():
        s = 0
        for source in zona_cfg["fuentes_estadisticas"]:
            if source in estadisticas and "total" in estadisticas[source].columns:
                s += estadisticas[source]["total"].sum()
        pesos[zona_name] = max(1, s)

    total_pesos = sum(pesos.values())
    short        = [z.split("–")[1].strip() if "–" in z else z for z in zona_names]
    visitantes   = [round(diario_total * pesos[z] / total_pesos) for z in zona_names]
    consumidores = [round(v * rates["tasa_consumo"]) for v in visitantes]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Visitantes/día",
        x=short, y=visitantes,
        marker_color="#3498db",
        text=[f"{v:,}" for v in visitantes],
        textposition="outside", textfont=dict(size=11),
    ))
    fig.add_trace(go.Bar(
        name="Consumidores potenciales/día",
        x=short, y=consumidores,
        marker_color="#2ecc71",
        text=[f"{v:,}" for v in consumidores],
        textposition="outside", textfont=dict(size=11),
    ))
    max_val = max(visitantes + consumidores)
    fig.update_layout(
        barmode="group",
        title=dict(text=f"Visitantes vs. Consumidores por zona / día  (total parque: {diario_total:,}/día)",
                   font=dict(size=15), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=13),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Personas/día", gridcolor="#f0f0f0", range=[0, max_val * 1.25]),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=100),
    )
    return fig


def fig_ingresos_diarios_zona(estadisticas: dict, df_encuesta: pd.DataFrame) -> go.Figure:
    """Ingreso diario total por zona = consumidores/día × gasto promedio."""
    zona_stats = _zona_stats_consistentes(estadisticas, df_encuesta)
    rates      = _conversion_rates(df_encuesta)
    gasto      = rates["gasto_promedio_usd"]
    zona_names = list(ZONAS.keys())
    short      = [z.split("–")[1].strip() if "–" in z else z for z in zona_names]
    colores    = [ZONAS[z].get("color", PALETTE[i]) for i, z in enumerate(zona_names)]

    ing_dia = [round(zona_stats[z]["visitantes_diarios"] * rates["tasa_consumo"] * gasto, 2)
               for z in zona_names]

    fig = go.Figure(go.Bar(
        x=short, y=ing_dia,
        text=[f"${v:,.2f}" for v in ing_dia],
        textposition="outside",
        marker_color=colores,
    ))
    fig.update_layout(
        title=dict(text=f"Ingreso diario por zona  (gasto prom. ${gasto:.2f})",
                   font=dict(size=15), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=13),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="USD / día", gridcolor="#f0f0f0",
                   range=[0, max(ing_dia) * 1.25]),
        showlegend=False,
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


def fig_ingresos_diarios_por_kiosko(estadisticas: dict,
                                    df_encuesta: pd.DataFrame) -> go.Figure:
    """Ingreso diario por kiosko en cada zona = ingreso_zona / n_kioskos (arranque 2026)."""
    zona_stats = _zona_stats_consistentes(estadisticas, df_encuesta)
    rates      = _conversion_rates(df_encuesta)
    gasto      = rates["gasto_promedio_usd"]
    fase1      = min(PLAN_FASES.keys())
    zona_names = list(ZONAS.keys())
    short      = [z.split("–")[1].strip() if "–" in z else z for z in zona_names]
    colores    = [ZONAS[z].get("color", PALETTE[i]) for i, z in enumerate(zona_names)]

    ing_por_kiosko = []
    for z in zona_names:
        ing_zona = zona_stats[z]["visitantes_diarios"] * rates["tasa_consumo"] * gasto
        n_k      = PLAN_FASES[fase1]["kioskos"].get(z, 1)
        ing_por_kiosko.append(round(ing_zona / n_k, 2))

    n_labels = [f"({PLAN_FASES[fase1]['kioskos'].get(z,1)} kiosko{'s' if PLAN_FASES[fase1]['kioskos'].get(z,1)>1 else ''})"
                for z in zona_names]

    fig = go.Figure(go.Bar(
        x=[f"{s}<br><sub>{n}</sub>" for s, n in zip(short, n_labels)],
        y=ing_por_kiosko,
        text=[f"${v:,.2f}" for v in ing_por_kiosko],
        textposition="outside",
        marker_color=colores,
    ))
    fig.update_layout(
        title=dict(text=f"Ingreso diario por kiosko – arranque {fase1}  (gasto prom. ${gasto:.2f})",
                   font=dict(size=15), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=13),
        xaxis=dict(showgrid=False),
        yaxis=dict(title="USD / kiosko / día", gridcolor="#f0f0f0",
                   range=[0, max(ing_por_kiosko) * 1.25]),
        showlegend=False,
        margin=dict(l=40, r=20, t=70, b=60),
    )
    return fig


def fig_ingresos_fases_zona(estadisticas: dict, df_encuesta: pd.DataFrame) -> go.Figure:
    """Ingreso diario por zona para cada fase del plan (2026-2036)."""
    zona_stats = _zona_stats_consistentes(estadisticas, df_encuesta)
    rates      = _conversion_rates(df_encuesta)
    gasto      = rates["gasto_promedio_usd"]
    tasa       = rates["tasa_consumo"]

    años       = sorted(PLAN_FASES.keys())
    pop_base   = PLAN_FASES[años[0]]["poblacion_hab"]
    zona_names = list(ZONAS.keys())
    short      = [z.split("–")[1].strip() if "–" in z else z for z in zona_names]
    colores    = [ZONAS[z].get("color", PALETTE[i]) for i, z in enumerate(zona_names)]

    fig = go.Figure()
    for zona, label, color in zip(zona_names, short, colores):
        vis_base = zona_stats[zona]["visitantes_diarios"]
        ing_by_year = []
        for a in años:
            factor = PLAN_FASES[a]["poblacion_hab"] / pop_base
            ing_by_year.append(round(vis_base * factor * tasa * gasto, 2))
        fig.add_trace(go.Bar(
            name=label, x=[str(a) for a in años], y=ing_by_year,
            text=[f"${v:,.0f}" for v in ing_by_year],
            textposition="outside", marker_color=color,
        ))

    max_val = max(
        round(zona_stats[z]["visitantes_diarios"]
              * (PLAN_FASES[max(años)]["poblacion_hab"] / pop_base) * tasa * gasto, 2)
        for z in zona_names
    )
    fig.update_layout(
        title=dict(text="Crecimiento de ingresos diarios por zona (2026–2036)",
                   font=dict(size=15), x=0.02),
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=12),
        xaxis=dict(title="Fase / Año", showgrid=False),
        yaxis=dict(title="USD / día", gridcolor="#f0f0f0", range=[0, max_val * 1.28]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=90, b=40),
    )
    return fig


def fig_ingresos_fases_por_kiosko(estadisticas: dict, df_encuesta: pd.DataFrame) -> go.Figure:
    """Ingreso diario por kiosko en cada zona para cada fase del plan."""
    zona_stats = _zona_stats_consistentes(estadisticas, df_encuesta)
    rates      = _conversion_rates(df_encuesta)
    gasto      = rates["gasto_promedio_usd"]
    tasa       = rates["tasa_consumo"]

    años       = sorted(PLAN_FASES.keys())
    pop_base   = PLAN_FASES[años[0]]["poblacion_hab"]
    zona_names = list(ZONAS.keys())
    short      = [z.split("–")[1].strip() if "–" in z else z for z in zona_names]
    colores    = [ZONAS[z].get("color", PALETTE[i]) for i, z in enumerate(zona_names)]

    fig = go.Figure()
    all_vals = []
    for zona, label, color in zip(zona_names, short, colores):
        vis_base = zona_stats[zona]["visitantes_diarios"]
        ing_by_year = []
        for a in años:
            factor  = PLAN_FASES[a]["poblacion_hab"] / pop_base
            ing_zona = vis_base * factor * tasa * gasto
            n_k      = max(1, PLAN_FASES[a]["kioskos"].get(zona, 1))
            ing_by_year.append(round(ing_zona / n_k, 2))
        all_vals.extend(ing_by_year)
        k_labels = [PLAN_FASES[a]["kioskos"].get(zona, 1) for a in años]
        fig.add_trace(go.Bar(
            name=label, x=[str(a) for a in años], y=ing_by_year,
            text=[f"${v:,.0f}<br><sub>({k}k)</sub>" for v, k in zip(ing_by_year, k_labels)],
            textposition="outside", marker_color=color,
        ))

    fig.update_layout(
        title=dict(text="Ingreso diario por kiosko por zona – evolución por fase",
                   font=dict(size=15), x=0.02),
        barmode="group",
        plot_bgcolor="white", paper_bgcolor="white", font=dict(family="Arial", size=12),
        xaxis=dict(title="Fase / Año", showgrid=False),
        yaxis=dict(title="USD / kiosko / día", gridcolor="#f0f0f0",
                   range=[0, max(all_vals) * 1.32]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=90, b=40),
    )
    return fig


def _zona_stats_consistentes(estadisticas: dict, df_encuesta: pd.DataFrame) -> dict:
    """
    Calcula visitantes y consumidores por zona usando la misma base que
    fig_trafico_parque_total: total_anual/365, distribuido proporcionalmente.
    """
    rates = _conversion_rates(df_encuesta)
    gasto_map = {
        "Menos de $2": 1.5, "Entre $2 y $5": 3.5, "Entre $5 y $10": 7.5,
        "Entre $10 y $20": 15.0, "Más de $20": 22.0,
    }
    gasto_prom = rates["gasto_promedio_usd"]

    total_anual = sum(
        df["total"].sum() for df in estadisticas.values() if "total" in df.columns
    )
    diario_total = max(1, round(total_anual / 365))

    pesos = {}
    for zona_name, zona_cfg in ZONAS.items():
        s = 0
        for source in zona_cfg["fuentes_estadisticas"]:
            if source in estadisticas and "total" in estadisticas[source].columns:
                s += estadisticas[source]["total"].sum()
        pesos[zona_name] = max(1, s)
    total_pesos = sum(pesos.values())

    result = {}
    for zona_name in ZONAS:
        vis_dia  = round(diario_total * pesos[zona_name] / total_pesos)
        vis_anio = vis_dia * 365
        cons_anio = round(vis_anio * rates["tasa_consumo"])
        ingresos  = round(cons_anio * gasto_prom)
        result[zona_name] = {
            "visitantes_diarios": vis_dia,
            "visitantes_anuales": vis_anio,
            "consumidores_anuales": cons_anio,
            "ingresos_anuales_usd": ingresos,
        }
    return result


def tabla_resumen_forecast(forecast: dict, estadisticas: dict = None,
                           df_encuesta=None) -> pd.DataFrame:
    """Tabla por zona. Si se pasan estadisticas+encuesta usa base consistente 661/día."""
    fase1 = min(PLAN_FASES.keys())
    fase4 = max(PLAN_FASES.keys())
    rows = []

    if estadisticas is not None and df_encuesta is not None:
        zona_stats = _zona_stats_consistentes(estadisticas, df_encuesta)
        for zona_name in ZONAS:
            s = zona_stats[zona_name]
            k_ini = PLAN_FASES[fase1]["kioskos"].get(zona_name, 1)
            k_fin = PLAN_FASES[fase4]["kioskos"].get(zona_name, 1)
            rows.append({
                "Zona": zona_name.split("–")[1].strip() if "–" in zona_name else zona_name,
                "Visitantes/día (2025)": f"{s['visitantes_diarios']:,}",
                "Visitantes/año (2025)": f"{s['visitantes_anuales']:,}",
                "Consumidores/año": f"{s['consumidores_anuales']:,}",
                f"Kioskos {fase1}": k_ini,
                f"Kioskos {fase4}": k_fin,
                "Ingresos potenciales/año (USD)": f"${s['ingresos_anuales_usd']:,}",
            })
    else:
        for zona, data in forecast["zonas"].items():
            rows.append({
                "Zona": zona,
                "Visitantes/año est.": f"{data['visitantes_anuales_estimados']:,}",
                "Consumidores/año": f"{data['demanda_anual_consumidores']:,}",
                "Kioskos rec.": data["kioskos_recomendados"],
                "Ingresos/año (USD)": f"${data['ingresos_potenciales_anuales_usd']:,}",
            })
    return pd.DataFrame(rows)
