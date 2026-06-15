"""
Análisis de sentimientos y texto sobre los comentarios y sugerencias.
"""

import re
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64

PALETTE = px.colors.qualitative.Safe

STOPWORDS_ES = {
    "de", "la", "el", "en", "y", "a", "los", "del", "se", "las", "un",
    "por", "con", "una", "para", "es", "al", "lo", "que", "su", "más",
    "no", "o", "si", "hay", "me", "esto", "este", "esta", "son", "muy",
    "le", "todo", "pero", "ser", "tiene", "ya", "como", "su", "sus",
    "también", "bien", "solo", "sería", "siendo", "pueden", "puede",
    "cuando", "han", "sobre", "sin", "también", "así", "ni", "creo",
    "podría", "qué", "parque", "bicentenario", "área", "áreas",
}

POSITIVE_KEYWORDS = {
    "excelente", "bueno", "buena", "bonito", "bonita", "lindo", "linda",
    "agradable", "bello", "bella", "limpio", "limpia", "seguro", "segura",
    "tranquilo", "tranquila", "cómodo", "cómoda", "fantástico",
    "maravilloso", "increíble", "perfecto", "ideal", "necesario",
    "importante", "útil", "mejor", "grande", "amplio", "hermoso",
}

NEGATIVE_KEYWORDS = {
    "malo", "mala", "feo", "fea", "sucio", "sucia", "peligroso",
    "peligrosa", "inseguro", "insegura", "falta", "faltan", "necesitan",
    "mejorar", "arreglar", "problema", "problemas", "basura", "descuidado",
    "abandonado", "incómodo", "deficiente", "insuficiente", "poco",
    "pocos", "pocas", "escaso", "escasos",
}

TOPIC_KEYWORDS = {
    "Seguridad": ["seguridad", "vigilancia", "policía", "cámaras", "inseguro", "peligro"],
    "Baños": ["baños", "sanitarios", "servicios higiénicos", "baño"],
    "Comida y bebida": ["comida", "bebida", "alimentos", "snacks", "restaurante",
                        "café", "agua", "jugos", "helados", "kioskos", "kioscos"],
    "Deporte": ["deporte", "canchas", "fútbol", "tenis", "basquet", "voley",
                "correr", "ciclismo", "ejercicio", "atletismo", "gimnasio"],
    "Parqueadero": ["parqueadero", "estacionamiento", "parking", "parqueo"],
    "Iluminación": ["iluminación", "luz", "luminarias", "alumbrado"],
    "Mantenimiento": ["mantenimiento", "limpieza", "mantenimiento", "cuidado"],
    "Zonas verdes": ["árboles", "jardines", "vegetación", "plantas", "pasto"],
    "Accesibilidad": ["acceso", "rampas", "discapacidad", "inclusión", "señalética"],
    "Eventos": ["eventos", "conciertos", "espectáculos", "actividades", "cultural"],
}


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-záéíóúüñ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _classify_sentiment(text: str) -> str:
    words = set(_clean_text(text).split())
    pos_hits = len(words & POSITIVE_KEYWORDS)
    neg_hits = len(words & NEGATIVE_KEYWORDS)
    if pos_hits > neg_hits:
        return "Positivo"
    elif neg_hits > pos_hits:
        return "Negativo"
    elif pos_hits == neg_hits and pos_hits > 0:
        return "Mixto"
    else:
        return "Neutro"


def _tag_topics(text: str) -> list:
    text_lower = _clean_text(text)
    tags = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(topic)
    return tags if tags else ["Otro"]


def analyze_comments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Procesa la columna 'comentarios' y agrega columnas de:
    sentiment, topics, palabras clave.
    Retorna DataFrame solo con filas que tienen comentarios válidos.
    """
    comments = df["comentarios"].dropna().copy()
    result = pd.DataFrame({"comentario": comments})
    result["sentiment"] = result["comentario"].apply(_classify_sentiment)
    result["temas"] = result["comentario"].apply(_tag_topics)
    return result.reset_index(drop=True)


def sentiment_pie(result_df: pd.DataFrame) -> go.Figure:
    counts = result_df["sentiment"].value_counts()
    color_map = {
        "Positivo": "#27ae60",
        "Negativo": "#e74c3c",
        "Mixto": "#f39c12",
        "Neutro": "#95a5a6",
    }
    colors = [color_map.get(s, "#bdc3c7") for s in counts.index]
    fig = go.Figure(go.Pie(
        labels=counts.index.tolist(),
        values=counts.values.tolist(),
        hole=0.38,
        marker_colors=colors,
        textinfo="percent+label",
    ))
    fig.update_layout(
        title=dict(text="Análisis de Sentimientos en Comentarios", font=dict(size=16), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=13),
    )
    return fig


def topics_bar(result_df: pd.DataFrame) -> go.Figure:
    all_topics = [t for topics in result_df["temas"] for t in topics]
    counts = Counter(all_topics)
    topics_df = pd.DataFrame(list(counts.items()), columns=["Tema", "Menciones"])
    topics_df = topics_df.sort_values("Menciones", ascending=True).tail(12)

    fig = go.Figure(go.Bar(
        y=topics_df["Tema"].tolist(),
        x=topics_df["Menciones"].tolist(),
        orientation="h",
        marker_color=PALETTE[:len(topics_df)],
        text=topics_df["Menciones"].tolist(),
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(text="Temas más mencionados en comentarios", font=dict(size=16), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        xaxis=dict(title="Número de menciones", showgrid=False),
        yaxis=dict(showgrid=False),
        margin=dict(l=120, r=40, t=60, b=40),
    )
    return fig


def wordcloud_img(df: pd.DataFrame) -> str:
    """Genera wordcloud como imagen base64 PNG."""
    comments = df["comentarios"].dropna().apply(_clean_text)
    all_words = " ".join(comments)
    words_filtered = [w for w in all_words.split() if w not in STOPWORDS_ES and len(w) > 3]
    text = " ".join(words_filtered)

    wc = WordCloud(
        width=900, height=400,
        background_color="white",
        colormap="Blues",
        max_words=80,
        collocations=False,
    ).generate(text)

    buf = io.BytesIO()
    plt.figure(figsize=(10, 4.5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def top_keywords(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Tabla de palabras clave más frecuentes."""
    comments = df["comentarios"].dropna().apply(_clean_text)
    all_words = " ".join(comments).split()
    filtered = [w for w in all_words if w not in STOPWORDS_ES and len(w) > 3]
    counts = Counter(filtered).most_common(top_n)
    return pd.DataFrame(counts, columns=["Palabra", "Frecuencia"])


def sentiment_by_topic_heatmap(result_df: pd.DataFrame) -> go.Figure:
    """Heatmap de sentimiento por tema."""
    rows = []
    for _, row in result_df.iterrows():
        for topic in row["temas"]:
            rows.append({"tema": topic, "sentiment": row["sentiment"]})
    if not rows:
        return go.Figure()
    cross = pd.DataFrame(rows)
    ct = pd.crosstab(cross["tema"], cross["sentiment"]).fillna(0)
    sentiments = ["Positivo", "Negativo", "Mixto", "Neutro"]
    ct = ct.reindex(columns=[s for s in sentiments if s in ct.columns])

    fig = go.Figure(go.Heatmap(
        z=ct.values,
        x=ct.columns.tolist(),
        y=ct.index.tolist(),
        colorscale="RdYlGn",
        text=ct.values.astype(int),
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title=dict(text="Sentimiento por tema", font=dict(size=16), x=0.02),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Arial", size=13),
        xaxis=dict(title="Sentimiento"),
        yaxis=dict(title="Tema"),
        margin=dict(l=120, r=20, t=60, b=40),
    )
    return fig
