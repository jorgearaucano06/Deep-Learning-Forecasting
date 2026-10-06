"""
Componentes Visuales del Dashboard
=====================================
Funciones que generan graficos Plotly para el dashboard.

Plotly vs Matplotlib:
- Matplotlib: Graficos estaticos, bueno para papers/notebooks.
- Plotly: Graficos INTERACTIVOS, bueno para dashboards web.
  Zoom, hover, pan, export - todo sin escribir JavaScript.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List


def create_risk_map(segments_data: pd.DataFrame) -> go.Figure:
    """
    Crea mapa de riesgo de la red de fibra optica en Lima.

    Cada punto representa un segmento de fibra.
    Color indica nivel de riesgo: verde (ok) -> rojo (critico).
    """
    # Generar coordenadas ficticias alrededor de Lima
    n = len(segments_data)
    np.random.seed(42)
    lats = -12.0464 + np.random.normal(0, 0.05, n)
    lons = -77.0428 + np.random.normal(0, 0.05, n)

    color_map = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e67e22", "critical": "#e74c3c"}
    colors = [color_map.get(r, "#95a5a6") for r in segments_data.get("risk_level", ["low"] * n)]

    fig = go.Figure()
    fig.add_trace(go.Scattermap(
        lat=lats, lon=lons,
        mode="markers",
        marker=dict(size=12, color=colors, opacity=0.8),
        text=[f"Segmento: {s}<br>Riesgo: {r}<br>Prob: {p:.1%}"
              for s, r, p in zip(
                  segments_data.get("segment_id", [f"SEG-{i:03d}" for i in range(n)]),
                  segments_data.get("risk_level", ["low"] * n),
                  segments_data.get("fault_probability", [0.1] * n),
              )],
        hoverinfo="text",
    ))

    fig.update_layout(
        map=dict(style="open-street-map", center=dict(lat=-12.0464, lon=-77.0428), zoom=11),
        margin=dict(r=0, t=40, l=0, b=0),
        title="Mapa de Riesgo - Red de Fibra Optica Lima",
        height=500,
    )
    return fig


def create_kpi_cards(metrics: Dict) -> List[Dict]:
    """
    Genera datos para tarjetas KPI del dashboard.
    """
    return [
        {
            "title": "Segmentos Activos",
            "value": metrics.get("total_segments", 200),
            "color": "#3498db",
        },
        {
            "title": "Alertas Criticas",
            "value": metrics.get("critical_count", 0),
            "color": "#e74c3c",
        },
        {
            "title": "Deteccion Temprana",
            "value": f"{metrics.get('early_detection_rate', 0.95) * 100:.1f}%",
            "color": "#2ecc71",
        },
        {
            "title": "Ahorro Estimado",
            "value": f"${metrics.get('savings_usd', 45000):,}",
            "color": "#f39c12",
        },
    ]


def create_sensor_timeline(segment_data: pd.DataFrame, metric: str) -> go.Figure:
    """
    Timeline interactivo de una metrica de sensor.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=segment_data["timestamp"],
        y=segment_data[metric],
        mode="lines",
        name=metric,
        line=dict(color="#3498db", width=1.5),
    ))

    # Marcar zona de falla
    if "is_fault" in segment_data.columns:
        fault_data = segment_data[segment_data["is_fault"] == 1]
        if len(fault_data) > 0:
            fig.add_trace(go.Scatter(
                x=fault_data["timestamp"],
                y=fault_data[metric],
                mode="lines",
                name="Falla",
                line=dict(color="#e74c3c", width=2),
            ))

    fig.update_layout(
        title=f"Timeline - {metric}",
        xaxis_title="Tiempo",
        yaxis_title=metric,
        height=350,
        template="plotly_white",
    )
    return fig


def create_risk_distribution(risk_counts: Dict) -> go.Figure:
    """
    Dona chart de distribucion de riesgo en la red.
    """
    labels = list(risk_counts.keys())
    values = list(risk_counts.values())
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#e74c3c"]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values,
        hole=0.5,
        marker_colors=colors[:len(labels)],
        textinfo="label+value",
    )])

    fig.update_layout(
        title="Distribucion de Riesgo",
        height=350,
        template="plotly_white",
    )
    return fig


def create_model_performance_chart(results: Dict[str, Dict]) -> go.Figure:
    """
    Grafico de radar comparando modelos.
    """
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig = go.Figure()

    for model_name, model_metrics in results.items():
        values = [model_metrics.get(m, 0) for m in metrics]
        values.append(values[0])  # Cerrar el radar

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics + [metrics[0]],
            fill="toself",
            name=model_name,
            opacity=0.6,
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Rendimiento de Modelos",
        height=400,
        template="plotly_white",
    )
    return fig
