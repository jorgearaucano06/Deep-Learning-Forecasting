"""
Componentes Visuales del Dashboard NOC - GPON Sentinel AI
==========================================================
Funciones que generan graficos Plotly dark-themed y componentes HTML
para el dashboard estilo Network Operations Center.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dash import html
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Plotly theme constants
# ---------------------------------------------------------------------------
SURFACE_0 = "#0b1326"
SURFACE_1 = "#131b2e"
SURFACE_2 = "#171f33"
SURFACE_3 = "#222a3d"
SURFACE_4 = "#2a3348"

PRIMARY = "#38bdf8"
SECONDARY = "#4edea3"
TERTIARY = "#f59e0b"
TERTIARY_LIGHT = "#ffc174"
ERROR = "#e74c3c"
ERROR_LIGHT = "#ffb4ab"

ON_SURFACE = "#dae2fd"
ON_SURFACE_DIM = "#a0aec0"
OUTLINE = "#87929a"
OUTLINE_DIM = "#3d4a5c"

FONT_SANS = "Inter, sans-serif"
FONT_MONO = "JetBrains Mono, Consolas, monospace"

RISK_COLORS = {
    "Normal": SECONDARY,
    "Medio": TERTIARY,
    "Alto": TERTIARY_LIGHT,
    "Critico": ERROR,
}

RISK_COLORS_EN = {
    "low": SECONDARY,
    "medium": TERTIARY,
    "high": TERTIARY_LIGHT,
    "critical": ERROR,
}

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor=SURFACE_2,
    plot_bgcolor=SURFACE_2,
    font=dict(family=FONT_SANS, color=ON_SURFACE, size=11),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=10, color=ON_SURFACE_DIM),
    ),
)


def _base_layout(**overrides) -> dict:
    """Return a copy of the dark layout with optional overrides."""
    layout = {**DARK_LAYOUT}
    layout.update(overrides)
    return layout


# ---------------------------------------------------------------------------
# 1. Risk Map (GIS)
# ---------------------------------------------------------------------------
def create_risk_map(segments_data: pd.DataFrame) -> go.Figure:
    """
    Mapa interactivo de segmentos coloreados por riesgo.
    Expects columns: segment_id, risk_level, fault_probability, lat, lon
    (lat/lon are generated if missing).
    """
    n = len(segments_data)
    np.random.seed(42)

    if "lat" not in segments_data.columns:
        segments_data = segments_data.copy()
        segments_data["lat"] = -12.0464 + np.random.normal(0, 0.045, n)
        segments_data["lon"] = -77.0428 + np.random.normal(0, 0.045, n)

    colors = [
        RISK_COLORS_EN.get(r, OUTLINE)
        for r in segments_data.get("risk_level", ["low"] * n)
    ]

    hover_texts = [
        f"<b>{s}</b><br>"
        f"Distrito: {d}<br>"
        f"Riesgo: {r.upper()}<br>"
        f"Prob. Falla: {p:.1%}<br>"
        f"Potencia: {pw:.1f} dBm"
        for s, d, r, p, pw in zip(
            segments_data.get("segment_id", [f"SEG-{i:03d}" for i in range(n)]),
            segments_data.get("distrito", ["Lima"] * n),
            segments_data.get("risk_level", ["low"] * n),
            segments_data.get("fault_probability", np.random.uniform(0, 1, n)),
            segments_data.get(
                "optical_power_dbm", np.random.uniform(-18, -8, n)
            ),
        )
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scattermap(
            lat=segments_data["lat"],
            lon=segments_data["lon"],
            mode="markers",
            marker=dict(size=10, color=colors, opacity=0.85),
            text=hover_texts,
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor=SURFACE_1,
                bordercolor=OUTLINE_DIM,
                font=dict(family=FONT_MONO, size=11, color=ON_SURFACE),
            ),
        )
    )

    fig.update_layout(
        map=dict(
            style="carto-darkmatter",
            center=dict(lat=-12.0464, lon=-77.0428),
            zoom=11,
        ),
        margin=dict(r=0, t=0, l=0, b=0),
        height=420,
        paper_bgcolor=SURFACE_2,
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Sensor Timeline
# ---------------------------------------------------------------------------
METRIC_LABELS = {
    "optical_power_dbm": "Potencia Optica (dBm)",
    "attenuation_db_km": "Atenuacion (dB/km)",
    "osnr_db": "OSNR (dB)",
    "ber": "BER",
    "temperature": "Temperatura (C)",
    "humidity": "Humedad (%)",
    "chromatic_dispersion": "Dispersion Cromatica",
}

METRIC_THRESHOLDS = {
    "optical_power_dbm": -25.0,
    "attenuation_db_km": 0.40,
    "osnr_db": 15.0,
    "ber": 1e-6,
}


def create_sensor_timeline(
    segment_data: pd.DataFrame,
    metric: str,
    segment_id: str = "",
) -> go.Figure:
    """
    Timeline interactivo de una metrica con zona de falla sombreada y umbral critico.
    """
    fig = go.Figure()

    # Main metric line
    fig.add_trace(
        go.Scatter(
            x=segment_data["timestamp"],
            y=segment_data[metric],
            mode="lines",
            name=METRIC_LABELS.get(metric, metric),
            line=dict(color=PRIMARY, width=1.5),
            hovertemplate="%{x|%d %b %H:%M}<br>%{y:.4g}<extra></extra>",
        )
    )

    # Fault shading
    if "is_fault" in segment_data.columns:
        fault_data = segment_data[segment_data["is_fault"] == 1]
        if len(fault_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=fault_data["timestamp"],
                    y=fault_data[metric],
                    mode="markers",
                    name="Zona de Falla",
                    marker=dict(color=ERROR, size=3, opacity=0.6),
                    hoverinfo="skip",
                )
            )
            # Add shaded fault region
            ymin = segment_data[metric].min()
            ymax = segment_data[metric].max()
            for _, group in fault_data.groupby(
                (fault_data["timestamp"].diff() > pd.Timedelta("30min")).cumsum()
            ):
                fig.add_vrect(
                    x0=group["timestamp"].iloc[0],
                    x1=group["timestamp"].iloc[-1],
                    fillcolor="rgba(231,76,60,0.08)",
                    line_width=0,
                    layer="below",
                )

    # Threshold line
    threshold = METRIC_THRESHOLDS.get(metric)
    if threshold is not None:
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color=ERROR_LIGHT,
            line_width=1,
            annotation_text="Umbral Critico",
            annotation_position="top right",
            annotation_font=dict(size=9, color=ERROR_LIGHT),
        )

    label = METRIC_LABELS.get(metric, metric)
    fig.update_layout(
        **_base_layout(
            height=280,
            title=dict(
                text=f"Telemetria: {label}" + (f" - {segment_id}" if segment_id else ""),
                font=dict(size=12, color=ON_SURFACE_DIM),
                x=0,
                xanchor="left",
            ),
            xaxis=dict(
                gridcolor=OUTLINE_DIM,
                zeroline=False,
            ),
            yaxis=dict(
                title=label,
                gridcolor=OUTLINE_DIM,
                zeroline=False,
            ),
            hovermode="x unified",
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Risk Distribution (Donut)
# ---------------------------------------------------------------------------
def create_risk_distribution(risk_counts: Dict[str, int]) -> go.Figure:
    """Dona chart de distribucion de riesgo."""
    ordered_labels = ["Normal", "Medio", "Alto", "Critico"]
    ordered_colors = [SECONDARY, TERTIARY, TERTIARY_LIGHT, ERROR]

    labels = []
    values = []
    colors = []
    for lbl, clr in zip(ordered_labels, ordered_colors):
        if lbl in risk_counts:
            labels.append(lbl)
            values.append(risk_counts[lbl])
            colors.append(clr)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.52,
                marker=dict(colors=colors, line=dict(color=SURFACE_2, width=2)),
                textinfo="label+value",
                textfont=dict(size=11, family=FONT_SANS),
                hovertemplate="<b>%{label}</b><br>Segmentos: %{value}<br>%{percent}<extra></extra>",
                sort=False,
            )
        ]
    )
    fig.update_layout(
        **_base_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Model Radar Chart
# ---------------------------------------------------------------------------
MODEL_RESULTS = {
    "LSTM BiDir": {
        "accuracy": 0.9997,
        "precision": 0.9994,
        "recall": 0.9991,
        "f1": 0.9994,
    },
    "Random Forest": {
        "accuracy": 0.9870,
        "precision": 0.9720,
        "recall": 0.9650,
        "f1": 0.9685,
    },
    "CNN 1D": {
        "accuracy": 0.9940,
        "precision": 0.9880,
        "recall": 0.9850,
        "f1": 0.9865,
    },
    "Autoencoder": {
        "accuracy": 0.9780,
        "precision": 0.9600,
        "recall": 0.9500,
        "f1": 0.9549,
    },
}

MODEL_COLORS = {
    "LSTM BiDir": PRIMARY,
    "Random Forest": SECONDARY,
    "CNN 1D": TERTIARY,
    "Autoencoder": ERROR_LIGHT,
}


def create_model_radar(results: Optional[Dict[str, Dict]] = None) -> go.Figure:
    """Radar chart comparando modelos ML."""
    if results is None:
        results = MODEL_RESULTS

    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]

    fig = go.Figure()
    for model_name, model_metrics in results.items():
        values = [model_metrics.get(m, 0) for m in metric_keys]
        values.append(values[0])

        fig.add_trace(
            go.Scatterpolar(
                r=values,
                theta=metrics + [metrics[0]],
                fill="toself",
                name=model_name,
                opacity=0.55,
                line=dict(
                    color=MODEL_COLORS.get(model_name, OUTLINE), width=2
                ),
                fillcolor=MODEL_COLORS.get(model_name, OUTLINE),
            )
        )

    fig.update_layout(
        **_base_layout(
            height=280,
            margin=dict(l=40, r=40, t=30, b=30),
            polar=dict(
                bgcolor=SURFACE_2,
                radialaxis=dict(
                    visible=True,
                    range=[0.90, 1.0],
                    gridcolor=OUTLINE_DIM,
                    linecolor=OUTLINE_DIM,
                    tickfont=dict(size=8, color=OUTLINE),
                ),
                angularaxis=dict(
                    gridcolor=OUTLINE_DIM,
                    linecolor=OUTLINE_DIM,
                    tickfont=dict(size=10, color=ON_SURFACE_DIM),
                ),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5,
                font=dict(size=9),
            ),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Prediction History (Dual-Axis Bar + Line)
# ---------------------------------------------------------------------------
def create_prediction_history(history_data: dict = None) -> go.Figure:
    """Historial de predicciones: barras (fallas reales vs anticipadas) + linea (precision)."""
    if history_data is not None:
        weeks = history_data["weeks"]
        real_faults = history_data["real_faults"]
        anticipated = history_data["anticipated"]
        accuracy = history_data["accuracy"]
    else:
        weeks = [f"Sem {i}" for i in range(1, 9)]
        real_faults = np.array([12, 15, 9, 18, 11, 14, 16, 13])
        anticipated = np.array([11, 14, 8, 17, 10, 13, 15, 12])
        accuracy = (anticipated / real_faults * 100).round(1)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=weeks,
            y=real_faults,
            name="Fallas Reales",
            marker_color=ERROR,
            opacity=0.7,
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Bar(
            x=weeks,
            y=anticipated,
            name="Anticipadas",
            marker_color=PRIMARY,
            opacity=0.7,
            yaxis="y",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks,
            y=accuracy,
            name="Precision %",
            mode="lines+markers",
            line=dict(color=SECONDARY, width=2),
            marker=dict(size=6),
            yaxis="y2",
        )
    )

    fig.update_layout(
        **_base_layout(
            height=280,
            barmode="group",
            title=dict(
                text="Historial de Predicciones (8 semanas)",
                font=dict(size=12, color=ON_SURFACE_DIM),
                x=0,
                xanchor="left",
            ),
            xaxis=dict(gridcolor=OUTLINE_DIM),
            yaxis=dict(
                title="Fallas",
                gridcolor=OUTLINE_DIM,
                zeroline=False,
            ),
            yaxis2=dict(
                title=dict(text="Precision %", font=dict(color=SECONDARY)),
                overlaying="y",
                side="right",
                range=[70, 105],
                gridcolor="rgba(0,0,0,0)",
                tickfont=dict(color=SECONDARY),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.25,
                xanchor="center",
                x=0.5,
            ),
        )
    )
    return fig


# ---------------------------------------------------------------------------
# 6. KPI Card HTML builder
# ---------------------------------------------------------------------------
def build_kpi_card(
    label: str,
    value: str,
    sub_text: str = "",
    accent: str = "kpi-primary",
    icon: str = "monitoring",
    has_pulse: bool = False,
    service_pct: float = 0,
) -> html.Div:
    """Returns an html.Div styled as a KPI card."""
    children = [
        html.Div(
            [
                html.Span(
                    icon,
                    className="material-symbols-outlined",
                    style={"fontSize": "16px", "opacity": "0.6"},
                ),
                label,
            ],
            className="kpi-card-label",
        ),
        html.Div(
            [
                value,
                html.Span(className="pulse-dot-red" if has_pulse else "", style={"marginLeft": "8px"}) if has_pulse else None,
            ],
            className="kpi-card-value",
            style={"color": _accent_color(accent)},
        ),
    ]

    if sub_text:
        children.append(html.Div(sub_text, className="kpi-card-sub"))

    if service_pct > 0:
        children.append(
            html.Div(
                html.Div(
                    style={"width": f"{service_pct}%"},
                    className="kpi-service-bar-fill",
                ),
                className="kpi-service-bar",
            )
        )

    return html.Div(children, className=f"kpi-card {accent}")


def _accent_color(accent: str) -> str:
    mapping = {
        "kpi-primary": PRIMARY,
        "kpi-error": ERROR,
        "kpi-secondary": SECONDARY,
        "kpi-tertiary": TERTIARY,
    }
    return mapping.get(accent, ON_SURFACE)


# ---------------------------------------------------------------------------
# 7. Alert Panel Items
# ---------------------------------------------------------------------------
def build_alert_item(
    segment_id: str,
    distrito: str,
    fault_type: str,
    probability: float,
    hours_to_fault: float,
    risk_level: str = "critical",
) -> html.Div:
    """Returns an html.Div styled as an alert item."""
    level_class = "critical" if risk_level == "critical" else "high"
    alert_class = "" if risk_level == "critical" else "alert-high"

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(segment_id, className="alert-item-segment"),
                            html.Span(
                                f" - {distrito}",
                                style={"fontSize": "0.7rem", "color": OUTLINE},
                            ),
                        ]
                    ),
                    html.Span(
                        f"{probability:.0%}",
                        className=f"alert-item-prob {level_class}",
                    ),
                ],
                className="alert-item-header",
            ),
            html.Div(
                f"{fault_type.replace('_',' ').title()} - {hours_to_fault:.0f}h estimadas a falla",
                className="alert-item-detail",
            ),
            html.Div(
                html.Div(
                    style={"width": f"{probability * 100:.0f}%"},
                    className=f"alert-progress-fill {level_class}",
                ),
                className="alert-progress-bar",
            ),
            html.Div(
                [
                    html.Button("Despachar", className="btn-alert btn-alert-dispatch"),
                    html.Button("Telemetria", className="btn-alert"),
                ],
                className="alert-actions",
            ),
        ],
        className=f"alert-item {alert_class}",
    )


# ---------------------------------------------------------------------------
# 8. HUD Metrics Strip
# ---------------------------------------------------------------------------
def build_hud_strip(metrics: Dict[str, str]) -> html.Div:
    """
    Builds the HUD strip with metric cells.
    metrics: dict of {label: value}
    """
    cells = []
    for label, value in metrics.items():
        cells.append(
            html.Div(
                [
                    html.Div(label, className="hud-cell-label"),
                    html.Div(value, className="hud-cell-value"),
                ],
                className="hud-cell",
            )
        )
    return html.Div(cells, className="hud-strip")


# ---------------------------------------------------------------------------
# 9. Map Legend (overlay)
# ---------------------------------------------------------------------------
def build_map_legend() -> html.Div:
    items = [
        ("Normal", SECONDARY),
        ("Medio", TERTIARY),
        ("Alto", TERTIARY_LIGHT),
        ("Critico", ERROR),
    ]
    return html.Div(
        [
            html.Div("Nivel de Riesgo", className="map-legend-title"),
            *[
                html.Div(
                    [
                        html.Span(
                            className="map-legend-dot",
                            style={"backgroundColor": color},
                        ),
                        label,
                    ],
                    className="map-legend-item",
                )
                for label, color in items
            ],
        ],
        className="map-legend",
    )


# ---------------------------------------------------------------------------
# 10. Summary Pills
# ---------------------------------------------------------------------------
def build_summary_pills(items: List[Dict[str, str]]) -> html.Div:
    """items: list of {label, value}"""
    return html.Div(
        [
            html.Div(
                [
                    html.Span(item["label"], className="summary-pill-label"),
                    html.Span(item["value"], className="summary-pill-value"),
                ],
                className="summary-pill",
            )
            for item in items
        ],
        className="summary-pills",
    )


# ---------------------------------------------------------------------------
# 11. NOC Panel wrapper
# ---------------------------------------------------------------------------
def noc_panel(title: str, children, icon: str = "analytics", badge: str = "") -> html.Div:
    """Wraps content in a styled NOC panel with header."""
    header_children = [
        html.Div(
            [
                html.Span(
                    icon,
                    className="material-symbols-outlined",
                    style={"fontSize": "16px"},
                ),
                title,
            ],
            className="noc-panel-title",
        ),
    ]
    if badge:
        header_children.append(
            html.Span(badge, className="header-badge")
        )

    return html.Div(
        [
            html.Div(header_children, className="noc-panel-header"),
            html.Div(children, className="noc-panel-body", style={"flex": "1"}),
        ],
        className="noc-panel",
        style={"display": "flex", "flexDirection": "column", "width": "100%"},
    )
