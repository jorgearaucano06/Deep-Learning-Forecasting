"""
Dashboard Plotly Dash
=======================
Panel de control en tiempo real para monitorear la red de fibra optica.

Dash es un framework de Plotly que permite crear dashboards web
interactivos usando solo Python (sin necesidad de JavaScript).

Secciones del dashboard:
1. KPI Cards: Metricas clave de la red.
2. Mapa de riesgo: Visualizacion geografica de segmentos.
3. Timeline de sensores: Metricas en tiempo real por segmento.
4. Distribucion de riesgo: Estado general de la red.

Uso:
    python scripts/run_dashboard.py
    -> Abrir http://localhost:8050
"""

import dash
from dash import html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from pathlib import Path

from src.dashboard.components import (
    create_risk_map,
    create_kpi_cards,
    create_sensor_timeline,
    create_risk_distribution,
)
from src.utils.config_loader import load_config, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_data() -> pd.DataFrame:
    """Carga los datos para el dashboard."""
    raw_path = get_path("raw_data") / "fiber_readings.parquet"
    if raw_path.exists():
        return pd.read_parquet(raw_path)

    # Datos de demo si no hay datos reales
    logger.warning("Datos no encontrados. Generando datos de demo.")
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "segment_id": [f"SEG-{i % 20 + 1:03d}" for i in range(n)],
        "optical_power_dbm": np.random.normal(-5, 1, n),
        "attenuation_db_km": np.random.normal(0.22, 0.02, n),
        "ber": 10 ** np.random.normal(-10, 0.5, n),
        "osnr_db": np.random.normal(30, 2, n),
        "chromatic_dispersion": np.random.normal(1.2, 0.2, n),
        "is_fault": np.random.choice([0, 1], n, p=[0.9, 0.1]),
        "fault_type": np.random.choice(
            ["normal", "physical_cut", "degradation", "bad_splice"],
            n, p=[0.7, 0.1, 0.1, 0.1],
        ),
    })


def create_dashboard() -> dash.Dash:
    """Crea y configura la aplicacion Dash."""

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY],
        title="Fiber Optic Network Monitor",
    )

    df = load_data()
    segments = sorted(df["segment_id"].unique())

    # Simular datos de riesgo
    np.random.seed(42)
    segment_risk = pd.DataFrame({
        "segment_id": segments,
        "risk_level": np.random.choice(
            ["low", "medium", "high", "critical"],
            len(segments), p=[0.6, 0.2, 0.15, 0.05],
        ),
        "fault_probability": np.random.uniform(0, 1, len(segments)),
    })

    # KPIs
    kpi_data = {
        "total_segments": len(segments),
        "critical_count": int((segment_risk["risk_level"] == "critical").sum()),
        "early_detection_rate": 0.92,
        "savings_usd": 47500,
    }
    kpi_cards = create_kpi_cards(kpi_data)

    # --- Layout ---
    app.layout = dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("Fiber Optic Network Monitor", className="text-primary mb-0"),
                html.P("Mantenimiento Predictivo con Deep Learning", className="text-muted"),
            ], width=8),
            dbc.Col([
                html.Div([
                    html.Span("Estado: ", className="text-muted"),
                    html.Span("Operativo", className="text-success fw-bold"),
                ], className="text-end mt-3"),
            ], width=4),
        ], className="my-3"),

        # KPI Cards
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H6(kpi["title"], className="text-muted"),
                        html.H3(str(kpi["value"]), style={"color": kpi["color"]}),
                    ])
                ], className="shadow-sm")
            ], width=3)
            for kpi in kpi_cards
        ], className="mb-4"),

        # Mapa y Distribucion
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=create_risk_map(segment_risk))
                    ])
                ], className="shadow-sm")
            ], width=8),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Graph(figure=create_risk_distribution(
                            segment_risk["risk_level"].value_counts().to_dict()
                        ))
                    ])
                ], className="shadow-sm")
            ], width=4),
        ], className="mb-4"),

        # Selector de segmento + Timeline
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Detalle por Segmento"),
                        dbc.Row([
                            dbc.Col([
                                dcc.Dropdown(
                                    id="segment-selector",
                                    options=[{"label": s, "value": s} for s in segments[:20]],
                                    value=segments[0],
                                    className="mb-3",
                                ),
                            ], width=4),
                            dbc.Col([
                                dcc.Dropdown(
                                    id="metric-selector",
                                    options=[
                                        {"label": "Potencia Optica (dBm)", "value": "optical_power_dbm"},
                                        {"label": "Atenuacion (dB/km)", "value": "attenuation_db_km"},
                                        {"label": "BER", "value": "ber"},
                                        {"label": "OSNR (dB)", "value": "osnr_db"},
                                        {"label": "Dispersion Cromatica", "value": "chromatic_dispersion"},
                                    ],
                                    value="optical_power_dbm",
                                    className="mb-3",
                                ),
                            ], width=4),
                        ]),
                        dcc.Graph(id="sensor-timeline"),
                    ])
                ], className="shadow-sm")
            ]),
        ]),
    ], fluid=True, className="py-3")

    # --- Callbacks ---
    @app.callback(
        Output("sensor-timeline", "figure"),
        [Input("segment-selector", "value"), Input("metric-selector", "value")],
    )
    def update_timeline(segment_id, metric):
        seg_data = df[df["segment_id"] == segment_id].sort_values("timestamp")
        return create_sensor_timeline(seg_data, metric)

    return app


# Instancia global
dash_app = create_dashboard()
