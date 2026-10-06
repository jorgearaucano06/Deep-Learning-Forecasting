"""
Dashboard NOC - GPON Sentinel AI
==================================
Panel de control estilo Network Operations Center para monitoreo
predictivo de la red de fibra optica GPON.

Datos reales:
- Lecturas de sensores desde fiber_readings.parquet (134,400 filas)
- Predicciones LSTM reales por segmento (modelo lstm_best.pt)
- Evaluacion real de 4 modelos ML en test data
- Metricas de negocio calculadas con calculate_business_metrics()

Uso:
    python scripts/run_dashboard.py
    -> Abrir http://localhost:8050
"""

import dash
from dash import html, dcc, Output, Input, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from src.dashboard.components import (
    create_risk_map,
    create_sensor_timeline,
    create_risk_distribution,
    create_model_radar,
    create_prediction_history,
    build_kpi_card,
    build_alert_item,
    build_hud_strip,
    build_map_legend,
    build_summary_pills,
    noc_panel,
    METRIC_LABELS,
    SURFACE_1,
    SURFACE_2,
    SURFACE_3,
    PRIMARY,
    SECONDARY,
    ERROR,
    ON_SURFACE,
    ON_SURFACE_DIM,
    OUTLINE,
    OUTLINE_DIM,
    FONT_MONO,
)
from src.dashboard.data_service import initialize_dashboard_data
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Lima timezone offset (UTC-5)
LIMA_TZ = timezone(timedelta(hours=-5))


# ===========================================================================
# LAYOUT BUILDERS
# ===========================================================================
def build_sidebar(sidebar_metrics: dict = None) -> html.Aside:
    sm = sidebar_metrics or {}
    olt_power = sm.get("olt_power", "+4.8 dBm")
    olt_pct = sm.get("olt_pct", 78)
    splitter_status = sm.get("splitter_status", "1:32 OK")
    splitter_pct = sm.get("splitter_pct", 95)

    nav_items = [
        ("dashboard", "Dashboard", True),
        ("map", "Mapa GIS", False),
        ("timeline", "Telemetria", False),
        ("table_chart", "Segmentos", False),
        ("model_training", "Modelos ML", False),
        ("settings", "Configuracion", False),
    ]

    return html.Aside(
        [
            html.Div(
                [
                    html.H2("GPON CORE"),
                    html.Div("NOC Engine v4.2", className="sidebar-version"),
                ],
                className="sidebar-logo",
            ),
            html.Nav(
                [
                    html.Div(
                        [
                            html.Span(icon, className="material-symbols-outlined"),
                            html.Span(label),
                        ],
                        className=f"sidebar-nav-item {'active' if active else ''}",
                    )
                    for icon, label, active in nav_items
                ],
                className="sidebar-nav",
            ),
            html.Div(
                [
                    html.Div("Potencia Media OLT", className="sidebar-widget-label"),
                    html.Div(olt_power, className="sidebar-widget-value"),
                    html.Div(
                        html.Div(
                            style={"width": f"{olt_pct}%", "backgroundColor": SECONDARY},
                            className="sidebar-widget-bar-fill",
                        ),
                        className="sidebar-widget-bar",
                    ),
                ],
                className="sidebar-widget",
                style={"borderTop": f"1px solid {OUTLINE_DIM}"},
            ),
            html.Div(
                [
                    html.Div("Nivel Splitters", className="sidebar-widget-label"),
                    html.Div(splitter_status, className="sidebar-widget-value"),
                    html.Div(
                        html.Div(
                            style={"width": f"{splitter_pct}%", "backgroundColor": PRIMARY},
                            className="sidebar-widget-bar-fill",
                        ),
                        className="sidebar-widget-bar",
                    ),
                ],
                className="sidebar-widget",
            ),
        ],
        className="noc-sidebar",
    )


def build_header(segments: list) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H1("GPON SENTINEL AI"),
                            html.Span(
                                [html.Span(className="pulse-dot"), " METRO LIMA"],
                                className="header-badge",
                            ),
                        ],
                        className="header-title",
                    ),
                    html.Div(
                        [
                            html.Div(id="lima-clock", className="header-clock"),
                            html.Div("Operador: NOC-Lima-01", className="header-operator"),
                        ],
                        className="header-right",
                    ),
                ],
                className="header-title-row",
            ),
            html.P(
                "Mantenimiento Predictivo Deep Learning",
                className="header-subtitle",
            ),
            html.Div(
                [
                    html.Span("Segmento", className="filter-label"),
                    dcc.Dropdown(
                        id="filter-segment",
                        options=[{"label": s, "value": s} for s in segments],
                        value=segments[0] if segments else None,
                        clearable=False,
                        style={"width": "160px", "fontSize": "0.8rem"},
                    ),
                    html.Span("Riesgo", className="filter-label", style={"marginLeft": "8px"}),
                    dbc.Checklist(
                        id="filter-risk",
                        options=[
                            {"label": "Critico", "value": "critical"},
                            {"label": "Alto", "value": "high"},
                            {"label": "Medio", "value": "medium"},
                            {"label": "Normal", "value": "low"},
                        ],
                        value=["critical", "high", "medium", "low"],
                        inline=True,
                        switch=True,
                        style={"fontSize": "0.78rem"},
                    ),
                    html.Span("Metrica", className="filter-label", style={"marginLeft": "8px"}),
                    dcc.Dropdown(
                        id="filter-metric",
                        options=[
                            {"label": v, "value": k}
                            for k, v in METRIC_LABELS.items()
                        ],
                        value="optical_power_dbm",
                        clearable=False,
                        style={"width": "200px", "fontSize": "0.8rem"},
                    ),
                    html.Div(style={"flex": "1"}),
                    html.Div(
                        [
                            html.Span(
                                "refresh",
                                className="material-symbols-outlined",
                                style={"fontSize": "16px"},
                            ),
                            html.Span(id="countdown-text", children="30s"),
                        ],
                        className="countdown-badge",
                        id="btn-refresh",
                        n_clicks=0,
                    ),
                ],
                className="filter-bar",
            ),
            dcc.Interval(id="interval-clock", interval=1000, n_intervals=0),
            dcc.Interval(id="interval-countdown", interval=1000, n_intervals=0),
        ],
        className="noc-header",
    )


def build_kpi_row(
    summary: pd.DataFrame,
    df: pd.DataFrame,
    business_metrics: dict = None,
) -> dbc.Row:
    n_segments = len(summary)
    n_critical = int((summary["risk_level"] == "critical").sum())
    n_high = int((summary["risk_level"] == "high").sum())

    bm = business_metrics or {}

    # Early detection rate - from real business metrics
    early_det = bm.get("early_detection_rate", 0.92)

    # Savings - from real business metrics
    savings = bm.get("savings_usd", 0)
    # Format large numbers nicely
    if savings >= 1_000_000:
        savings_str = f"${savings / 1_000_000:.1f}M"
    elif savings >= 1_000:
        savings_str = f"${savings:,.0f}"
    else:
        savings_str = f"${savings}"

    # MTBF
    total_hours = len(df) * 0.25
    total_faults = int(df["is_fault"].sum()) if "is_fault" in df.columns else 1
    mtbf = round(total_hours / max(total_faults, 1), 0)

    # Availability - from real business metrics if available
    sla = bm.get("sla_compliance", None)
    if sla is not None:
        availability = round(sla * 100, 2)
    else:
        downtime_h = total_faults * 4
        availability = round((total_hours - downtime_h) / max(total_hours, 1) * 100, 2)

    # Service bar pct
    service_pct = round((n_segments - n_critical - n_high) / max(n_segments, 1) * 100, 0)

    cards = [
        build_kpi_card(
            "Segmentos Activos",
            str(n_segments),
            f"En servicio: {service_pct:.0f}%",
            accent="kpi-primary",
            icon="lan",
            service_pct=service_pct,
        ),
        build_kpi_card(
            "Falla Inminente",
            str(n_critical),
            f"+{n_high} alto riesgo",
            accent="kpi-error",
            icon="warning",
            has_pulse=n_critical > 0,
        ),
        build_kpi_card(
            "Deteccion 24h SLA",
            f"{early_det * 100:.1f}%",
            f"Detectadas: {bm.get('detected_faults', '?')}/{bm.get('total_faults', '?')}",
            accent="kpi-secondary",
            icon="trending_up",
        ),
        build_kpi_card(
            "Ahorro Preventivo",
            savings_str,
            f"Ahorro {bm.get('savings_pct', 0):.0f}% vs reactivo",
            accent="kpi-tertiary",
            icon="savings",
        ),
        build_kpi_card(
            "MTBF Red GPON",
            f"{mtbf:.0f}h",
            "Mean Time Between Failures",
            accent="kpi-primary",
            icon="schedule",
        ),
        build_kpi_card(
            "Disponibilidad",
            f"{availability:.2f}%",
            f"SLA Target: {bm.get('sla_target', 0.999) * 100:.1f}% {'OK' if bm.get('sla_met', True) else 'FAIL'}",
            accent="kpi-secondary",
            icon="verified",
        ),
    ]

    return dbc.Row(
        [dbc.Col(card, xs=12, sm=6, md=4, lg=2) for card in cards],
        className="g-3 mb-3",
    )


def build_alerts_panel(summary: pd.DataFrame) -> html.Div:
    top = summary[summary["risk_level"].isin(["critical", "high"])].sort_values(
        "fault_probability", ascending=False
    ).head(5)

    if len(top) == 0:
        return html.Div(
            "Sin alertas activas",
            style={"color": OUTLINE, "textAlign": "center", "padding": "40px"},
        )

    alerts = [
        build_alert_item(
            segment_id=row["segment_id"],
            distrito=row["distrito"],
            fault_type=row["fault_type"],
            probability=row["fault_probability"],
            hours_to_fault=row["hours_to_fault"],
            risk_level=row["risk_level"],
        )
        for _, row in top.iterrows()
    ]
    return html.Div(alerts, id="alerts-list")


def build_data_table(summary: pd.DataFrame) -> dash_table.DataTable:
    table_df = summary.copy()
    table_df["ber_str"] = table_df["ber"].apply(lambda x: f"{x:.2e}")
    prob_col = "lstm_probability" if "lstm_probability" in table_df.columns else "fault_probability"
    table_df["prob_str"] = (table_df[prob_col] * 100).round(1).astype(str) + "%"
    table_df["risk_display"] = table_df["risk_level"].map(
        {"critical": "CRITICO", "high": "ALTO", "medium": "MEDIO", "low": "NORMAL"}
    )

    columns = [
        {"name": "Segmento", "id": "segment_id"},
        {"name": "Distrito", "id": "distrito"},
        {"name": "Dist (km)", "id": "distance_km", "type": "numeric"},
        {"name": "Pot Rx (dBm)", "id": "optical_power_dbm", "type": "numeric"},
        {"name": "Atten (dB/km)", "id": "attenuation_db_km", "type": "numeric"},
        {"name": "OSNR/BER", "id": "ber_str"},
        {"name": "Estado", "id": "risk_display"},
        {"name": "Prob LSTM", "id": "prob_str"},
        {"name": "H. a Falla", "id": "hours_to_fault", "type": "numeric"},
        {"name": "Tipo Falla", "id": "fault_type"},
    ]

    return dash_table.DataTable(
        id="segments-table",
        columns=columns,
        data=table_df.to_dict("records"),
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": SURFACE_1,
            "color": PRIMARY,
            "fontWeight": "600",
            "fontSize": "0.7rem",
            "textTransform": "uppercase",
            "letterSpacing": "0.5px",
            "border": f"1px solid {OUTLINE_DIM}",
        },
        style_cell={
            "backgroundColor": SURFACE_2,
            "color": ON_SURFACE,
            "border": f"1px solid {OUTLINE_DIM}",
            "fontFamily": FONT_MONO,
            "fontSize": "0.75rem",
            "padding": "8px 10px",
            "textAlign": "left",
        },
        style_data_conditional=[
            {
                "if": {"filter_query": '{risk_display} = "CRITICO"'},
                "backgroundColor": "rgba(231,76,60,0.1)",
                "borderLeft": f"3px solid {ERROR}",
            },
            {
                "if": {"filter_query": '{risk_display} = "ALTO"'},
                "backgroundColor": "rgba(245,158,11,0.06)",
            },
        ],
        style_filter={
            "backgroundColor": SURFACE_3,
            "color": ON_SURFACE,
        },
    )


# ===========================================================================
# MAIN APP FACTORY
# ===========================================================================
def create_dashboard() -> dash.Dash:
    """Crea y configura la aplicacion Dash NOC con datos reales."""

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="GPON Sentinel AI - NOC Dashboard",
        suppress_callback_exceptions=True,
    )

    # --- Load ALL data: models, predictions, evaluations ---
    logger.info("Inicializando dashboard con datos reales...")
    data = initialize_dashboard_data()

    df = data["raw_df"]
    summary = data["summary"]
    model_metrics = data["model_metrics"]
    business_metrics = data["business_metrics"] or {}
    prediction_history = data["prediction_history"]
    sidebar_metrics = data["sidebar_metrics"]

    segments = sorted(df["segment_id"].unique())

    # Source indicator
    source_label = "LSTM REAL" if data["models_loaded"] else "HEURISTICO"
    logger.info("Dashboard inicializado. Fuente de predicciones: %s", source_label)

    # Risk distribution counts
    risk_label_map = {"low": "Normal", "medium": "Medio", "high": "Alto", "critical": "Critico"}
    risk_counts = {
        risk_label_map[k]: int(v)
        for k, v in summary["risk_level"].value_counts().items()
    }

    # Summary pills from real business metrics
    bm = business_metrics
    summary_pills_data = [
        {"label": "Prevenidas", "value": str(bm.get("detected_faults", "?"))},
        {"label": "Falsos Positivos", "value": str(bm.get("false_alarms", "?"))},
        {"label": "Falsos Negativos", "value": str(bm.get("missed_faults", "?"))},
        {
            "label": "Ahorro/evento",
            "value": f"${round(bm.get('savings_usd', 0) / max(bm.get('detected_faults', 1), 1)):,}",
        },
    ]

    # --- LAYOUT ---
    app.layout = html.Div(
        [
            build_sidebar(sidebar_metrics),
            html.Div(
                [
                    build_header(segments),
                    html.Div(
                        [
                            # === KPI Row ===
                            build_kpi_row(summary, df, business_metrics),

                            # === Map (8 cols) + Alerts (4 cols) — same height ===
                            dbc.Row(
                                [
                                    dbc.Col(
                                        noc_panel(
                                            "Mapa GIS - Red Fibra Lima",
                                            html.Div(
                                                [
                                                    dcc.Graph(
                                                        id="risk-map",
                                                        figure=create_risk_map(summary),
                                                        config={"scrollZoom": True},
                                                        style={"height": "100%"},
                                                    ),
                                                    build_map_legend(),
                                                ],
                                                style={"position": "relative", "flex": "1"},
                                            ),
                                            icon="map",
                                        ),
                                        lg=8, md=12,
                                        style={"display": "flex"},
                                    ),
                                    dbc.Col(
                                        noc_panel(
                                            "Alertas Activas",
                                            html.Div(
                                                id="alerts-container",
                                                children=build_alerts_panel(summary),
                                                style={"overflowY": "auto", "flex": "1"},
                                            ),
                                            icon="notifications_active",
                                            badge=f"{int((summary['risk_level'].isin(['critical', 'high'])).sum())} ALERTAS",
                                        ),
                                        lg=4, md=12,
                                        style={"display": "flex"},
                                    ),
                                ],
                                className="g-3 mb-3 noc-row-stretch",
                                style={"height": "480px", "overflow": "hidden"},
                            ),

                            # === 2x2 Mosaic: Telemetria / Donut / Historial / Radar ===
                            dbc.Row(
                                [
                                    # Top-left: Telemetria
                                    dbc.Col(
                                        noc_panel(
                                            "Telemetria en Tiempo Real",
                                            [
                                                html.Div(id="hud-strip-container"),
                                                dcc.Graph(id="sensor-timeline"),
                                                html.Div(
                                                    [
                                                        html.Span("Linea azul: Lectura", style={"color": PRIMARY, "fontSize": "0.7rem", "marginRight": "16px"}),
                                                        html.Span("Zona roja: Falla", style={"color": ERROR, "fontSize": "0.7rem", "marginRight": "16px"}),
                                                        html.Span("Linea punteada: Umbral", style={"color": ON_SURFACE_DIM, "fontSize": "0.7rem"}),
                                                    ],
                                                    style={"padding": "4px 0"},
                                                ),
                                            ],
                                            icon="show_chart",
                                        ),
                                        lg=6, md=12,
                                        style={"display": "flex"},
                                    ),
                                    # Top-right: Distribucion de Riesgo
                                    dbc.Col(
                                        noc_panel(
                                            "Distribucion de Riesgo",
                                            [
                                                dcc.Graph(
                                                    id="risk-donut",
                                                    figure=create_risk_distribution(risk_counts),
                                                    config={"displayModeBar": False},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Span(f"Normal: {risk_counts.get('Normal', 0)}", className="text-secondary", style={"fontSize": "0.72rem", "marginRight": "12px"}),
                                                        html.Span(f"Medio: {risk_counts.get('Medio', 0)}", className="text-tertiary", style={"fontSize": "0.72rem", "marginRight": "12px"}),
                                                        html.Span(f"Alto: {risk_counts.get('Alto', 0)}", style={"fontSize": "0.72rem", "color": "#ffc174", "marginRight": "12px"}),
                                                        html.Span(f"Critico: {risk_counts.get('Critico', 0)}", className="text-error", style={"fontSize": "0.72rem"}),
                                                    ],
                                                    style={"textAlign": "center", "padding": "0 0 6px"},
                                                ),
                                            ],
                                            icon="donut_large",
                                        ),
                                        lg=6, md=12,
                                        style={"display": "flex"},
                                    ),
                                ],
                                className="g-3 mb-3 noc-row-stretch",
                            ),
                            dbc.Row(
                                [
                                    # Bottom-left: Historial de Predicciones
                                    dbc.Col(
                                        noc_panel(
                                            "Historial de Predicciones",
                                            [
                                                dcc.Graph(
                                                    id="prediction-history",
                                                    figure=create_prediction_history(prediction_history),
                                                ),
                                                build_summary_pills(summary_pills_data),
                                            ],
                                            icon="history",
                                        ),
                                        lg=6, md=12,
                                        style={"display": "flex"},
                                    ),
                                    # Bottom-right: Radar Modelos ML
                                    dbc.Col(
                                        noc_panel(
                                            "Radar Modelos ML",
                                            [
                                                dcc.Graph(
                                                    id="model-radar",
                                                    figure=create_model_radar(model_metrics),
                                                    config={"displayModeBar": False},
                                                ),
                                                html.Div(
                                                    [
                                                        html.Span(
                                                            "LSTM LIDER" if data["models_loaded"] else "DATOS DEMO",
                                                            className="header-badge",
                                                            style={"backgroundColor": "rgba(56,189,248,0.15)"},
                                                        ),
                                                        html.Span(
                                                            f" {source_label}",
                                                            style={"fontSize": "0.6rem", "color": OUTLINE, "marginLeft": "8px"},
                                                        ),
                                                    ],
                                                    style={"textAlign": "center"},
                                                ),
                                            ],
                                            icon="radar",
                                            badge="4 MODELOS",
                                        ),
                                        lg=6, md=12,
                                        style={"display": "flex"},
                                    ),
                                ],
                                className="g-3 mb-3 noc-row-stretch",
                            ),

                            # === Segments Table (12 cols) ===
                            dbc.Row(
                                dbc.Col(
                                    noc_panel(
                                        "Tabla de Segmentos",
                                        [
                                            html.Div(
                                                [
                                                    dcc.Input(
                                                        id="table-search",
                                                        type="text",
                                                        placeholder="Buscar segmento...",
                                                        className="table-search",
                                                        debounce=True,
                                                    ),
                                                    dcc.Dropdown(
                                                        id="table-olt-filter",
                                                        options=[{"label": "Todos los distritos", "value": "all"}]
                                                        + [{"label": d, "value": d} for d in sorted(summary["distrito"].unique())],
                                                        value="all",
                                                        clearable=False,
                                                        style={"width": "200px", "fontSize": "0.8rem"},
                                                    ),
                                                ],
                                                style={"display": "flex", "gap": "12px", "marginBottom": "12px", "alignItems": "center"},
                                            ),
                                            html.Div(id="table-container", children=build_data_table(summary)),
                                        ],
                                        icon="table_chart",
                                        badge=f"{len(summary)} SEGMENTOS",
                                    ),
                                    width=12,
                                ),
                                className="g-3 mb-4",
                            ),
                        ],
                        className="noc-content",
                    ),
                ],
                className="noc-main",
            ),
            dcc.Store(id="countdown-store", data=30),
        ]
    )

    # ===================================================================
    # CALLBACKS
    # ===================================================================

    @app.callback(
        Output("lima-clock", "children"),
        Input("interval-clock", "n_intervals"),
    )
    def update_clock(_n):
        now = datetime.now(LIMA_TZ)
        return now.strftime("%d %b %Y  %H:%M:%S") + "  PET"

    @app.callback(
        [Output("countdown-store", "data"), Output("countdown-text", "children")],
        [Input("interval-countdown", "n_intervals"), Input("btn-refresh", "n_clicks")],
        State("countdown-store", "data"),
    )
    def update_countdown(n_intervals, n_clicks, current):
        ctx = dash.callback_context
        if ctx.triggered and ctx.triggered[0]["prop_id"] == "btn-refresh.n_clicks":
            return 30, "30s"
        val = max(0, current - 1) if current > 0 else 30
        return val, f"{val}s"

    @app.callback(
        [Output("sensor-timeline", "figure"), Output("hud-strip-container", "children")],
        [Input("filter-segment", "value"), Input("filter-metric", "value")],
    )
    def update_timeline(segment_id, metric):
        if not segment_id or not metric:
            return dash.no_update, dash.no_update

        seg_data = df[df["segment_id"] == segment_id].sort_values("timestamp")
        fig = create_sensor_timeline(seg_data, metric, segment_id)

        last = seg_data.iloc[-1] if len(seg_data) > 0 else {}
        hud_data = {
            "Potencia": f"{last.get('optical_power_dbm', 0):.1f} dBm",
            "Atenuacion": f"{last.get('attenuation_db_km', 0):.3f} dB/km",
            "OSNR": f"{last.get('osnr_db', 0):.1f} dB",
            "BER": f"{last.get('ber', 0):.1e}",
            "Temp": f"{last.get('temperature', 0):.1f} C",
            "Humedad": f"{last.get('humidity', 0):.0f}%",
            "Disp.Crom": f"{last.get('chromatic_dispersion', 0):.1f}",
        }
        return fig, build_hud_strip(hud_data)

    @app.callback(
        Output("table-container", "children"),
        [
            Input("filter-risk", "value"),
            Input("table-search", "value"),
            Input("table-olt-filter", "value"),
        ],
    )
    def filter_table(risk_levels, search_text, distrito_filter):
        filtered = summary.copy()
        if risk_levels:
            filtered = filtered[filtered["risk_level"].isin(risk_levels)]
        if search_text:
            filtered = filtered[filtered["segment_id"].str.contains(search_text, case=False, na=False)]
        if distrito_filter and distrito_filter != "all":
            filtered = filtered[filtered["distrito"] == distrito_filter]
        return build_data_table(filtered)

    @app.callback(
        [Output("risk-map", "figure"), Output("alerts-container", "children")],
        Input("filter-risk", "value"),
    )
    def filter_map_alerts(risk_levels):
        if not risk_levels:
            risk_levels = ["critical", "high", "medium", "low"]
        filtered = summary[summary["risk_level"].isin(risk_levels)]
        return create_risk_map(filtered), build_alerts_panel(filtered)

    return app


# Module-level instance used by run_dashboard.py
dash_app = create_dashboard()
