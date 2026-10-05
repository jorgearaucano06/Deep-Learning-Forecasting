"""
Funciones de Visualizacion Reutilizables
==========================================
Funciones de ploteo estandarizadas para el proyecto.

Centralizamos las visualizaciones para:
1. Mantener un estilo visual consistente en todos los notebooks.
2. Evitar codigo duplicado (DRY - Don't Repeat Yourself).
3. Facilitar cambios de estilo en un solo lugar.

Uso:
    from src.utils.visualization import plot_sensor_timeline, plot_confusion_matrix
    plot_sensor_timeline(df, segment_id="SEG-001")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import List, Optional, Dict

# Estilo base para todo el proyecto
STYLE_CONFIG = {
    "figure.figsize": (14, 6),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.size": 10,
}

# Colores por tipo de falla
FAULT_COLORS = {
    "normal": "#2ecc71",
    "physical_cut": "#e74c3c",
    "degradation": "#f39c12",
    "bad_splice": "#9b59b6",
}

METRIC_LABELS = {
    "optical_power_dbm": "Potencia Optica (dBm)",
    "attenuation_db_km": "Atenuacion (dB/km)",
    "ber": "Bit Error Rate",
    "osnr_db": "OSNR (dB)",
    "chromatic_dispersion": "Dispersion Cromatica",
    "temperature": "Temperatura (C)",
    "humidity": "Humedad (%)",
}


def apply_style():
    """Aplica el estilo base del proyecto."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(STYLE_CONFIG)


def plot_sensor_timeline(
    df: pd.DataFrame,
    segment_id: str,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Grafica la linea temporal de metricas de un segmento.

    Args:
        df: DataFrame con datos de sensores.
        segment_id: ID del segmento a visualizar (ej: "SEG-001").
        metrics: Lista de metricas a graficar. Si None, usa las 5 principales.
        figsize: Tamano de la figura.

    Returns:
        Figura de matplotlib.
    """
    apply_style()

    if metrics is None:
        metrics = ["optical_power_dbm", "attenuation_db_km", "ber", "osnr_db", "chromatic_dispersion"]

    seg_data = df[df["segment_id"] == segment_id].sort_values("timestamp")
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

    if n_metrics == 1:
        axes = [axes]

    fault_type = "normal"
    if seg_data["is_fault"].any():
        fault_type = seg_data.loc[seg_data["is_fault"] == 1, "fault_type"].iloc[0]

    fig.suptitle(
        f"Segmento {segment_id} | Estado: {fault_type}",
        fontsize=14, fontweight="bold",
    )

    for ax, metric in zip(axes, metrics):
        ax.plot(seg_data["timestamp"], seg_data[metric], linewidth=0.8, alpha=0.85, color="steelblue")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))

        # Zona de falla
        fault_mask = seg_data["is_fault"] == 1
        if fault_mask.any():
            fault_start = seg_data.loc[fault_mask, "timestamp"].iloc[0]
            ax.axvline(x=fault_start, color="red", linestyle="--", alpha=0.7, label="Falla")

        # Zona pre-falla
        pre_mask = seg_data["pre_fault_hours"] > 0
        if pre_mask.any():
            pre_start = seg_data.loc[pre_mask, "timestamp"].iloc[0]
            ax.axvline(x=pre_start, color="orange", linestyle="--", alpha=0.7, label="Pre-falla")

        ax.legend(loc="upper right", fontsize=8)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_distribution_by_fault(
    df: pd.DataFrame,
    metric: str,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Compara la distribucion de una metrica entre tipos de falla.

    Util para entender como cambian las metricas segun el estado del segmento.
    """
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Histograma
    for fault_type, color in FAULT_COLORS.items():
        data = df[df["fault_type"] == fault_type][metric]
        if len(data) > 0:
            axes[0].hist(data, bins=50, alpha=0.5, color=color, label=fault_type, density=True)

    axes[0].set_xlabel(METRIC_LABELS.get(metric, metric))
    axes[0].set_ylabel("Densidad")
    axes[0].set_title(f"Distribucion de {METRIC_LABELS.get(metric, metric)}")
    axes[0].legend()

    # Box plot
    fault_types_present = df["fault_type"].unique()
    colors = [FAULT_COLORS.get(ft, "#95a5a6") for ft in sorted(fault_types_present)]
    box_data = [df[df["fault_type"] == ft][metric].values for ft in sorted(fault_types_present)]
    bp = axes[1].boxplot(box_data, labels=sorted(fault_types_present), patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    axes[1].set_ylabel(METRIC_LABELS.get(metric, metric))
    axes[1].set_title(f"Box Plot por Tipo de Falla")

    plt.tight_layout()
    return fig


def plot_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    Matriz de correlacion entre metricas.

    La correlacion mide la relacion lineal entre variables (-1 a 1).
    Valores cercanos a +/-1 indican fuerte relacion.
    """
    apply_style()

    if columns is None:
        columns = ["optical_power_dbm", "attenuation_db_km", "ber", "osnr_db",
                    "chromatic_dispersion", "temperature", "humidity"]

    numeric_cols = [c for c in columns if c in df.columns]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # Triangulo superior
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Matriz de Correlacion", fontsize=14)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Matriz de Confusion",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    Visualiza una matriz de confusion.

    Args:
        cm: Matriz de confusion (numpy array NxN).
        class_names: Nombres de las clases.
        title: Titulo del grafico.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    metric_name: str = "Loss",
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Grafica las curvas de entrenamiento y validacion.

    Buscar:
    - Convergencia: ambas curvas bajan y se estabilizan.
    - Overfitting: train baja pero val sube (gap creciente).
    - Underfitting: ambas se estabilizan en valores altos.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", label=f"Train {metric_name}", linewidth=2)
    ax.plot(epochs, val_losses, "r-", label=f"Val {metric_name}", linewidth=2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Curvas de Entrenamiento - {metric_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Compara metricas entre multiples modelos con barras agrupadas.

    Args:
        results: Dict de {nombre_modelo: {metrica: valor}}.
        metrics: Metricas a comparar. Si None, usa todas.
    """
    apply_style()

    if metrics is None:
        metrics = list(next(iter(results.values())).keys())

    n_models = len(results)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (model_name, model_metrics) in enumerate(results.items()):
        values = [model_metrics.get(m, 0) for m in metrics]
        ax.bar(x + i * width, values, width, label=model_name, color=colors[i], alpha=0.8)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Valor")
    ax.set_title("Comparacion de Modelos")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig
