"""
Metricas de Negocio
=====================
Traduce metricas tecnicas de ML a KPIs de negocio comprensibles
para stakeholders no-tecnicos.

En una entrevista para YOFC, lo que importa NO es el AUC del modelo,
sino CUANTO DINERO ahorra y CUANTO DOWNTIME evita.

KPIs principales:
- Costo de downtime evitado ($): Cada hora de caida cuesta dinero.
- Deteccion temprana (horas): Cuanto antes detectamos vs sin modelo.
- SLA compliance (%): Porcentaje de cumplimiento del acuerdo de servicio.
- ROI del sistema: Retorno sobre la inversion en el sistema predictivo.
"""

import numpy as np
from typing import Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Constantes de negocio (valores tipicos de la industria telco)
COST_PER_HOUR_DOWNTIME_USD = 5000   # Costo por hora de caida
COST_PREVENTIVE_MAINTENANCE_USD = 200  # Costo de mantenimiento preventivo
COST_EMERGENCY_REPAIR_USD = 3000     # Costo de reparacion de emergencia
AVG_REPAIR_TIME_HOURS = 4            # Tiempo promedio de reparacion
SLA_UPTIME_TARGET = 0.999           # 99.9% de uptime (three nines)


def calculate_business_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pre_fault_hours: np.ndarray = None,
    prediction_horizon: int = 24,
) -> Dict:
    """
    Calcula KPIs de negocio a partir de predicciones.

    Args:
        y_true: Labels reales (0=normal, 1=falla).
        y_pred: Predicciones del modelo (0=normal, 1=falla).
        pre_fault_hours: Horas antes de la falla para cada registro.
        prediction_horizon: Horizonte de prediccion en horas.

    Returns:
        Dict con metricas de negocio.
    """
    # Metricas basicas
    true_positives = int(((y_pred == 1) & (y_true == 1)).sum())
    false_positives = int(((y_pred == 1) & (y_true == 0)).sum())
    false_negatives = int(((y_pred == 0) & (y_true == 1)).sum())
    true_negatives = int(((y_pred == 0) & (y_true == 0)).sum())

    total_faults = int(y_true.sum())
    detected_faults = true_positives

    # --- Deteccion temprana ---
    early_detection_rate = detected_faults / max(total_faults, 1)
    avg_early_detection_hours = 0.0
    if pre_fault_hours is not None and detected_faults > 0:
        detected_mask = (y_pred == 1) & (y_true == 1)
        avg_early_detection_hours = float(np.mean(pre_fault_hours[detected_mask]))

    # --- Costos ---
    # Fallas detectadas a tiempo -> mantenimiento preventivo (barato)
    cost_preventive = detected_faults * COST_PREVENTIVE_MAINTENANCE_USD

    # Fallas NO detectadas -> reparacion de emergencia + downtime
    cost_emergency = false_negatives * COST_EMERGENCY_REPAIR_USD
    cost_downtime = false_negatives * AVG_REPAIR_TIME_HOURS * COST_PER_HOUR_DOWNTIME_USD

    # Falsas alarmas -> mantenimiento innecesario (pero barato)
    cost_false_alarms = false_positives * COST_PREVENTIVE_MAINTENANCE_USD * 0.5

    # Costo total CON modelo
    total_cost_with_model = cost_preventive + cost_emergency + cost_downtime + cost_false_alarms

    # Costo total SIN modelo (todas las fallas son emergencias)
    total_cost_without_model = total_faults * (COST_EMERGENCY_REPAIR_USD + AVG_REPAIR_TIME_HOURS * COST_PER_HOUR_DOWNTIME_USD)

    # Ahorro
    savings = total_cost_without_model - total_cost_with_model
    savings_pct = savings / max(total_cost_without_model, 1) * 100

    # --- Downtime ---
    downtime_prevented_hours = detected_faults * AVG_REPAIR_TIME_HOURS
    downtime_remaining_hours = false_negatives * AVG_REPAIR_TIME_HOURS

    # --- SLA ---
    total_hours = len(y_true) * 0.25  # Cada registro = 15 min
    uptime_hours = total_hours - downtime_remaining_hours
    sla_compliance = uptime_hours / max(total_hours, 1)

    metrics = {
        # Deteccion
        "total_faults": total_faults,
        "detected_faults": detected_faults,
        "missed_faults": false_negatives,
        "false_alarms": false_positives,
        "early_detection_rate": round(early_detection_rate, 4),
        "avg_early_detection_hours": round(avg_early_detection_hours, 1),
        # Costos (USD)
        "cost_with_model_usd": round(total_cost_with_model),
        "cost_without_model_usd": round(total_cost_without_model),
        "savings_usd": round(savings),
        "savings_pct": round(savings_pct, 1),
        # Downtime
        "downtime_prevented_hours": round(downtime_prevented_hours, 1),
        "downtime_remaining_hours": round(downtime_remaining_hours, 1),
        # SLA
        "sla_compliance": round(sla_compliance, 6),
        "sla_target": SLA_UPTIME_TARGET,
        "sla_met": sla_compliance >= SLA_UPTIME_TARGET,
    }

    logger.info("--- KPIs de Negocio ---")
    logger.info("Fallas detectadas: %d/%d (%.1f%%)", detected_faults, total_faults, early_detection_rate * 100)
    logger.info("Ahorro estimado: $%d (%.1f%%)", savings, savings_pct)
    logger.info("Downtime prevenido: %.1f horas", downtime_prevented_hours)
    logger.info("SLA: %.4f%% (target: %.1f%%)", sla_compliance * 100, SLA_UPTIME_TARGET * 100)

    return metrics


def format_business_report(metrics: Dict) -> str:
    """
    Formatea las metricas de negocio como reporte legible.
    """
    return f"""
{'='*60}
REPORTE DE IMPACTO DE NEGOCIO
{'='*60}

DETECCION DE FALLAS
  Total fallas:           {metrics['total_faults']}
  Detectadas a tiempo:    {metrics['detected_faults']} ({metrics['early_detection_rate']*100:.1f}%)
  No detectadas:          {metrics['missed_faults']}
  Falsas alarmas:         {metrics['false_alarms']}
  Anticipacion promedio:  {metrics['avg_early_detection_hours']:.1f} horas

IMPACTO FINANCIERO
  Costo SIN modelo:       ${metrics['cost_without_model_usd']:,}
  Costo CON modelo:       ${metrics['cost_with_model_usd']:,}
  Ahorro estimado:        ${metrics['savings_usd']:,} ({metrics['savings_pct']:.1f}%)

DISPONIBILIDAD
  Downtime prevenido:     {metrics['downtime_prevented_hours']:.1f} horas
  Downtime restante:      {metrics['downtime_remaining_hours']:.1f} horas
  SLA compliance:         {metrics['sla_compliance']*100:.3f}%
  SLA target:             {metrics['sla_target']*100:.1f}%
  SLA cumplido:           {'SI' if metrics['sla_met'] else 'NO'}
{'='*60}
"""
