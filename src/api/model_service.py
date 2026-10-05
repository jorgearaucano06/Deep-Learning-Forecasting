"""
Servicio de Modelos para la API
=================================
Carga modelos entrenados y realiza inferencia.

Patron Singleton: Solo se carga el modelo UNA vez en memoria.
Las requests subsiguientes reutilizan el mismo modelo.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from src.models.lstm_predictor import LSTMPredictor
from src.utils.config_loader import load_config, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelService:
    """
    Servicio que gestiona la carga e inferencia de modelos.

    En produccion, el modelo se carga al iniciar la API y se mantiene
    en memoria para responder requests rapidamente (<100ms).
    """

    def __init__(self):
        self.config = load_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        self.feature_cols = [
            "optical_power_dbm", "attenuation_db_km", "ber", "osnr_db",
            "chromatic_dispersion", "temperature", "humidity",
        ]

    def load_model(self, model_path: str = None) -> bool:
        """
        Carga el modelo LSTM desde disco.

        Args:
            model_path: Ruta al archivo .pt. Si None, busca en models/lstm_best.pt.

        Returns:
            True si se cargo exitosamente.
        """
        if model_path is None:
            model_path = str(get_path("models") / "lstm_best.pt")

        try:
            cfg = self.config["models"]["lstm_predictor"]
            self.model = LSTMPredictor(
                n_features=len(self.feature_cols),
                hidden_size=cfg["hidden_size"],
                num_layers=cfg["num_layers"],
                dropout=cfg["dropout"],
                bidirectional=cfg["bidirectional"],
            ).to(self.device)

            if Path(model_path).exists():
                self.model.load_state_dict(
                    torch.load(model_path, weights_only=True, map_location=self.device)
                )
                logger.info("Modelo cargado desde %s", model_path)
            else:
                logger.warning("Modelo no encontrado en %s. Usando pesos aleatorios.", model_path)

            self.model.eval()
            self.model_loaded = True
            return True

        except Exception as e:
            logger.error("Error cargando modelo: %s", e)
            self.model_loaded = False
            return False

    def predict(self, readings: List[Dict]) -> Dict:
        """
        Realiza prediccion para una lista de lecturas de sensor.

        Args:
            readings: Lista de dicts con las metricas del sensor.
                     Idealmente 48 lecturas (12 horas de historia).

        Returns:
            Dict con probabilidad de falla, clase, riesgo y recomendaciones.
        """
        if not self.model_loaded:
            self.load_model()

        # Convertir lecturas a tensor
        features = []
        for reading in readings:
            row = [reading.get(col, 0.0) for col in self.feature_cols]
            features.append(row)

        # Padding si hay menos de 48 lecturas
        seq_len = self.config["preprocessing"]["sequence_length"]
        while len(features) < seq_len:
            features.insert(0, features[0])  # Repetir primera lectura

        features = features[-seq_len:]  # Tomar ultimas 48

        # Crear tensor: (1, seq_len, n_features)
        x = torch.FloatTensor([features]).to(self.device)

        # Inferencia
        with torch.no_grad():
            output = self.model(x)
            probability = torch.sigmoid(output).item()

        # Clasificar riesgo
        risk_level, fault_class, recommendations = self._classify_risk(probability, readings[-1])

        return {
            "fault_probability": round(probability, 4),
            "fault_class": fault_class,
            "risk_level": risk_level,
            "confidence": round(min(abs(probability - 0.5) * 2, 1.0), 4),
            "recommendations": recommendations,
        }

    def _classify_risk(self, probability: float, last_reading: Dict):
        """
        Clasifica el nivel de riesgo y genera recomendaciones.

        Umbrales:
        - <0.2: Low - Operacion normal.
        - 0.2-0.5: Medium - Monitorear mas de cerca.
        - 0.5-0.8: High - Programar inspeccion.
        - >0.8: Critical - Accion inmediata requerida.
        """
        if probability < 0.2:
            risk = "low"
            fault_class = "normal"
            recs = ["Operacion normal. Continuar monitoreo estandar."]
        elif probability < 0.5:
            risk = "medium"
            fault_class = "degradation"
            recs = [
                "Incrementar frecuencia de monitoreo.",
                "Programar inspeccion preventiva en proximos 7 dias.",
            ]
        elif probability < 0.8:
            risk = "high"
            fault_class = "degradation"
            recs = [
                "Programar inspeccion en proximas 24 horas.",
                "Preparar equipo de mantenimiento.",
                "Verificar empalmes y conectores del segmento.",
            ]
        else:
            risk = "critical"
            fault_class = "physical_cut"
            recs = [
                "ACCION INMEDIATA: Riesgo critico de falla.",
                "Despachar equipo de emergencia.",
                "Activar ruta alternativa de fibra.",
                "Notificar a clientes afectados.",
            ]

        # Ajustar clase basado en metricas
        power = last_reading.get("optical_power_dbm", -5)
        if power < -15:
            fault_class = "physical_cut"
        elif last_reading.get("attenuation_db_km", 0.2) > 0.4:
            fault_class = "bad_splice"

        return risk, fault_class, recs


# Singleton global
_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Retorna la instancia singleton del servicio."""
    global _service_instance
    if _service_instance is None:
        _service_instance = ModelService()
    return _service_instance
