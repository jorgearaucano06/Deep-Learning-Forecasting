"""
Schemas Pydantic para la API
===============================
Pydantic valida automaticamente los datos de entrada/salida de la API.

Si alguien envia un JSON con campos incorrectos o tipos invalidos,
Pydantic retorna un error claro sin que tengas que escribir validacion manual.

Ejemplo:
    Si el endpoint espera {"optical_power_dbm": -5.0} y recibe {"power": "abc"},
    Pydantic retorna: "field required: optical_power_dbm" + "not a valid float".
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class SensorReading(BaseModel):
    """
    Una lectura individual de un sensor de fibra optica.
    Field(...) con descripcion ayuda a generar documentacion automatica en Swagger.
    """
    optical_power_dbm: float = Field(..., description="Potencia optica en dBm")
    attenuation_db_km: float = Field(..., description="Atenuacion en dB/km")
    ber: float = Field(..., description="Bit Error Rate")
    osnr_db: float = Field(..., description="Signal-to-Noise Ratio en dB")
    chromatic_dispersion: float = Field(..., description="Dispersion cromatica")
    temperature: float = Field(25.0, description="Temperatura en Celsius")
    humidity: float = Field(70.0, description="Humedad relativa %")

    class Config:
        json_schema_extra = {
            "example": {
                "optical_power_dbm": -5.0,
                "attenuation_db_km": 0.22,
                "ber": 1e-10,
                "osnr_db": 30.0,
                "chromatic_dispersion": 1.2,
                "temperature": 25.0,
                "humidity": 70.0,
            }
        }


class PredictionRequest(BaseModel):
    """Request para prediccion de un segmento."""
    segment_id: str = Field(..., description="ID del segmento (ej: SEG-001)")
    readings: List[SensorReading] = Field(
        ..., description="Lista de lecturas de sensores (min 1, idealmente 48)"
    )


class PredictionResponse(BaseModel):
    """Respuesta de prediccion."""
    segment_id: str
    fault_probability: float = Field(..., description="Probabilidad de falla en 24h (0-1)")
    fault_class: str = Field(..., description="Clase predicha: normal, physical_cut, degradation, bad_splice")
    risk_level: str = Field(..., description="Nivel de riesgo: low, medium, high, critical")
    confidence: float = Field(..., description="Confianza de la prediccion (0-1)")
    recommendations: List[str] = Field(default=[], description="Acciones recomendadas")
    timestamp: datetime = Field(default_factory=datetime.now)


class BatchPredictionRequest(BaseModel):
    """Request para prediccion en lote."""
    segments: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Respuesta de prediccion en lote."""
    predictions: List[PredictionResponse]
    summary: Dict = Field(default={}, description="Resumen de riesgo de la red")


class HealthResponse(BaseModel):
    """Respuesta del health check."""
    status: str
    model_loaded: bool
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
