"""
Endpoints REST de la API
==========================
Define las rutas de la API de prediccion.

Endpoints:
- GET  /api/v1/health           - Health check
- POST /api/v1/predict/{id}     - Prediccion por segmento
- POST /api/v1/predict/batch    - Prediccion en lote

REST API best practices aplicadas:
- Versionamiento en la URL (/api/v1/)
- Codigos HTTP semanticos (200, 422, 500)
- Documentacion automatica (Swagger UI en /docs)
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import List

from src.api.schemas import (
    SensorReading,
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from src.api.model_service import get_model_service
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check del servicio.
    Util para load balancers y monitoreo.
    """
    service = get_model_service()
    return HealthResponse(
        status="healthy" if service.model_loaded else "degraded",
        model_loaded=service.model_loaded,
        version="1.0.0",
    )


@router.post("/predict/{segment_id}", response_model=PredictionResponse)
async def predict_segment(segment_id: str, readings: List[SensorReading]):
    """
    Predice probabilidad de falla para un segmento de fibra.

    Args:
        segment_id: Identificador del segmento (ej: SEG-001).
        readings: Lista de lecturas de sensores (idealmente 48 = 12 horas).

    Returns:
        Prediccion con probabilidad, clase de falla, riesgo y recomendaciones.
    """
    service = get_model_service()
    if not service.model_loaded:
        service.load_model()

    try:
        readings_dicts = [r.model_dump() for r in readings]
        result = service.predict(readings_dicts)

        return PredictionResponse(
            segment_id=segment_id,
            **result,
        )

    except Exception as e:
        logger.error("Error en prediccion para %s: %s", segment_id, e)
        raise HTTPException(status_code=500, detail=f"Error en prediccion: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Prediccion en lote para multiples segmentos.

    Util para evaluar toda la red de una vez.
    """
    service = get_model_service()
    if not service.model_loaded:
        service.load_model()

    predictions = []
    risk_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

    for segment in request.segments:
        try:
            readings_dicts = [r.model_dump() for r in segment.readings]
            result = service.predict(readings_dicts)

            pred = PredictionResponse(segment_id=segment.segment_id, **result)
            predictions.append(pred)
            risk_counts[result["risk_level"]] += 1

        except Exception as e:
            logger.error("Error en batch para %s: %s", segment.segment_id, e)

    summary = {
        "total_segments": len(predictions),
        "risk_distribution": risk_counts,
        "critical_segments": [
            p.segment_id for p in predictions if p.risk_level == "critical"
        ],
    }

    return BatchPredictionResponse(predictions=predictions, summary=summary)
