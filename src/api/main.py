"""
FastAPI Application
=====================
Punto de entrada de la API REST.

FastAPI genera automaticamente:
- Documentacion interactiva en /docs (Swagger UI)
- Validacion de datos con Pydantic
- Serializacion JSON automatica
- OpenAPI schema en /openapi.json
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.api.model_service import get_model_service
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_app() -> FastAPI:
    """Factory para crear la aplicacion FastAPI."""

    app = FastAPI(
        title="Fiber Optic Predictive Maintenance API",
        description=(
            "API de mantenimiento predictivo para redes de fibra optica. "
            "Predice fallas con 24 horas de anticipacion usando Deep Learning."
        ),
        version="1.0.0",
        docs_url="/docs",      # Swagger UI
        redoc_url="/redoc",    # ReDoc (alternativa)
    )

    # CORS: Permite requests desde el dashboard (u otro frontend)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Registrar rutas
    app.include_router(router)

    @app.on_event("startup")
    async def startup():
        """Se ejecuta al iniciar la API. Carga el modelo en memoria."""
        logger.info("Iniciando API...")
        service = get_model_service()
        service.load_model()
        logger.info("API lista. Modelo cargado: %s", service.model_loaded)

    @app.get("/")
    async def root():
        return {
            "service": "Fiber Optic Predictive Maintenance",
            "version": "1.0.0",
            "docs": "/docs",
        }

    return app


# Instancia global para uvicorn
app = create_app()
