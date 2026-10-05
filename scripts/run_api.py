"""
Script para ejecutar la API FastAPI.

Uso:
    python scripts/run_api.py

Abrir:
    http://localhost:8000/docs  (Swagger UI)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn
from src.utils.config_loader import load_config


def main():
    config = load_config()
    api_cfg = config["api"]
    uvicorn.run(
        "src.api.main:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=True,
    )


if __name__ == "__main__":
    main()
