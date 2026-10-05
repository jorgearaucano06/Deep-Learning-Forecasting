"""
Script CLI para generar datos sinteticos.

Uso:
    python scripts/generate_data.py
"""

import sys
from pathlib import Path

# Agregar raiz del proyecto al path para imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generator import FiberNetworkGenerator
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Iniciando generacion de datos sinteticos...")
    generator = FiberNetworkGenerator()
    df, path = generator.generate_and_save()

    # Resumen
    logger.info("--- Resumen de datos generados ---")
    logger.info("Total filas: %d", len(df))
    logger.info("Segmentos: %d", df["segment_id"].nunique())
    logger.info("Rango temporal: %s a %s", df["timestamp"].min(), df["timestamp"].max())
    logger.info("Distribucion de fallas:")
    for fault, count in df["fault_type"].value_counts().items():
        logger.info("  %s: %d (%.1f%%)", fault, count, 100 * count / len(df))
    logger.info("Archivo guardado: %s", path)


if __name__ == "__main__":
    main()
