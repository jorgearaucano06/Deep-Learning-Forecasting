"""
Sistema de Logging
==================
Configura logging consistente para todo el proyecto.

Por que usar logging en vez de print()?
- print() va a stdout y se pierde; logging puede ir a archivo + consola.
- Puedes filtrar por nivel: DEBUG, INFO, WARNING, ERROR, CRITICAL.
- Cada mensaje incluye timestamp y origen (que modulo lo genero).

Uso:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    logger.info("Entrenamiento iniciado")
    logger.warning("Loss no decrece hace 5 epochs")
    logger.error("Archivo no encontrado: %s", path)
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Crea y configura un logger con formato consistente.

    Args:
        name: Nombre del logger (usa __name__ para auto-identificar el modulo).
        level: Nivel minimo de logging (default: INFO).
               DEBUG < INFO < WARNING < ERROR < CRITICAL

    Returns:
        Logger configurado listo para usar.

    Ejemplo:
        logger = get_logger(__name__)
        logger.info("Procesando segmento %s", segment_id)
    """
    logger = logging.getLogger(name)

    # Evitar agregar handlers duplicados si se llama multiples veces
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Formato: timestamp - modulo - nivel - mensaje
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Handler para consola (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger
