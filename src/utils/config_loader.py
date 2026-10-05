"""
Cargador de Configuracion YAML
===============================
Este modulo centraliza la carga del archivo config.yaml.

Patron de diseno: Singleton-like con cache.
- La primera vez que llamas a load_config(), lee el archivo YAML del disco.
- Las siguientes llamadas devuelven la misma configuracion cacheada (rapido).
- Esto evita leer el disco multiples veces en un mismo proceso.

Uso:
    from src.utils.config_loader import load_config
    config = load_config()
    print(config['models']['lstm_predictor']['hidden_size'])  # 128
"""

import os
from pathlib import Path
from typing import Any, Dict

import yaml


# Variable de modulo que actua como cache
_config_cache: Dict[str, Any] | None = None


def get_project_root() -> Path:
    """
    Encuentra la raiz del proyecto subiendo desde este archivo.

    Este archivo esta en: proyecto/src/utils/config_loader.py
    La raiz esta 3 niveles arriba: proyecto/

    Returns:
        Path: Ruta absoluta a la raiz del proyecto
    """
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    """
    Carga la configuracion desde config.yaml.

    Args:
        config_path: Ruta opcional al archivo YAML.
                     Si no se proporciona, busca en config/config.yaml

    Returns:
        Diccionario con toda la configuracion del proyecto.

    Ejemplo:
        config = load_config()
        lr = config['models']['lstm_predictor']['learning_rate']
    """
    global _config_cache

    if _config_cache is not None and config_path is None:
        return _config_cache

    if config_path is None:
        config_path = str(get_project_root() / "config" / "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config_path is None or config_path == str(get_project_root() / "config" / "config.yaml"):
        _config_cache = config

    return config


def get_path(key: str) -> Path:
    """
    Obtiene una ruta del proyecto como Path absoluto.

    Args:
        key: Clave dentro de config['paths'], por ejemplo 'raw_data'

    Returns:
        Path absoluto al directorio solicitado.

    Ejemplo:
        raw_path = get_path('raw_data')  # -> /proyecto/data/raw
    """
    config = load_config()
    relative_path = config["paths"][key]
    absolute_path = get_project_root() / relative_path

    # Crear el directorio si no existe
    absolute_path.mkdir(parents=True, exist_ok=True)

    return absolute_path
