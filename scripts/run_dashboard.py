"""
Script para ejecutar el dashboard Plotly Dash.

Uso:
    python scripts/run_dashboard.py

Abrir:
    http://localhost:8050
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dashboard.app import dash_app
from src.utils.config_loader import load_config


def main():
    config = load_config()
    dash_cfg = config["dashboard"]
    dash_app.run(
        host=dash_cfg["host"],
        port=dash_cfg["port"],
        debug=True,
    )


if __name__ == "__main__":
    main()
