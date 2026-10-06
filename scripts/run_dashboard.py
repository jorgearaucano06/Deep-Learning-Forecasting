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
    port = config["dashboard"]["port"]
    print(f"\n  GPON Sentinel AI - NOC Dashboard")
    print(f"  Abrir en navegador: http://localhost:{port}\n")
    dash_app.run(
        host="127.0.0.1",
        port=port,
        debug=True,
        use_reloader=False,
    )


if __name__ == "__main__":
    main()
