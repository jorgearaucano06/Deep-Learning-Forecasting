"""
Orquestador de Experimentos con MLflow
========================================
Gestiona el ciclo de vida de experimentos de ML.

MLflow organiza experimentos en:
- Experiment: Agrupacion logica (ej: "fiber_optic_predictive_maintenance").
- Run: Una ejecucion individual de un modelo con hiperparametros especificos.
- Cada Run registra: parametros, metricas, artefactos (modelos, graficos).

Beneficios:
- Reproducibilidad: Puedes recrear cualquier experimento.
- Comparacion: UI web para comparar runs lado a lado.
- Versionamiento: Cada modelo queda registrado con su version.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from src.utils.config_loader import load_config, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExperimentManager:
    """
    Gestiona experimentos con MLflow.

    Uso:
        em = ExperimentManager()
        with em.start_run("lstm_v1", params={...}):
            # entrenar modelo
            em.log_metrics({"loss": 0.1, "accuracy": 0.95})
            em.log_model(model, "lstm_model")
    """

    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.mlflow_cfg = self.config["mlflow"]
        self._mlflow = None
        self._current_run = None

    def _init_mlflow(self):
        """Inicializa MLflow de forma lazy (solo cuando se necesita)."""
        if self._mlflow is None:
            try:
                import mlflow
                self._mlflow = mlflow
                tracking_uri = str(get_path("mlruns"))
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(self.mlflow_cfg["experiment_name"])
                logger.info("MLflow inicializado: %s", tracking_uri)
            except ImportError:
                logger.warning("MLflow no instalado. Logging deshabilitado.")

    def start_run(self, run_name: str, params: Dict[str, Any] = None):
        """
        Inicia un run de MLflow.

        Args:
            run_name: Nombre descriptivo del run.
            params: Hiperparametros a registrar.

        Returns:
            self (para uso como context manager).
        """
        self._init_mlflow()
        if self._mlflow is None:
            return self

        self._current_run = self._mlflow.start_run(run_name=run_name)
        if params:
            self._mlflow.log_params(params)
        logger.info("MLflow run iniciado: %s", run_name)
        return self

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Registra metricas en el run actual."""
        if self._mlflow is None:
            return
        self._mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: Dict[str, Any]):
        """Registra parametros adicionales."""
        if self._mlflow is None:
            return
        self._mlflow.log_params(params)

    def log_artifact(self, filepath: str):
        """Registra un archivo como artefacto (grafico, CSV, etc)."""
        if self._mlflow is None:
            return
        self._mlflow.log_artifact(filepath)

    def end_run(self):
        """Finaliza el run actual."""
        if self._mlflow is not None and self._current_run:
            self._mlflow.end_run()
            self._current_run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False

    def get_best_run(self, metric: str = "val_loss", ascending: bool = True) -> Optional[Dict]:
        """
        Obtiene el mejor run segun una metrica.

        Args:
            metric: Nombre de la metrica para ordenar.
            ascending: Si True, menor es mejor (loss). Si False, mayor es mejor (accuracy).

        Returns:
            Dict con info del mejor run, o None si no hay runs.
        """
        self._init_mlflow()
        if self._mlflow is None:
            return None

        client = self._mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(self.mlflow_cfg["experiment_name"])
        if experiment is None:
            return None

        order = "ASC" if ascending else "DESC"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if not runs:
            return None

        best = runs[0]
        return {
            "run_id": best.info.run_id,
            "run_name": best.info.run_name,
            "metrics": best.data.metrics,
            "params": best.data.params,
        }
