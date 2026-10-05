"""
Modelos Baseline: ARIMA + Random Forest
=========================================
Los baselines son modelos simples que sirven como punto de referencia.
Si un modelo de Deep Learning no supera al baseline, algo esta mal.

ARIMA (AutoRegressive Integrated Moving Average):
- Modelo clasico de series temporales.
- AutoRegressive (AR): Predice usando valores pasados.
- Integrated (I): Diferenciacion para hacer la serie estacionaria.
- Moving Average (MA): Usa errores pasados para ajustar prediccion.
- Limitacion: Solo usa UNA variable a la vez (univariado).

Random Forest:
- Ensemble de arboles de decision.
- Cada arbol ve un subconjunto aleatorio de features y datos.
- La prediccion final es el voto mayoritario (clasificacion).
- Ventajas: Robusto, no necesita mucho tuning, da feature importance.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score

from src.utils.config_loader import load_config, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ARIMABaseline:
    """
    Modelo ARIMA para prediccion univariada de potencia optica.

    ARIMA es util como baseline porque:
    - Es el modelo estandar para series temporales univariadas.
    - No requiere GPU ni mucha memoria.
    - Si un LSTM no lo supera, el problema no necesita DL.

    Nota: Importamos statsmodels dentro de los metodos para evitar
    error de importacion si no esta instalado.
    """

    def __init__(self, order: Tuple[int, int, int] = (5, 1, 2)):
        """
        Args:
            order: Tupla (p, d, q) para ARIMA.
                p = Orden autoregresivo (cuantos lags usar).
                d = Grado de diferenciacion (normalmente 0 o 1).
                q = Orden de media movil (cuantos errores pasados usar).
        """
        self.order = order
        self.model = None
        self.fitted = None

    def fit(self, series: pd.Series) -> "ARIMABaseline":
        """
        Ajusta ARIMA a una serie temporal univariada.

        Args:
            series: Serie temporal (ej: potencia optica de un segmento).
        """
        from statsmodels.tsa.arima.model import ARIMA

        logger.info("Ajustando ARIMA%s a serie de %d puntos", self.order, len(series))
        self.model = ARIMA(series.values, order=self.order)
        self.fitted = self.model.fit()
        logger.info("ARIMA ajustado. AIC: %.2f", self.fitted.aic)
        return self

    def predict(self, steps: int = 96) -> np.ndarray:
        """
        Predice los siguientes `steps` valores.

        Args:
            steps: Numero de pasos a predecir (96 = 24 horas).

        Returns:
            Array con predicciones.
        """
        if self.fitted is None:
            raise RuntimeError("Modelo no ajustado. Llama a fit() primero.")
        forecast = self.fitted.forecast(steps=steps)
        return forecast

    def get_summary(self) -> str:
        """Retorna resumen estadistico del modelo."""
        if self.fitted is None:
            return "Modelo no ajustado"
        return str(self.fitted.summary())


class RandomForestBaseline:
    """
    Random Forest para clasificacion de tipos de falla.

    Random Forest funciona asi:
    1. Crea N arboles de decision (default: 200).
    2. Cada arbol se entrena con un subconjunto aleatorio de datos Y features.
    3. Para predecir, cada arbol vota y gana la clase con mas votos.

    Esta aleatoriedad reduce el overfitting (variance) comparado con un
    solo arbol de decision.
    """

    def __init__(self, config: Dict = None):
        cfg = (config or load_config())["models"]["random_forest"]
        self.model = RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            random_state=42,
            n_jobs=-1,  # Usar todos los cores
            class_weight="balanced",  # Ajustar por desbalance de clases
        )
        self.feature_names = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: list = None,
    ) -> "RandomForestBaseline":
        """
        Entrena el Random Forest.

        Args:
            X_train: Features de entrenamiento, shape (n_samples, n_features).
            y_train: Labels, shape (n_samples,).
            feature_names: Nombres de las features (para feature importance).
        """
        self.feature_names = feature_names
        logger.info("Entrenando Random Forest: %s muestras, %s features",
                     X_train.shape[0], X_train.shape[1])

        self.model.fit(X_train, y_train)
        logger.info("Random Forest entrenado.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predice clase para cada muestra."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidades para cada clase."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evalua el modelo con metricas de clasificacion.

        Returns:
            Dict con accuracy, f1_weighted, y classification_report.
        """
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, output_dict=True)

        logger.info("RF Evaluation - Accuracy: %.4f, F1-weighted: %.4f", acc, f1)
        return {
            "accuracy": acc,
            "f1_weighted": f1,
            "classification_report": report,
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Retorna las features mas importantes.

        Feature importance en RF = cuanto reduce cada feature la impureza
        (Gini) en promedio a traves de todos los arboles.
        """
        importance = self.model.feature_importances_
        names = self.feature_names or [f"feature_{i}" for i in range(len(importance))]

        df = pd.DataFrame({
            "feature": names,
            "importance": importance,
        }).sort_values("importance", ascending=False)

        return df.head(top_n)

    def save(self, filename: str = "random_forest.joblib") -> str:
        """Guarda el modelo con joblib (eficiente para sklearn)."""
        path = get_path("models") / filename
        joblib.dump(self.model, path)
        logger.info("Modelo guardado: %s", path)
        return str(path)

    def load(self, filename: str = "random_forest.joblib") -> "RandomForestBaseline":
        """Carga un modelo guardado."""
        path = get_path("models") / filename
        self.model = joblib.load(path)
        logger.info("Modelo cargado: %s", path)
        return self
