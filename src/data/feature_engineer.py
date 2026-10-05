"""
Feature Engineering Pipeline
=============================
Crea features derivadas a partir de las lecturas de sensores.

Feature Engineering es el proceso de crear nuevas variables a partir
de los datos existentes para ayudar al modelo a aprender patrones.

Tipos de features que creamos:
1. Rolling Statistics: Media, std, min, max en ventanas temporales.
   Capturan tendencias y volatilidad reciente.
2. Lag Features: Valores anteriores de las metricas.
   Permiten al modelo "recordar" estados pasados.
3. Rate of Change: Velocidad de cambio de las metricas.
   Cambios rapidos suelen preceder fallas.
4. Domain Features: Ratios y combinaciones con significado fisico.
   Ejemplo: ratio potencia/atenuacion indica salud del segmento.

Por que es importante?
Un modelo de ML es tan bueno como sus features. Features bien disenadas
pueden hacer que un Random Forest sea mejor que una red neuronal con features crudas.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FiberFeatureEngineer:
    """
    Pipeline de creacion de features para datos de fibra optica.

    Las features se crean POR SEGMENTO (groupby segment_id)
    porque cada segmento tiene su propia serie temporal independiente.
    """

    SENSOR_COLUMNS = [
        "optical_power_dbm",
        "attenuation_db_km",
        "ber",
        "osnr_db",
        "chromatic_dispersion",
    ]

    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.feat_cfg = self.config["features"]
        self.rolling_windows = self.feat_cfg["rolling_windows"]
        self.lag_steps = self.feat_cfg["lag_steps"]

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline completo de feature engineering.

        El orden importa: primero features simples, luego las que dependen de otras.
        """
        logger.info("Creando features para %d filas...", len(df))

        df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

        # 1. Features temporales (del timestamp)
        df = self._add_temporal_features(df)

        # 2. Rolling statistics (por segmento)
        df = self._add_rolling_features(df)

        # 3. Lag features (valores anteriores)
        df = self._add_lag_features(df)

        # 4. Rate of change (derivada)
        df = self._add_rate_of_change(df)

        # 5. Features de dominio (conocimiento experto)
        df = self._add_domain_features(df)

        # Eliminar filas con NaN creadas por rolling/lag
        initial = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info("Filas eliminadas por NaN de rolling/lag: %d", initial - len(df))

        n_features = len([c for c in df.columns if c not in ["timestamp", "segment_id", "fault_type", "is_fault", "fault_label", "fault_in_24h", "pre_fault_hours"]])
        logger.info("Feature engineering completado: %d features creadas", n_features)

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extrae features del timestamp.

        Las redes de fibra tienen patrones temporales:
        - Hora del dia: mayor trafico en horario laboral.
        - Dia de la semana: diferente carga lunes vs domingo.

        Usamos codificacion ciclica (sin/cos) en vez de one-hot porque:
        - Hora 23 y hora 0 estan cerca temporalmente pero lejos en one-hot.
        - sin/cos preservan la naturaleza ciclica: sin(23h) ≈ sin(0h).
        """
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Codificacion ciclica para hora
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Codificacion ciclica para dia de la semana
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        return df

    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estadisticas moviles (rolling) por ventana temporal.

        Rolling mean: Suaviza la senal, muestra tendencia.
        Rolling std: Mide volatilidad. Alta volatilidad = posible problema.
        Rolling min/max: Capturan valores extremos recientes.

        Ventanas: [4, 12, 48] lecturas = [1h, 3h, 12h]
        """
        for window in self.rolling_windows:
            for col in self.SENSOR_COLUMNS:
                grouped = df.groupby("segment_id")[col]

                # Media movil
                df[f"{col}_rolling_mean_{window}"] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                # Desviacion estandar movil
                df[f"{col}_rolling_std_{window}"] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )

            logger.info("Rolling features ventana=%d creadas", window)

        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features con valores anteriores (lags).

        Lag-1: Valor hace 15 min
        Lag-4: Valor hace 1 hora
        Lag-12: Valor hace 3 horas
        Lag-48: Valor hace 12 horas

        Estas features son cruciales porque el modelo necesita ver
        como evolucionaron las metricas para predecir el futuro.
        """
        for lag in self.lag_steps:
            for col in self.SENSOR_COLUMNS:
                df[f"{col}_lag_{lag}"] = df.groupby("segment_id")[col].shift(lag)

        return df

    def _add_rate_of_change(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la tasa de cambio (diferencia) de cada metrica.

        diff() calcula x[t] - x[t-1].
        - Valor positivo = la metrica aumento.
        - Valor negativo = la metrica disminuyo.
        - Cambio grande = posible anomalia.

        Una caida rapida en potencia optica (diff negativo grande)
        es una senal clasica de falla inminente.
        """
        for col in self.SENSOR_COLUMNS:
            df[f"{col}_diff_1"] = df.groupby("segment_id")[col].diff(1)
            df[f"{col}_diff_4"] = df.groupby("segment_id")[col].diff(4)

        return df

    def _add_domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features basadas en conocimiento del dominio de fibra optica.

        Estas features combinan metricas de formas que tienen
        significado fisico en telecomunicaciones:

        - power_attenuation_ratio: Indica eficiencia del segmento.
          Ratio bajo = problema potencial.
        - snr_ber_product: Relacion entre calidad de senal y errores.
          En fibra sana, alto SNR implica bajo BER (producto bajo).
        - health_score: Indice compuesto de salud del segmento.
          Combina potencia, SNR y BER en un solo numero 0-100.
        """
        # Ratio potencia/atenuacion (indicador de eficiencia)
        df["power_attenuation_ratio"] = (
            df["optical_power_dbm"] / df["attenuation_db_km"].replace(0, np.nan)
        )

        # Producto SNR * log(BER) - correlacion de calidad
        df["snr_ber_interaction"] = df["osnr_db"] * np.log10(df["ber"].clip(lower=1e-15))

        # Score de salud del segmento (0-100)
        # Normalizamos cada componente a un rango 0-1 y promediamos
        power_score = (df["optical_power_dbm"] - (-30)) / (0 - (-30))
        snr_score = df["osnr_db"] / 40
        ber_score = 1 - (np.log10(df["ber"].clip(1e-15)) + 15) / 15

        df["health_score"] = (
            (power_score.clip(0, 1) + snr_score.clip(0, 1) + ber_score.clip(0, 1)) / 3 * 100
        )

        # Desviacion de la media por segmento (Z-score local)
        for col in self.SENSOR_COLUMNS:
            seg_mean = df.groupby("segment_id")[col].transform("mean")
            seg_std = df.groupby("segment_id")[col].transform("std").replace(0, 1)
            df[f"{col}_zscore"] = (df[col] - seg_mean) / seg_std

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Retorna la lista de columnas que son features (excluye metadata y targets).
        """
        exclude = {
            "timestamp", "segment_id", "fault_type", "is_fault",
            "fault_label", "fault_in_24h", "pre_fault_hours",
        }
        return [c for c in df.columns if c not in exclude]
