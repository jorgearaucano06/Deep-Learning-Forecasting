"""
Preprocesador de Datos
=======================
Limpieza, normalizacion y creacion de targets para los modelos.

Pipeline de preprocesamiento:
1. Limpieza: Manejo de NaN, outliers extremos, tipos de datos.
2. Encoding: Convertir fault_type categorico a numerico.
3. Target creation: Crear variable objetivo (falla en proximas 24h).
4. Normalizacion: Escalar features a media=0, std=1 (StandardScaler).
5. Split: Dividir en train/val/test respetando el orden temporal.

Nota importante: El split es TEMPORAL, no aleatorio.
En series temporales, el split aleatorio causa data leakage porque
el modelo veria datos "del futuro" durante el entrenamiento.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.utils.config_loader import load_config, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FiberDataPreprocessor:
    """
    Preprocesa datos de la red de fibra optica.

    Atributos:
        config: Configuracion del proyecto.
        scaler: StandardScaler para normalizar features.
        label_encoder: LabelEncoder para tipos de falla.
        feature_columns: Lista de columnas usadas como features.
    """

    # Columnas que sirven como features de entrada al modelo
    SENSOR_COLUMNS = [
        "optical_power_dbm",
        "attenuation_db_km",
        "ber",
        "osnr_db",
        "chromatic_dispersion",
        "temperature",
        "humidity",
    ]

    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = self.SENSOR_COLUMNS.copy()

    def load_raw_data(self, filename: str = "fiber_readings.parquet") -> pd.DataFrame:
        """Carga datos crudos desde parquet."""
        path = get_path("raw_data") / filename
        df = pd.read_parquet(path)
        logger.info("Datos cargados: %s filas, %s columnas", len(df), len(df.columns))
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza basica de datos.

        Pasos:
        1. Eliminar duplicados exactos.
        2. Asegurar tipos de datos correctos.
        3. Interpolar NaN con el valor anterior (forward fill).
           Forward fill es apropiado para series temporales porque
           asume que el valor no cambia hasta la siguiente lectura.
        4. Clipear outliers fisicamente imposibles.
        """
        logger.info("Limpiando datos...")
        initial_len = len(df)

        # 1. Duplicados
        df = df.drop_duplicates(subset=["timestamp", "segment_id"])
        if len(df) < initial_len:
            logger.warning("Eliminados %d duplicados", initial_len - len(df))

        # 2. Tipos de datos
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)

        # 3. NaN: forward fill dentro de cada segmento
        df[self.SENSOR_COLUMNS] = df.groupby("segment_id")[self.SENSOR_COLUMNS].ffill()
        # Si quedan NaN al inicio, backward fill
        df[self.SENSOR_COLUMNS] = df.groupby("segment_id")[self.SENSOR_COLUMNS].bfill()

        # 4. Clipear valores fisicamente imposibles
        df["ber"] = df["ber"].clip(lower=1e-15, upper=1.0)
        df["osnr_db"] = df["osnr_db"].clip(lower=0)
        df["attenuation_db_km"] = df["attenuation_db_km"].clip(lower=0)
        df["humidity"] = df["humidity"].clip(0, 100)

        logger.info("Limpieza completada: %d filas", len(df))
        return df

    def create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea variables objetivo para los modelos.

        Targets creados:
        - fault_label: Clase numerica (0=normal, 1=corte, 2=degradacion, 3=empalme).
        - fault_in_24h: Binario, 1 si habra falla en las proximas 24 horas.
          Este es el target principal: queremos PREDECIR fallas antes de que ocurran.
        - time_to_fault: Horas hasta la proxima falla (NaN si no hay falla programada).

        La columna fault_in_24h se calcula mirando si pre_fault_hours esta entre 0 y 24,
        O si ya estamos en falla. Esto entrena al modelo a dar alerta temprana.
        """
        logger.info("Creando targets...")

        # Label encoding para tipos de falla
        fault_map = {"normal": 0, "physical_cut": 1, "degradation": 2, "bad_splice": 3}
        df["fault_label"] = df["fault_type"].map(fault_map)

        # Target binario: falla en proximas 24 horas
        # Un registro tiene fault_in_24h=1 si:
        # - Ya esta en falla (is_fault == 1), O
        # - Esta en fase pre-falla con menos de 24h restantes
        horizon = self.config["preprocessing"]["prediction_horizon"]
        df["fault_in_24h"] = (
            (df["is_fault"] == 1) |
            ((df["pre_fault_hours"] > 0) & (df["pre_fault_hours"] <= horizon))
        ).astype(int)

        # BER en escala logaritmica (mas util para modelos)
        # log10 comprime el rango de 1e-15...1.0 a -15...0
        df["log_ber"] = np.log10(df["ber"].clip(lower=1e-15))

        logger.info(
            "Targets creados. fault_in_24h: %.1f%% positivos",
            df["fault_in_24h"].mean() * 100,
        )
        return df

    def normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normaliza features con StandardScaler.

        StandardScaler: x_norm = (x - media) / desviacion_estandar
        Resultado: media=0, std=1 para cada feature.

        Por que normalizar?
        - Las redes neuronales convergen mas rapido con datos normalizados.
        - Sin normalizar, features con rangos grandes (ej: BER ~1e-10)
          dominarian sobre features con rangos pequenos (ej: atenuacion ~0.2).

        Args:
            df: DataFrame con las features.
            fit: Si True, ajusta el scaler (solo en train). Si False, transforma (val/test).
        """
        cols_to_scale = self.SENSOR_COLUMNS + ["log_ber"]
        cols_present = [c for c in cols_to_scale if c in df.columns]

        if fit:
            df[cols_present] = self.scaler.fit_transform(df[cols_present])
        else:
            df[cols_present] = self.scaler.transform(df[cols_present])

        return df

    def temporal_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split temporal: train -> val -> test en orden cronologico.

        A diferencia del split aleatorio, el temporal garantiza que:
        - Train contiene los datos mas antiguos.
        - Validation contiene datos intermedios.
        - Test contiene los datos mas recientes.

        Esto simula el uso real: entrenamos con datos historicos
        y evaluamos con datos que el modelo nunca ha visto.
        """
        cfg = self.config["preprocessing"]
        test_size = cfg["test_size"]
        val_size = cfg["val_size"]

        # Ordenar por timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        n = len(df)

        # Indices de corte
        train_end = int(n * (1 - test_size - val_size))
        val_end = int(n * (1 - test_size))

        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()

        logger.info(
            "Split temporal - Train: %d (%.0f%%), Val: %d (%.0f%%), Test: %d (%.0f%%)",
            len(train_df), 100 * len(train_df) / n,
            len(val_df), 100 * len(val_df) / n,
            len(test_df), 100 * len(test_df) / n,
        )
        return train_df, val_df, test_df

    def process_pipeline(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Pipeline completo de preprocesamiento.

        Returns:
            Tupla (train_df, val_df, test_df) con datos listos para modelar.
        """
        # Cargar y limpiar
        df = self.load_raw_data()
        df = self.clean(df)
        df = self.create_targets(df)

        # Split temporal ANTES de normalizar
        # (para evitar data leakage del test al scaler)
        train_df, val_df, test_df = self.temporal_split(df)

        # Normalizar: fit solo en train, transform en val y test
        train_df = self.normalize(train_df, fit=True)
        val_df = self.normalize(val_df, fit=False)
        test_df = self.normalize(test_df, fit=False)

        # Guardar
        processed_path = get_path("processed_data")
        train_df.to_parquet(processed_path / "train.parquet", index=False)
        val_df.to_parquet(processed_path / "val.parquet", index=False)
        test_df.to_parquet(processed_path / "test.parquet", index=False)

        logger.info("Datos procesados guardados en %s", processed_path)
        return train_df, val_df, test_df
