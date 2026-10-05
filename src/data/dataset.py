"""
PyTorch Datasets y DataLoaders
================================
Convierte DataFrames de pandas a formatos que PyTorch puede consumir.

Conceptos clave:
- Dataset: Define COMO acceder a un dato individual (un par input-target).
- DataLoader: Itera sobre el Dataset en batches, con shuffle y paralelismo.
- Secuencias: Para modelos temporales (LSTM, CNN1D), necesitamos ventanas
  de tiempo consecutivas, no puntos individuales.

Ejemplo:
    Si sequence_length=48 (12 horas), cada muestra es:
    Input: 48 lecturas consecutivas de 12 metricas -> shape (48, 12)
    Target: Si habra falla en las proximas 24h -> shape (1,)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FiberSequenceDataset(Dataset):
    """
    Dataset para modelos secuenciales (LSTM, CNN1D).

    Crea ventanas deslizantes de longitud fija sobre la serie temporal.
    Cada ventana es una muestra de entrenamiento.

    Ejemplo con sequence_length=4:
    Datos: [a, b, c, d, e, f, g, h]
    Muestras: [a,b,c,d]->e, [b,c,d,e]->f, [c,d,e,f]->g, [d,e,f,g]->h
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "fault_in_24h",
        sequence_length: int = 48,
    ):
        """
        Args:
            df: DataFrame preprocesado y ordenado por (segment_id, timestamp).
            feature_columns: Columnas a usar como features de entrada.
            target_column: Columna objetivo.
            sequence_length: Longitud de la ventana temporal.
        """
        self.sequence_length = sequence_length
        self.target_column = target_column
        self.feature_columns = feature_columns

        # Crear secuencias por segmento
        self.sequences = []
        self.targets = []
        self._create_sequences(df)

        logger.info(
            "Dataset creado: %d secuencias, shape=(%d, %d), target=%s",
            len(self.sequences), sequence_length, len(feature_columns), target_column,
        )

    def _create_sequences(self, df: pd.DataFrame):
        """
        Genera ventanas deslizantes por cada segmento.

        Es CRITICO crear secuencias por segmento separado porque
        datos de distintos segmentos no son continuos temporalmente.
        Mezclarlos crearia secuencias invalidas.
        """
        for segment_id, seg_df in df.groupby("segment_id"):
            values = seg_df[self.feature_columns].values  # shape: (n_timesteps, n_features)
            targets = seg_df[self.target_column].values    # shape: (n_timesteps,)

            # Ventana deslizante
            for i in range(len(values) - self.sequence_length):
                seq = values[i : i + self.sequence_length]  # (seq_len, n_features)
                target = targets[i + self.sequence_length]   # Valor al final de la ventana
                self.sequences.append(seq)
                self.targets.append(target)

        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)

    def __len__(self) -> int:
        """Numero total de muestras (requerido por PyTorch)."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna una muestra individual (requerido por PyTorch).

        Args:
            idx: Indice de la muestra.

        Returns:
            Tupla (input_tensor, target_tensor).
            input_tensor shape: (sequence_length, n_features)
            target_tensor shape: (1,) o escalar
        """
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


class FiberPointDataset(Dataset):
    """
    Dataset para modelos no secuenciales (Autoencoder, Random Forest).

    Cada muestra es un unico punto en el tiempo con todas sus features.
    No se necesitan ventanas temporales.

    Shape de salida: (n_features,) para input, escalar para target.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        target_column: str = "fault_label",
    ):
        self.features = torch.tensor(
            df[feature_columns].values, dtype=torch.float32
        )
        self.targets = torch.tensor(
            df[target_column].values, dtype=torch.float32
        )

        logger.info(
            "PointDataset: %d muestras, %d features", len(self.features), self.features.shape[1],
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: List[str],
    target_column: str = "fault_in_24h",
    sequence_length: int = 48,
    batch_size: int = 64,
    dataset_type: str = "sequence",
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Crea DataLoaders para train, validation y test.

    DataLoader automatiza:
    - Batching: Agrupa muestras en mini-batches.
    - Shuffling: Mezcla datos en cada epoch (solo train).
    - num_workers: Carga datos en paralelo (mas rapido en GPUs).

    Args:
        dataset_type: "sequence" para LSTM/CNN, "point" para Autoencoder/RF.

    Returns:
        Tupla de (train_loader, val_loader, test_loader).
    """
    DatasetClass = FiberSequenceDataset if dataset_type == "sequence" else FiberPointDataset

    kwargs = {"feature_columns": feature_columns, "target_column": target_column}
    if dataset_type == "sequence":
        kwargs["sequence_length"] = sequence_length

    train_ds = DatasetClass(train_df, **kwargs)
    val_ds = DatasetClass(val_df, **kwargs)
    test_ds = DatasetClass(test_df, **kwargs)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    logger.info(
        "DataLoaders creados - Train: %d batches, Val: %d batches, Test: %d batches",
        len(train_loader), len(val_loader), len(test_loader),
    )

    return train_loader, val_loader, test_loader
