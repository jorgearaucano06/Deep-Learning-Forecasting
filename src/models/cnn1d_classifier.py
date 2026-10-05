"""
CNN 1D para Clasificacion de Tipos de Falla (PyTorch)
=======================================================
Clasifica en 4 categorias: normal, corte fisico, degradacion, empalme defectuoso.

CNN (Convolutional Neural Network) en 1D:
- Originalmente disenada para imagenes (2D), adaptada a secuencias (1D).
- Aplica "filtros" deslizantes que detectan patrones locales en la secuencia.
- Cada filtro aprende a detectar un patron especifico (ej: caida abrupta de potencia).

Por que CNN 1D para series temporales?
- Captura patrones LOCALES: un filtro de tamano 3 ve 3 timesteps consecutivos.
- Es mas rapida que LSTM para secuencias largas (operaciones paralelas vs secuenciales).
- Funciona bien cuando los patrones son locales (la firma pre-falla tiene formas especificas).

Arquitectura:
    Input (batch, n_features=12, seq_len=48)  <- Nota: features son los "canales"
        |
    Conv1d(12->32, kernel=3) + BatchNorm + ReLU + MaxPool
        |
    Conv1d(32->64, kernel=3) + BatchNorm + ReLU + MaxPool
        |
    Conv1d(64->128, kernel=3) + BatchNorm + ReLU
        |
    AdaptiveAvgPool1d(1)  <- Reduce a tamano fijo independiente del seq_len
        |
    FC(128 -> 64 -> 4 clases)

Parametros: ~50K
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CNN1DClassifier(nn.Module):
    """
    Red Convolucional 1D para clasificar tipos de falla en fibra optica.

    Input:  (batch, n_features, sequence_length) = (batch, 12, 48)
    Output: (batch, num_classes) = (batch, 4)

    Nota sobre dimensiones:
    En Conv1d, las features actuan como "canales" (similar a RGB en imagenes)
    y la secuencia temporal es la dimension espacial.
    """

    def __init__(
        self,
        n_features: int = 12,
        channels: list = None,
        kernel_size: int = 3,
        num_classes: int = 4,
        dropout: float = 0.3,
    ):
        """
        Args:
            n_features: Numero de features de entrada (canales).
            channels: Lista de canales para cada capa conv [32, 64, 128].
            kernel_size: Tamano del filtro convolucional.
                        kernel_size=3 significa que cada filtro ve 3 timesteps.
            num_classes: Numero de clases de salida.
            dropout: Probabilidad de dropout.
        """
        super().__init__()

        if channels is None:
            channels = [32, 64, 128]

        self.n_features = n_features
        self.num_classes = num_classes

        # Bloque convolucional 1: detecta patrones basicos
        # padding=1 mantiene la longitud de la secuencia
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_features, channels[0], kernel_size, padding=1),
            nn.BatchNorm1d(channels[0]),  # Normaliza activaciones -> entrena mas estable
            nn.ReLU(),
            nn.MaxPool1d(2),  # Reduce la secuencia a la mitad
        )

        # Bloque convolucional 2: detecta patrones mas complejos
        self.conv2 = nn.Sequential(
            nn.Conv1d(channels[0], channels[1], kernel_size, padding=1),
            nn.BatchNorm1d(channels[1]),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Bloque convolucional 3: patrones de alto nivel
        self.conv3 = nn.Sequential(
            nn.Conv1d(channels[1], channels[2], kernel_size, padding=1),
            nn.BatchNorm1d(channels[2]),
            nn.ReLU(),
        )

        # AdaptiveAvgPool: reduce CUALQUIER longitud de secuencia a 1
        # Esto hace al modelo flexible al tamano del input
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Clasificador fully connected
        self.classifier = nn.Sequential(
            nn.Linear(channels[2], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

        total_params = sum(p.numel() for p in self.parameters())
        logger.info("CNN1D: %d features, %s canales, %d clases, %d parametros",
                     n_features, channels, num_classes, total_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input shape (batch, n_features, seq_len).
               Si llega como (batch, seq_len, n_features), se transpone.

        Returns:
            Logits shape (batch, num_classes).
            Logits son scores sin normalizar (aplicar softmax para probabilidades).
        """
        # Si el input viene como (batch, seq_len, features), transponer
        if x.shape[1] != self.n_features and x.shape[2] == self.n_features:
            x = x.transpose(1, 2)  # (batch, features, seq_len)

        # Capas convolucionales
        x = self.conv1(x)  # (batch, 32, seq_len/2)
        x = self.conv2(x)  # (batch, 64, seq_len/4)
        x = self.conv3(x)  # (batch, 128, seq_len/4)

        # Global average pooling
        x = self.pool(x)   # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)

        # Clasificacion
        x = self.classifier(x)  # (batch, 4)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna probabilidades (aplicando softmax a los logits)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def create_cnn1d_classifier(config: Dict = None) -> CNN1DClassifier:
    """Factory function para crear CNN1D desde config."""
    cfg = (config or load_config())["models"]["cnn1d"]
    return CNN1DClassifier(
        n_features=12,
        channels=cfg["channels"],
        kernel_size=cfg["kernel_size"],
        num_classes=cfg["num_classes"],
        dropout=cfg["dropout"],
    )
