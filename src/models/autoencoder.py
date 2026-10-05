"""
Autoencoder para Deteccion de Anomalias (PyTorch)
====================================================
Detecta anomalias comparando la reconstruccion con el input original.

Como funciona un Autoencoder:
1. ENCODER: Comprime el input a una representacion mas pequena (bottleneck).
   Input (12 features) -> 64 -> 32 -> 16 -> 8 (latent space)

2. DECODER: Reconstruye el input original a partir del bottleneck.
   Latent (8) -> 16 -> 32 -> 64 -> 12 (reconstruccion)

3. ENTRENAMIENTO: Se entrena SOLO con datos NORMALES.
   El modelo aprende a reconstruir fielmente patrones normales.

4. DETECCION: Cuando llega un dato anomalo:
   - El autoencoder no sabe reconstruirlo (nunca vio anomalias).
   - El error de reconstruccion (MSE) es ALTO.
   - Si MSE > umbral -> ANOMALIA detectada.

Por que funciona para mantenimiento predictivo?
- No necesitamos etiquetas de falla para entrenar (semi-supervisado).
- En produccion real, los datos de falla son escasos.
- El modelo aprende "lo normal" y alerta cuando algo no lo es.

Arquitectura:
    Input (batch, 12) -> Encoder -> Latent (batch, 8) -> Decoder -> Output (batch, 12)
    Parametros: ~8K (modelo muy ligero)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FiberAutoencoder(nn.Module):
    """
    Autoencoder simetrico para deteccion de anomalias.

    El encoder y decoder son "espejos": tienen las mismas capas
    pero en orden inverso. Esto crea una estructura tipo reloj de arena.

    Atributos:
        encoder: Red que comprime el input.
        decoder: Red que reconstruye el input.
        threshold: Umbral de MSE para clasificar como anomalia.
    """

    def __init__(
        self,
        n_features: int = 12,
        encoder_dims: list = None,
        dropout: float = 0.2,
    ):
        """
        Args:
            n_features: Numero de features de entrada.
            encoder_dims: Dimensiones del encoder [64, 32, 16, 8].
                         El decoder usa las mismas dimensiones invertidas.
            dropout: Probabilidad de dropout.
        """
        super().__init__()

        if encoder_dims is None:
            encoder_dims = [64, 32, 16, 8]

        self.n_features = n_features
        self.latent_dim = encoder_dims[-1]
        self.threshold = None  # Se calcula despues del entrenamiento

        # Construir encoder: input -> dims[0] -> dims[1] -> ... -> dims[-1]
        encoder_layers = []
        prev_dim = n_features
        for dim in encoder_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Construir decoder: dims[-1] -> ... -> dims[0] -> input
        decoder_dims = encoder_dims[-2::-1]  # Invertir sin el ultimo (es el input del decoder)
        decoder_layers = []
        prev_dim = encoder_dims[-1]
        for dim in decoder_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        # Ultima capa: reconstruccion sin activacion (valores pueden ser negativos)
        decoder_layers.append(nn.Linear(prev_dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

        total_params = sum(p.numel() for p in self.parameters())
        logger.info("Autoencoder: %d -> %d -> %d parametros: %d",
                     n_features, self.latent_dim, n_features, total_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: encode -> decode.

        Args:
            x: Input, shape (batch, n_features).

        Returns:
            Reconstruccion, shape (batch, n_features).
        """
        latent = self.encoder(x)       # Comprimir
        reconstructed = self.decoder(latent)  # Reconstruir
        return reconstructed

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Retorna la representacion latente (util para visualizacion)."""
        return self.encoder(x)

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula el error de reconstruccion por muestra.

        MSE (Mean Squared Error) por muestra:
        error_i = mean((x_i - reconstructed_i)^2)

        Returns:
            Tensor de errores, shape (batch,).
        """
        with torch.no_grad():
            reconstructed = self.forward(x)
            # MSE por muestra (no por batch)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error

    def compute_threshold(
        self, normal_data: torch.Tensor, percentile: float = 95
    ) -> float:
        """
        Calcula el umbral de anomalia a partir de datos normales.

        Logica: Si el 95% de los datos normales tiene error < X,
        entonces cualquier dato con error > X es probablemente anomalo.

        Args:
            normal_data: Tensor con datos normales, shape (n_samples, n_features).
            percentile: Percentil para el umbral (95 = top 5% es anomalia).

        Returns:
            Umbral de MSE.
        """
        errors = self.get_reconstruction_error(normal_data).numpy()
        self.threshold = float(np.percentile(errors, percentile))
        logger.info("Umbral de anomalia calculado: %.6f (percentil %d)", self.threshold, percentile)
        return self.threshold

    def detect_anomalies(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecta anomalias en datos nuevos.

        Args:
            x: Datos a evaluar, shape (n_samples, n_features).

        Returns:
            is_anomaly: Array booleano, True si es anomalia.
            errors: Array de errores de reconstruccion.
        """
        if self.threshold is None:
            raise RuntimeError("Umbral no calculado. Llama compute_threshold() primero.")

        errors = self.get_reconstruction_error(x).numpy()
        is_anomaly = errors > self.threshold
        return is_anomaly, errors


def create_autoencoder(config: Dict = None) -> FiberAutoencoder:
    """Factory function para crear el autoencoder desde config."""
    cfg = (config or load_config())["models"]["autoencoder"]
    return FiberAutoencoder(
        n_features=12,
        encoder_dims=cfg["encoder_dims"],
        dropout=cfg["dropout"],
    )
