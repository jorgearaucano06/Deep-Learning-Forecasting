"""
LSTM Bidireccional con Mecanismo de Atencion (PyTorch)
========================================================
Modelo principal para prediccion de fallas a 24 horas.

LSTM (Long Short-Term Memory):
- Red neuronal recurrente disenada para secuencias largas.
- Tiene "puertas" que controlan que informacion recordar u olvidar:
  * Forget gate: Decide que olvidar del estado anterior.
  * Input gate: Decide que informacion nueva guardar.
  * Output gate: Decide que parte del estado usar como salida.
- Resuelve el problema del "vanishing gradient" de las RNN simples.

Bidireccional:
- Procesa la secuencia en AMBAS direcciones (pasado->futuro Y futuro->pasado).
- La concatenacion de ambas direcciones captura contexto completo.
- Output size = 2 * hidden_size (una por cada direccion).

Atencion:
- No todos los timesteps son igualmente importantes para la prediccion.
- El mecanismo de atencion aprende a "enfocarse" en los momentos clave.
- Ejemplo: las ultimas horas antes de una falla reciben mas atencion.

Arquitectura:
    Input (batch, 48, 12)
        |
    BiLSTM Layer 1 (128*2=256 dims)
        |
    BiLSTM Layer 2 (128*2=256 dims)
        |
    Attention (256 -> 256, weighted sum)
        |
    FC Layer (256 -> 64 -> 1)
        |
    Sigmoid -> Probabilidad de falla
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.utils.config_loader import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AttentionLayer(nn.Module):
    """
    Mecanismo de Atencion (Bahdanau-style).

    Calcula un peso de importancia para cada timestep de la secuencia.
    Los timesteps con mayor peso contribuyen mas a la prediccion final.

    Matematicamente:
    1. scores = tanh(W @ h)          # Proyeccion no-lineal
    2. weights = softmax(v @ scores)  # Normalizar pesos (suman 1)
    3. context = sum(weights * h)     # Suma ponderada

    Donde h son los hidden states del LSTM.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Capa lineal que proyecta cada hidden state a un score
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_output: Shape (batch, seq_len, hidden_size)

        Returns:
            context: Shape (batch, hidden_size) - Representacion ponderada
            weights: Shape (batch, seq_len) - Pesos de atencion (para visualizar)
        """
        # Calcular scores para cada timestep
        scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        scores = scores.squeeze(-1)           # (batch, seq_len)

        # Softmax para que los pesos sumen 1
        weights = F.softmax(scores, dim=1)    # (batch, seq_len)

        # Suma ponderada de los hidden states
        context = torch.bmm(
            weights.unsqueeze(1),  # (batch, 1, seq_len)
            lstm_output,           # (batch, seq_len, hidden_size)
        ).squeeze(1)               # (batch, hidden_size)

        return context, weights


class LSTMPredictor(nn.Module):
    """
    BiLSTM con Atencion para prediccion binaria de fallas.

    Input:  (batch_size, sequence_length=48, n_features=12)
    Output: (batch_size, 1) - Probabilidad de falla en 24h

    Parametros aproximados: ~400K (dependiendo de n_features).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        """
        Args:
            n_features: Numero de features de entrada (columnas del sensor).
            hidden_size: Dimensionalidad del estado oculto del LSTM.
            num_layers: Numero de capas LSTM apiladas.
            dropout: Probabilidad de dropout entre capas LSTM.
                     Dropout "apaga" neuronas aleatoriamente durante entrenamiento
                     para prevenir overfitting.
            bidirectional: Si True, procesa la secuencia en ambas direcciones.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1

        # LSTM: la capa recurrente principal
        # batch_first=True significa que el batch va en la primera dimension
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Dropout solo entre capas
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Tamano del hidden state despues del LSTM
        lstm_output_size = hidden_size * self.n_directions

        # Capa de atencion
        self.attention = AttentionLayer(lstm_output_size)

        # Capas fully connected para clasificacion
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Contar parametros
        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("LSTMPredictor: %d parametros totales (%d entrenables)", total_params, trainable)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass del modelo.

        Args:
            x: Input tensor, shape (batch, seq_len, n_features).
            return_attention: Si True, retorna tambien los pesos de atencion.

        Returns:
            output: Shape (batch, 1) - Probabilidades de falla.
            attention_weights: Shape (batch, seq_len) - Solo si return_attention=True.
        """
        # LSTM: procesa la secuencia completa
        # lstm_out shape: (batch, seq_len, hidden_size * n_directions)
        lstm_out, (hidden, cell) = self.lstm(x)

        # Atencion: pondera los timesteps por importancia
        context, attn_weights = self.attention(lstm_out)

        # Clasificacion: contexto -> probabilidad
        output = self.classifier(context)  # (batch, 1)

        if return_attention:
            return output, attn_weights
        return output


def create_lstm_predictor(config: Dict = None) -> LSTMPredictor:
    """
    Factory function para crear el modelo con configuracion del YAML.

    Factory pattern: centraliza la creacion del modelo para que
    no tengas que recordar todos los hiperparametros cada vez.
    """
    cfg = (config or load_config())["models"]["lstm_predictor"]
    prep_cfg = (config or load_config())["preprocessing"]

    # n_features depende del feature engineering (se ajusta en runtime)
    # Usamos 12 como default razonable
    n_features = 12

    model = LSTMPredictor(
        n_features=n_features,
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
    )
    return model
