"""
LSTM con TensorFlow/Keras
===========================
Mismo objetivo que el LSTM de PyTorch pero implementado en TensorFlow/Keras.

Keras vs PyTorch:
- Keras es mas "alto nivel": menos codigo, mas rapido de prototipar.
- PyTorch es mas "bajo nivel": mas control, mas flexible.
- En la industria se usan ambos. Conocer ambos es una ventaja competitiva.
- YOFC podria usar cualquiera de los dos en produccion.

Diferencias de implementacion:
- En Keras, defines el modelo con Sequential o Functional API.
- No necesitas escribir el training loop manualmente.
- model.fit() hace todo: forward, backward, optimizer step, metricas.

Arquitectura:
    Input (batch, 48, 12)
        |
    LSTM Layer 1 (128 units, return_sequences=True)
        |
    Dropout(0.3)
        |
    LSTM Layer 2 (64 units, return_sequences=False)
        |
    Dropout(0.3)
        |
    Dense(32, relu)
        |
    Dense(1, sigmoid) -> Probabilidad de falla
"""

import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path

from src.utils.config_loader import load_config, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_tf_lstm_model(
    n_features: int = 12,
    sequence_length: int = 48,
    config: Dict = None,
):
    """
    Crea el modelo LSTM con la API Funcional de Keras.

    Nota: Importamos TensorFlow dentro de la funcion para que
    el resto del proyecto no falle si TF no esta instalado.

    Args:
        n_features: Numero de features por timestep.
        sequence_length: Longitud de la secuencia de entrada.
        config: Configuracion. Si None, carga config.yaml.

    Returns:
        Modelo Keras compilado y listo para entrenar.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    cfg = (config or load_config())["models"]["tf_lstm"]
    units = cfg["units"]       # [128, 64]
    dropout = cfg["dropout"]
    lr = cfg["learning_rate"]

    # Input layer: define la forma del dato de entrada
    inputs = keras.Input(shape=(sequence_length, n_features), name="sensor_input")

    # Primera LSTM: return_sequences=True para pasar secuencia completa a la siguiente LSTM
    x = layers.LSTM(
        units[0],
        return_sequences=True,
        name="lstm_1",
    )(inputs)
    x = layers.Dropout(dropout)(x)

    # Segunda LSTM: return_sequences=False para obtener solo el ultimo output
    x = layers.LSTM(
        units[1],
        return_sequences=False,
        name="lstm_2",
    )(x)
    x = layers.Dropout(dropout)(x)

    # Capas densas para clasificacion
    x = layers.Dense(32, activation="relu", name="dense_1")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    # Crear modelo
    model = keras.Model(inputs=inputs, outputs=outputs, name="tf_lstm_predictor")

    # Compilar: definir loss function, optimizador y metricas
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",  # Loss para clasificacion binaria
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),           # Area bajo curva ROC
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary()
    return model


def train_tf_lstm(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict = None,
) -> dict:
    """
    Entrena el modelo TF LSTM.

    Usa callbacks de Keras:
    - EarlyStopping: Para el entrenamiento si val_loss no mejora.
    - ReduceLROnPlateau: Reduce learning rate cuando se estanca.

    Args:
        model: Modelo Keras compilado.
        X_train: Features de train, shape (n_samples, seq_len, n_features).
        y_train: Target de train, shape (n_samples,).
        X_val: Features de validacion.
        y_val: Target de validacion.

    Returns:
        Historial de entrenamiento (loss y metricas por epoch).
    """
    import tensorflow as tf
    from tensorflow import keras

    cfg = (config or load_config())["models"]["tf_lstm"]

    callbacks = [
        # Parar si val_loss no mejora en 10 epochs
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reducir learning rate si val_loss se estanca
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,       # Reducir LR a la mitad
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=cfg["batch_size"],
        epochs=cfg["epochs"],
        callbacks=callbacks,
        verbose=1,
    )

    return history.history


def save_tf_model(model, filename: str = "tf_lstm_model.h5") -> str:
    """Guarda el modelo en formato HDF5."""
    path = get_path("models") / filename
    model.save(str(path))
    logger.info("Modelo TF guardado: %s", path)
    return str(path)


def load_tf_model(filename: str = "tf_lstm_model.h5"):
    """Carga un modelo TF guardado."""
    import tensorflow as tf
    from tensorflow import keras

    path = get_path("models") / filename
    model = keras.models.load_model(str(path))
    logger.info("Modelo TF cargado: %s", path)
    return model
