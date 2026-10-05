"""
Training Loop Generico con MLflow Tracking
=============================================
Un training loop reusable para cualquier modelo de PyTorch.

Por que un trainer generico?
- Evitar duplicar codigo de entrenamiento en cada modelo.
- Centralizar buenas practicas: early stopping, gradient clipping, logging.
- Integrar MLflow automaticamente para tracking de experimentos.

MLflow:
- Framework open-source para gestionar el ciclo de vida de ML.
- Registra automaticamente: hiperparametros, metricas, modelos.
- Permite comparar experimentos y reproducir resultados.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import time
from typing import Dict, Optional, Callable
from pathlib import Path

from src.utils.config_loader import load_config, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    """
    Training loop generico para modelos PyTorch.

    Funcionalidades:
    - Early stopping: Para si la validacion no mejora.
    - Gradient clipping: Previene exploding gradients.
    - Learning rate scheduling: Reduce LR cuando se estanca.
    - MLflow logging: Registra todo automaticamente.
    - Checkpoint: Guarda el mejor modelo.

    Uso:
        trainer = Trainer(model, criterion, optimizer, device)
        history = trainer.fit(train_loader, val_loader, epochs=50)
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        max_grad_norm: float = 1.0,
        model_name: str = "model",
    ):
        """
        Args:
            model: Modelo PyTorch a entrenar.
            criterion: Funcion de perdida (BCEWithLogitsLoss, CrossEntropyLoss, etc).
            optimizer: Optimizador (Adam, SGD, etc).
            device: CPU o CUDA.
            scheduler: Learning rate scheduler (opcional).
            max_grad_norm: Maximo de norma del gradiente para clipping.
            model_name: Nombre para guardar checkpoints.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.model_name = model_name

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        patience: int = 10,
        use_mlflow: bool = False,
    ) -> Dict:
        """
        Entrena el modelo con early stopping.

        Args:
            train_loader: DataLoader de entrenamiento.
            val_loader: DataLoader de validacion.
            epochs: Numero maximo de epochs.
            patience: Epochs sin mejora antes de parar.
            use_mlflow: Si True, registra en MLflow.

        Returns:
            Diccionario con historial de entrenamiento.
        """
        history = {"train_loss": [], "val_loss": [], "lr": []}
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = get_path("models") / f"{self.model_name}_best.pt"

        # MLflow tracking (opcional)
        mlflow_run = None
        if use_mlflow:
            try:
                import mlflow
                mlflow.set_tracking_uri(str(get_path("mlruns")))
                config = load_config()
                mlflow.set_experiment(config["mlflow"]["experiment_name"])
                mlflow_run = mlflow.start_run(run_name=self.model_name)
                mlflow.log_param("model_name", self.model_name)
                mlflow.log_param("epochs", epochs)
                mlflow.log_param("patience", patience)
                mlflow.log_param("optimizer", type(self.optimizer).__name__)
                mlflow.log_param("learning_rate", self.optimizer.param_groups[0]["lr"])
            except ImportError:
                logger.warning("MLflow no disponible, continuando sin tracking")
                use_mlflow = False

        logger.info("Iniciando entrenamiento: %s epochs, patience=%d", epochs, patience)
        start_time = time.time()

        for epoch in range(epochs):
            # --- Train ---
            train_loss = self._train_epoch(train_loader)

            # --- Validate ---
            val_loss = self._validate_epoch(val_loader)

            # --- Learning rate ---
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # --- Registrar ---
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            if use_mlflow:
                import mlflow
                mlflow.log_metrics(
                    {"train_loss": train_loss, "val_loss": val_loss, "lr": current_lr},
                    step=epoch,
                )

            # --- Early stopping ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), str(best_model_path))
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0:
                logger.info(
                    "Epoch %d/%d | Train: %.4f | Val: %.4f | LR: %.6f | Patience: %d/%d",
                    epoch + 1, epochs, train_loss, val_loss, current_lr,
                    patience_counter, patience,
                )

            if patience_counter >= patience:
                logger.info("Early stopping en epoch %d", epoch + 1)
                break

        elapsed = time.time() - start_time
        logger.info("Entrenamiento completado en %.1f seg. Mejor val_loss: %.4f", elapsed, best_val_loss)

        # Cargar mejor modelo
        self.model.load_state_dict(torch.load(str(best_model_path), weights_only=True))

        if use_mlflow and mlflow_run:
            import mlflow
            mlflow.log_metric("best_val_loss", best_val_loss)
            mlflow.log_metric("training_time_sec", elapsed)
            mlflow.end_run()

        history["best_val_loss"] = best_val_loss
        history["total_epochs"] = len(history["train_loss"])
        return history

    def _train_epoch(self, loader: DataLoader) -> float:
        """Ejecuta una epoch de entrenamiento."""
        self.model.train()
        total_loss = 0

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(batch_x)

            # Ajustar shape si es necesario
            if output.dim() > 1 and output.shape[-1] == 1:
                output = output.squeeze(-1)

            # CrossEntropyLoss espera target long
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                batch_y = batch_y.long()

            loss = self.criterion(output, batch_y)
            loss.backward()

            # Gradient clipping: previene que gradientes muy grandes
            # desestabilicen el entrenamiento
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )

            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    def _validate_epoch(self, loader: DataLoader) -> float:
        """Ejecuta una epoch de validacion (sin gradientes)."""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                output = self.model(batch_x)
                if output.dim() > 1 and output.shape[-1] == 1:
                    output = output.squeeze(-1)

                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    batch_y = batch_y.long()

                loss = self.criterion(output, batch_y)
                total_loss += loss.item()

        return total_loss / len(loader)
