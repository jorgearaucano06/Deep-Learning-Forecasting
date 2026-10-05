"""
Evaluador de Modelos
======================
Calcula metricas tecnicas y de negocio para comparar modelos.

Metricas tecnicas:
- Accuracy: % de predicciones correctas. Engañosa con clases desbalanceadas.
- Precision: De los que predije positivo, cuantos lo eran realmente?
  Alta precision = pocas falsas alarmas.
- Recall (Sensibilidad): De los positivos reales, cuantos detecte?
  Alto recall = pocas fallas no detectadas.
- F1 Score: Media armonica de precision y recall (balance entre ambas).
- ROC AUC: Area bajo la curva ROC. 1.0 = perfecto, 0.5 = aleatorio.

Para mantenimiento predictivo:
- Recall es MAS IMPORTANTE que precision.
  Es preferible una falsa alarma (inspeccion innecesaria)
  a no detectar una falla (downtime costoso).
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from typing import Dict, List, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evalua modelos de PyTorch con metricas estandar.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def evaluate_binary(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Evaluacion para clasificacion binaria (falla si/no).

        Args:
            model: Modelo entrenado.
            test_loader: DataLoader de test.
            threshold: Umbral de decision (default 0.5).

        Returns:
            Dict con todas las metricas.
        """
        model.eval()
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                output = model(batch_x)

                if output.dim() > 1 and output.shape[-1] == 1:
                    output = output.squeeze(-1)

                probs = torch.sigmoid(output).cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(batch_y.numpy())

        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        all_preds = (all_probs >= threshold).astype(int)

        metrics = {
            "accuracy": accuracy_score(all_targets, all_preds),
            "precision": precision_score(all_targets, all_preds, zero_division=0),
            "recall": recall_score(all_targets, all_preds, zero_division=0),
            "f1": f1_score(all_targets, all_preds, zero_division=0),
            "roc_auc": roc_auc_score(all_targets, all_probs) if len(np.unique(all_targets)) > 1 else 0.0,
            "threshold": threshold,
            "n_samples": len(all_targets),
            "n_positive": int(all_targets.sum()),
        }

        logger.info(
            "Binary Eval - Acc: %.4f, Prec: %.4f, Recall: %.4f, F1: %.4f, AUC: %.4f",
            metrics["accuracy"], metrics["precision"], metrics["recall"],
            metrics["f1"], metrics["roc_auc"],
        )

        return metrics

    def evaluate_multiclass(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        class_names: List[str] = None,
    ) -> Dict:
        """
        Evaluacion para clasificacion multi-clase (tipo de falla).

        Args:
            model: Modelo entrenado.
            test_loader: DataLoader de test.
            class_names: Nombres de las clases.

        Returns:
            Dict con metricas por clase y globales.
        """
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                output = model(batch_x)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(batch_y.numpy())

        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets).astype(int)

        if class_names is None:
            class_names = [f"class_{i}" for i in range(len(np.unique(all_targets)))]

        metrics = {
            "accuracy": accuracy_score(all_targets, all_preds),
            "f1_weighted": f1_score(all_targets, all_preds, average="weighted", zero_division=0),
            "f1_macro": f1_score(all_targets, all_preds, average="macro", zero_division=0),
            "confusion_matrix": confusion_matrix(all_targets, all_preds).tolist(),
            "classification_report": classification_report(
                all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0,
            ),
        }

        logger.info(
            "Multiclass Eval - Acc: %.4f, F1-weighted: %.4f, F1-macro: %.4f",
            metrics["accuracy"], metrics["f1_weighted"], metrics["f1_macro"],
        )

        return metrics

    @staticmethod
    def compare_models(results: Dict[str, Dict]) -> str:
        """
        Genera tabla comparativa de modelos.

        Args:
            results: {nombre_modelo: {metrica: valor}}.

        Returns:
            Tabla formateada como string.
        """
        header = f"{'Modelo':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}"
        separator = "-" * len(header)
        lines = [separator, header, separator]

        for name, metrics in results.items():
            line = (
                f"{name:<20} "
                f"{metrics.get('accuracy', 0):<10.4f} "
                f"{metrics.get('precision', metrics.get('f1_weighted', 0)):<10.4f} "
                f"{metrics.get('recall', metrics.get('f1_macro', 0)):<10.4f} "
                f"{metrics.get('f1', metrics.get('f1_weighted', 0)):<10.4f} "
                f"{metrics.get('roc_auc', 0):<10.4f}"
            )
            lines.append(line)

        lines.append(separator)
        return "\n".join(lines)
