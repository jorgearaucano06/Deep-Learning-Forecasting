"""
Script para entrenar todos los modelos secuencialmente.

Uso:
    python scripts/train_all_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import numpy as np

from src.data.preprocessor import FiberDataPreprocessor
from src.data.feature_engineer import FiberFeatureEngineer
from src.data.dataset import create_dataloaders
from src.models.lstm_predictor import LSTMPredictor
from src.models.autoencoder import FiberAutoencoder
from src.models.cnn1d_classifier import CNN1DClassifier
from src.models.baseline import RandomForestBaseline
from src.training.trainer import Trainer
from src.training.evaluator import ModelEvaluator
from src.utils.config_loader import load_config, get_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo: %s", device)

    # --- Preparar datos ---
    logger.info("=== Preparando datos ===")
    preprocessor = FiberDataPreprocessor()
    train_df, val_df, test_df = preprocessor.process_pipeline()

    feature_cols = [
        "optical_power_dbm", "attenuation_db_km", "ber", "osnr_db",
        "chromatic_dispersion", "temperature", "humidity", "log_ber",
    ]
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    seq_len = config["preprocessing"]["sequence_length"]

    # --- Random Forest ---
    logger.info("\n=== Entrenando Random Forest ===")
    fe = FiberFeatureEngineer()
    train_feat = fe.create_all_features(train_df)
    test_feat = fe.create_all_features(test_df)
    feat_names = fe.get_feature_names(train_feat)

    rf = RandomForestBaseline()
    rf.fit(train_feat[feat_names].values, train_feat["fault_label"].values, feat_names)
    rf_results = rf.evaluate(test_feat[feat_names].values, test_feat["fault_label"].values)
    logger.info("RF - Accuracy: %.4f, F1: %.4f", rf_results["accuracy"], rf_results["f1_weighted"])
    rf.save()

    # --- BiLSTM ---
    logger.info("\n=== Entrenando BiLSTM + Attention ===")
    cfg_lstm = config["models"]["lstm_predictor"]
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df, feature_cols, "fault_in_24h",
        seq_len, cfg_lstm["batch_size"], "sequence",
    )

    lstm = LSTMPredictor(
        n_features=len(feature_cols),
        hidden_size=cfg_lstm["hidden_size"],
        num_layers=cfg_lstm["num_layers"],
        dropout=cfg_lstm["dropout"],
        bidirectional=cfg_lstm["bidirectional"],
    ).to(device)

    trainer = Trainer(
        lstm, nn.BCEWithLogitsLoss(),
        torch.optim.Adam(lstm.parameters(), lr=cfg_lstm["learning_rate"]),
        device, model_name="lstm",
    )
    trainer.fit(train_loader, val_loader, epochs=cfg_lstm["epochs"], patience=cfg_lstm["patience"])

    evaluator = ModelEvaluator(device)
    lstm_results = evaluator.evaluate_binary(lstm, test_loader)

    # --- Autoencoder ---
    logger.info("\n=== Entrenando Autoencoder ===")
    cfg_ae = config["models"]["autoencoder"]
    ae = FiberAutoencoder(
        n_features=len(feature_cols),
        encoder_dims=cfg_ae["encoder_dims"],
        dropout=cfg_ae["dropout"],
    ).to(device)

    train_normal = train_df[train_df["fault_type"] == "normal"][feature_cols].values
    train_tensor = torch.FloatTensor(train_normal).to(device)
    dataset = torch.utils.data.TensorDataset(train_tensor, train_tensor)
    ae_loader = torch.utils.data.DataLoader(dataset, batch_size=cfg_ae["batch_size"], shuffle=True)

    ae_trainer = Trainer(
        ae, nn.MSELoss(),
        torch.optim.Adam(ae.parameters(), lr=cfg_ae["learning_rate"]),
        device, model_name="autoencoder",
    )
    # For autoencoder, use same loader for train/val (self-supervised)
    ae_trainer.fit(ae_loader, ae_loader, epochs=cfg_ae["epochs"], patience=20)
    torch.save(ae.state_dict(), str(get_path("models") / "autoencoder_best.pt"))

    # --- CNN 1D ---
    logger.info("\n=== Entrenando CNN 1D ===")
    cfg_cnn = config["models"]["cnn1d"]
    train_loader_mc, val_loader_mc, test_loader_mc = create_dataloaders(
        train_df, val_df, test_df, feature_cols, "fault_label",
        seq_len, cfg_cnn["batch_size"], "sequence",
    )

    cnn = CNN1DClassifier(
        n_features=len(feature_cols),
        channels=cfg_cnn["channels"],
        kernel_size=cfg_cnn["kernel_size"],
        num_classes=cfg_cnn["num_classes"],
        dropout=cfg_cnn["dropout"],
    ).to(device)

    cnn_trainer = Trainer(
        cnn, nn.CrossEntropyLoss(),
        torch.optim.Adam(cnn.parameters(), lr=cfg_cnn["learning_rate"]),
        device, model_name="cnn1d",
    )
    cnn_trainer.fit(train_loader_mc, val_loader_mc, epochs=cfg_cnn["epochs"], patience=10)

    cnn_results = evaluator.evaluate_multiclass(
        cnn, test_loader_mc, ["normal", "physical_cut", "degradation", "bad_splice"]
    )

    # --- Resumen ---
    logger.info("\n" + "=" * 60)
    logger.info("RESUMEN DE ENTRENAMIENTO")
    logger.info("=" * 60)
    logger.info("Random Forest  - Accuracy: %.4f, F1: %.4f", rf_results["accuracy"], rf_results["f1_weighted"])
    logger.info("BiLSTM+Attn    - Accuracy: %.4f, F1: %.4f, AUC: %.4f", lstm_results["accuracy"], lstm_results["f1"], lstm_results["roc_auc"])
    logger.info("CNN 1D         - Accuracy: %.4f, F1-weighted: %.4f", cnn_results["accuracy"], cnn_results["f1_weighted"])
    logger.info("Autoencoder    - Entrenado (deteccion de anomalias)")
    logger.info("Modelos guardados en: %s", get_path("models"))


if __name__ == "__main__":
    main()
