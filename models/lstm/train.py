# models/lstm/train.py

from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config.settings import get_settings
from data.db import load_ohlcv_hourly
from models.lstm.config import LSTMConfig
from models.lstm.dataset import prepare_datasets_and_scaler
from models.lstm.model import LSTMForecastModel



def _to_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch).squeeze(-1)   # (batch,)
        y_true = y_batch.squeeze(-1)          # (batch,)

        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch).squeeze(-1)
            y_true = y_batch.squeeze(-1)

            loss = criterion(y_pred, y_true)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def _inverse_scale_target(
    scaler,
    feature_cols,
    target_col_idx: int,
    y_scaled: np.ndarray,
) -> np.ndarray:
    """
    Переносимо y_scaled (N,) назад в оригінальний масштаб через scaler.inverse_transform.

    Робимо трюк:
    - створюємо нульову матрицю (N, num_features)
    - в колонку target_col_idx кладемо y_scaled
    - ганяємо через inverse_transform
    - беремо ту ж саму колонку
    """
    num_features = len(feature_cols)
    tmp = np.zeros((len(y_scaled), num_features), dtype=np.float32)
    tmp[:, target_col_idx] = y_scaled

    inv = scaler.inverse_transform(tmp)
    y_inv = inv[:, target_col_idx]
    return y_inv


def train_lstm_for_coin(
    coin_id: str,
    vs_currency: str = None,
    config: LSTMConfig | None = None,
) -> Dict[str, Any]:
    """
    Повний цикл:
    - завантажує дані з DuckDB
    - будує фічі
    - готує датасети
    - тренує LSTM
    - рахує тестові MAE/RMSE в оригінальному масштабі ціни
    - зберігає чекпоінт
    """
    settings = get_settings()
    vs_currency = vs_currency or settings.default_vs_currency
    cfg = config or LSTMConfig()

    print(f"📥 Завантажуємо дані для {coin_id} ({vs_currency})...")
    df_raw = load_ohlcv_hourly(coin_id, vs_currency)

    if df_raw.empty:
        raise RuntimeError(
            f"У DuckDB немає даних для монети '{coin_id}' ({vs_currency}). "
            f"Спочатку запусти jobs.fetch_history."
        )

    print("🧮 Готуємо датасети та скейлер...")
    (
        train_loader,
        test_loader,
        scaler,
        feature_cols,
        target_col_idx,
        train_scaled,
        test_scaled,
    ) = prepare_datasets_and_scaler(df_raw, cfg)

    input_size = len(feature_cols)
    device = _to_device()
    print(f"🔧 Пристрій: {device}, input_size={input_size}")

    model = LSTMForecastModel(
        input_size=input_size,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # --- Тренування ---
    print("🚂 Починаємо тренування LSTM...")
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _evaluate(model, test_loader, criterion, device)

        print(
            f"[Epoch {epoch:03d}/{cfg.num_epochs}] "
            f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
        )

    # --- Оцінка на тесті в оригінальному масштабі ---
    print("📏 Оцінюємо модель на тесті...")
    model.eval()

    y_true_scaled_list = []
    y_pred_scaled_list = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            y_pred = model(x_batch).squeeze(-1)
            y_true = y_batch.squeeze(-1)

            y_true_scaled_list.append(y_true.cpu().numpy())
            y_pred_scaled_list.append(y_pred.cpu().numpy())

    y_true_scaled = np.concatenate(y_true_scaled_list, axis=0)
    y_pred_scaled = np.concatenate(y_pred_scaled_list, axis=0)

    y_true = _inverse_scale_target(scaler, feature_cols, target_col_idx, y_true_scaled)
    y_pred = _inverse_scale_target(scaler, feature_cols, target_col_idx, y_pred_scaled)

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    print(f"✅ Test MAE  = {mae:.4f} {vs_currency.upper()}")
    print(f"✅ Test RMSE = {rmse:.4f} {vs_currency.upper()}")

    # --- Зберігаємо чекпоінт ---
    artifact_path: Path = cfg.artifact_path(coin_id, vs_currency)
    checkpoint = {
        "config": cfg.__dict__,
        "feature_cols": feature_cols,
        "target_col_idx": target_col_idx,
        "state_dict": model.state_dict(),
        "scaler": scaler,
    }

    torch.save(checkpoint, artifact_path)
    print(f"💾 Модель збережено в: {artifact_path}")

    return {
        "artifact_path": artifact_path,
        "mae": float(mae),
        "rmse": float(rmse),
        "config": cfg,
    }
