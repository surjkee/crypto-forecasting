# models/gru/inference.py

import os
from datetime import timedelta
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch

from config.settings import get_settings
from data.db import load_ohlcv_hourly, store_hourly_forecast, ensure_forecast_continuity
from features.transform import build_feature_frame
from models.gru.config import GRUConfig
from models.gru.model import GRUForecastModel


def load_gru_checkpoint(
    coin_id: str,
    vs_currency: str,
    device: torch.device | None = None,
) -> Tuple[GRUForecastModel, MinMaxScaler, list[str], int, GRUConfig]:
    """
    Завантажує збережену GRU-модель + scaler + фічі з чекпоінта.

    Повертає:
        model          – GRUForecastModel на потрібному device
        scaler         – MinMaxScaler, на якому модель навчалась
        feature_cols   – порядок і список фіч, як у тренуванні
        target_col_idx – індекс таргету всередині feature_cols
        cfg            – GRUConfig, збережений у чекпоінті (якщо був)
    """
    device = device or torch.device("cpu")

    # Базовий конфіг тільки для побудови шляху до артефакту
    cfg = GRUConfig()
    artifact_path: Path = cfg.artifact_path(coin_id, vs_currency)

    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Не знайдено GRU-артефакт для {coin_id} ({vs_currency}): {artifact_path}"
        )

    checkpoint = torch.load(artifact_path, map_location=device, weights_only=False)

    # Якщо в чекпоінті збережений повний config – підміняємо ним дефолтний
    cfg_dict = checkpoint.get("config")
    if cfg_dict:
        cfg = GRUConfig(**cfg_dict)

    feature_cols: list[str] = checkpoint["feature_cols"]
    target_col_idx: int = int(checkpoint["target_col_idx"])
    scaler: MinMaxScaler = checkpoint["scaler"]

    # Витягуємо state_dict (підтримуємо варіант, коли збережено «голий» state_dict)
    state = checkpoint.get("state_dict", checkpoint)

    model = GRUForecastModel(
        input_size=len(feature_cols),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    return model, scaler, feature_cols, target_col_idx, cfg


def _inverse_scale_target(
    scaler: MinMaxScaler,
    feature_cols: list[str],
    target_col_idx: int,
    y_scaled: np.ndarray,
) -> np.ndarray:
    """
    Той самий трюк, що й у LSTM:
    кладемо y_scaled у колонку target_col, інші фічі = 0,
    ганяємо через inverse_transform і дістаємо лише target.
    """
    dummy = np.zeros((len(y_scaled), len(feature_cols)), dtype=np.float32)
    dummy[:, target_col_idx] = y_scaled
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_col_idx]


def forecast_next_t1_and_store(
    coin_id: str,
    vs_currency: str | None = None,
    model_name: str = "gru_v1.0",
) -> Dict[str, Any]:
    """
    t+1-прогноз для GRU з записом у forecast_hourly.

    1) тягнемо останні OHLCV з DuckDB
    2) рахуємо фічі як у train_gru
    3) вантажимо GRU + scaler + feature_cols з чекпоінта
    4) масштабуємо дані тим самим scaler'ом, що й на train
    5) робимо прогноз, inverse-scale тільки таргет, пишемо у forecast_hourly
    """
    settings = get_settings()
    vs_currency = vs_currency or settings.default_vs_currency

    # Гарантуємо безперервність ряду прогнозів (як і раніше)
    ensure_forecast_continuity(coin_id, vs_currency, model_name)

    # 1. Сирі дані з DuckDB
    df_raw = load_ohlcv_hourly(coin_id, vs_currency)
    if df_raw.empty:
        raise RuntimeError(
            f"Немає даних OHLCV для {coin_id} ({vs_currency}) у DuckDB. "
            "Спочатку запусти jobs.fetch_history."
        )

    # 2. Фічі як у тренуванні
    df_feat = build_feature_frame(df_raw)

    # 3. Завантажуємо модель + scaler + feature_cols із чекпоінта
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, feature_cols, target_col_idx, cfg = load_gru_checkpoint(
        coin_id=coin_id,
        vs_currency=vs_currency,
        device=device,
    )

    # Формуємо датафрейм у тому ж порядку колонок, що й при тренуванні
    df_model = (
        df_feat[["ts"] + feature_cols]
        .dropna(subset=feature_cols)
        .reset_index(drop=True)
    )

    if len(df_model) <= cfg.window_size:
        raise RuntimeError(
            f"Замало даних для прогнозу: {len(df_model)} рядків, "
            f"window_size={cfg.window_size}"
        )

    ts_anchor = df_model["ts"].max()
    ts_forecast = ts_anchor + timedelta(hours=1)

    # 4. Скейлінг ТИМ САМИМ scaler'ом, що був на тренуванні (transform, НЕ fit_transform)
    values = df_model[feature_cols].values.astype(np.float32)
    values_scaled = scaler.transform(values)

    # Беремо останнє вікно для t+1
    window = values_scaled[-cfg.window_size :]   # (window, n_features)
    x = torch.from_numpy(window).unsqueeze(0).to(device)  # (1, window, n_features)

    # 5. Прогноз у скейленому просторі
    with torch.no_grad():
        y_scaled = model(x).cpu().numpy().reshape(-1)[-1]

    # 6. inverse-scale тільки таргет назад у USD
    y_pred = float(
        _inverse_scale_target(
            scaler,
            feature_cols,
            target_col_idx,
            np.array([y_scaled], dtype=np.float32),
        )[0]
    )

    # 7. Запис у DuckDB
    store_hourly_forecast(
        coin_id=coin_id,
        vs_currency=vs_currency,
        ts_anchor=ts_anchor,
        ts_forecast=ts_forecast,
        model=model_name,
        y_pred=y_pred,
    )

    return {
        "coin_id": coin_id,
        "vs_currency": vs_currency,
        "ts_anchor": ts_anchor,
        "ts_forecast": ts_forecast,
        "y_pred": y_pred,
    }
