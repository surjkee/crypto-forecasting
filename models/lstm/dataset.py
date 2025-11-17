# models/lstm/dataset.py

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

from features.transform import build_feature_frame
from models.lstm.config import LSTMConfig


class SequenceDataset(Dataset):
    def __init__(
        self,
        data_scaled: np.ndarray,
        window_size: int,
        target_col_idx: int,
    ) -> None:
        """
        data_scaled: (N, num_features) — вже відмасштабований масив
        window_size: довжина вікна
        target_col_idx: індекс колонки-цілі в data_scaled
        """
        self.data_scaled = data_scaled
        self.window_size = window_size
        self.target_col_idx = target_col_idx

    def __len__(self) -> int:
        # останній y — на позиції i, де i = len - 1
        # перший доступний i = window_size
        return len(self.data_scaled) - self.window_size

    def __getitem__(self, idx: int):
        x_start = idx
        x_end = idx + self.window_size
        y_idx = x_end  # next step after window

        x = self.data_scaled[x_start:x_end, :]   # (window_size, num_features)
        y = self.data_scaled[y_idx, self.target_col_idx]  # scalar

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.float32)  # (1,)

        return x_tensor, y_tensor


def build_model_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Створює df_model для моделі: price + числові фічі з build_feature_frame.

    ВАЖЛИВО:
    - беремо тільки числові колонки (float/int),
    - 'price' ставимо першою,
    - ігноруємо будь-які string-поля (типу coin_id, symbol, vs_currency).
    """
    df_feat = build_feature_frame(df_raw)

    # Вибираємо тільки числові колонки
    numeric_cols = df_feat.select_dtypes(include=["number", "float", "int"]).columns.tolist()

    # 'ts' — не фіча, цю колонку тримаємо окремо
    numeric_cols = [c for c in numeric_cols if c != "ts"]

    if "price" not in numeric_cols:
        raise ValueError("Очікується колонка 'price' у фреймі з фічами (numeric_cols).")

    # price спочатку, потім решта числових фіч
    other_cols = [c for c in numeric_cols if c != "price"]
    ordered_cols = ["price"] + other_cols

    df_model = df_feat[["ts"] + ordered_cols].copy()
    return df_model



def prepare_datasets_and_scaler(
    df_raw: pd.DataFrame,
    config: LSTMConfig,
) -> Tuple[
    DataLoader, DataLoader, MinMaxScaler, List[str], int, np.ndarray, np.ndarray
]:
    """
    Готує train/test DataLoader'и, скейлер і службову інформацію.

    Повертає:
    - train_loader
    - test_loader
    - scaler (MinMaxScaler по всіх фічах)
    - feature_cols (список колонок без ts)
    - target_col_idx (індекс 'price' в цих фічах)
    - train_scaled (масив для можливого аналізу)
    - test_scaled (масив для можливого аналізу)
    """
    df_model = build_model_frame(df_raw)

    # Беремо тільки фічі (без ts)
    feature_cols = [c for c in df_model.columns if c != "ts"]
    target_col_idx = feature_cols.index(config.target_col)

    # 🔴 Критично: чистимо NaN перед скейлінгом
    df_model_clean = df_model.dropna(subset=feature_cols).reset_index(drop=True)

    if len(df_model_clean) <= config.window_size + 1:
        raise RuntimeError(
            f"Замало даних після dropna: {len(df_model_clean)} рядків, "
            f"а window_size={config.window_size}. "
            "Спробуй або зменшити window_size, або збільшити історію."
        )

    values = df_model_clean[feature_cols].values.astype(np.float32)

    n_total = len(df_model_clean)
    split_idx = int(n_total * config.train_ratio)

    if split_idx <= config.window_size:
        raise RuntimeError(
            f"split_idx={split_idx} <= window_size={config.window_size}. "
            "Збільште обсяг даних або змініть train_ratio/window_size."
        )

    # Як у Colab:
    # train = [0 : split_idx]
    # test  = [split_idx - window : end] (з нахльостом)
    train_values = values[:split_idx]
    test_values = values[split_idx - config.window_size :]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_values)
    test_scaled = scaler.transform(test_values)

    train_dataset = SequenceDataset(
        train_scaled,
        window_size=config.window_size,
        target_col_idx=target_col_idx,
    )
    test_dataset = SequenceDataset(
        test_scaled,
        window_size=config.window_size,
        target_col_idx=target_col_idx,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    return (
        train_loader,
        test_loader,
        scaler,
        feature_cols,
        target_col_idx,
        train_scaled,
        test_scaled,
    )

