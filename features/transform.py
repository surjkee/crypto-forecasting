import pandas as pd
import numpy as np
from typing import Sequence


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Додає до DataFrame колонки з:
    - простими відсотковими змінами (return)
    - логарифмічними змінами (log_return)
    """
    df = df.copy()

    df["return"] = df["price"].pct_change()
    df["log_return"] = np.log(df["price"] / df["price"].shift(1))

    return df


def add_sma(df: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    """
    Додає прості ковзні середні (SMA) для заданих вікон.
    Приклад: windows = [5, 10, 20]
    """
    df = df.copy()

    for w in windows:
        col = f"sma_{w}"
        df[col] = df["price"].rolling(window=w, min_periods=1).mean()

    return df


def add_ema(df: pd.DataFrame, windows: Sequence[int]) -> pd.DataFrame:
    """
    Додає експоненційні ковзні середні (EMA).
    """
    df = df.copy()

    for w in windows:
        col = f"ema_{w}"
        df[col] = df["price"].ewm(span=w, adjust=False, min_periods=1).mean()

    return df


def add_volatility(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Додає ковзну волатильність (стандартне відхилення)
    за вікном window (у кількості спостережень).
    За замовчуванням: 24 → добова волатильність для погодинних даних.
    """
    df = df.copy()

    # Використаємо log_return для оцінки волатильності
    if "log_return" not in df.columns:
        df = add_returns(df)

    df[f"volatility_{window}"] = (
        df["log_return"].rolling(window=window, min_periods=1).std()
    )

    return df


def build_feature_frame(
    df: pd.DataFrame,
    sma_windows: Sequence[int] = (5, 20, 50),
    ema_windows: Sequence[int] = (10, 50),
    volatility_window: int = 24,
) -> pd.DataFrame:
    """
    Повний пайплайн побудови фіч:
    - returns, log_returns
    - SMA
    - EMA
    - rolling volatility

    Повертає новий DataFrame з усіма колонками.
    """
    feat_df = df.copy()
    feat_df = add_returns(feat_df)
    feat_df = add_sma(feat_df, sma_windows)
    feat_df = add_ema(feat_df, ema_windows)
    feat_df = add_volatility(feat_df, volatility_window)

    return feat_df
