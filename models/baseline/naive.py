# models/baseline/naive.py

from typing import Tuple
import pandas as pd
import numpy as np


def naive_constant_forecast(
    history: pd.DataFrame,
    horizon_hours: int = 24,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Дуже простий baseline:

    - беремо останню відому ціну з history['price']
    - прогнозуємо її як константу на наступні horizon_hours годин

    Повертає:
    - DataFrame з колонками ['ts', 'y_pred']
    - окремо Series з останньою історичною ціною (для зручності, але можемо не використовувати)
    """
    if history.empty:
        raise ValueError("History is empty, cannot build forecast")

    last_ts = history["ts"].max()
    last_price = history["price"].iloc[-1]

    # Генеруємо горизонт (тут просто рівномірно по годинах)
    ts_future = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="H",
    )

    y_pred = np.full(shape=horizon_hours, fill_value=last_price, dtype=float)

    df_forecast = pd.DataFrame({"ts": ts_future, "y_pred": y_pred})

    return df_forecast, pd.Series([last_price], index=["last_price"])
