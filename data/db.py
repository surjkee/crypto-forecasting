from pathlib import Path
from typing import Optional
from datetime import datetime

import duckdb
import pandas as pd

from config.settings import PROJECT_ROOT, DATA_DIR, DUCKDB_PATH


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """
    Повертає підключення до DuckDB.
    """
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def init_db() -> None:
    """
    Ініціалізація схеми БД (якщо таблиць ще немає).
    """
    con = get_connection()

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv_hourly (
            coin_id TEXT,
            vs_currency TEXT,
            ts TIMESTAMP,
            price DOUBLE,
            market_cap DOUBLE,
            volume DOUBLE
        )
        """
    )

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_hourly (
            coin_id TEXT,
            vs_currency TEXT,
            ts_anchor TIMESTAMP,
            ts_forecast TIMESTAMP,
            model TEXT,
            y_pred DOUBLE,
            is_backfilled BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )


    con.close()


def load_ohlcv_hourly(
    coin_id: str,
    vs_currency: str,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Завантажує історичні дані з DuckDB у вигляді pandas.DataFrame
    для заданої монети та валюти.

    :param coin_id: id монети (наприклад, "bitcoin")
    :param vs_currency: валюта ("usd")
    :param limit: опціонально — обмеження кількості останніх рядків
    """
    init_db()
    con = get_connection(read_only=True)

    if limit is not None:
        query = """
            SELECT *
            FROM ohlcv_hourly
            WHERE coin_id = ? AND vs_currency = ?
            ORDER BY ts DESC
            LIMIT ?
        """
        df = con.execute(query, [coin_id, vs_currency, limit]).df()
        # повертаємо у зростаючому порядку часу
        df = df.sort_values("ts").reset_index(drop=True)
    else:
        query = """
            SELECT *
            FROM ohlcv_hourly
            WHERE coin_id = ? AND vs_currency = ?
            ORDER BY ts
        """
        df = con.execute(query, [coin_id, vs_currency]).df()

    con.close()
    return df

def ensure_forecast_continuity(
    coin_id: str,
    vs_currency: str,
    model: str,
    max_backfill_hours: int = 48,  # наприклад, останні 2 дні
):
    """
    Гарантує, що для кожного фактичного ts у ВІКНІ останніх max_backfill_hours
    існує рядок у forecast_hourly.

    Якщо для певного ts у цьому вікні немає прогнозу — створюється запис:
      ts_anchor   = ts
      ts_forecast = ts
      y_pred      = фактична ціна
      is_backfilled = TRUE
    """
    init_db()

    # 1) Фактичні дані
    df_price = load_ohlcv_hourly(coin_id, vs_currency)
    if df_price.empty:
        return

    df_price = df_price.copy()
    df_price["ts_hour"] = df_price["ts"].dt.floor("h")
    df_price = (
        df_price.sort_values("ts_hour")
        .drop_duplicates(subset=["ts_hour"], keep="last")
        .reset_index(drop=True)
    )

    last_fact_ts = df_price["ts_hour"].max()
    # обмежуємо backfill лише останніми max_backfill_hours
    window_start_ts = last_fact_ts - pd.Timedelta(hours=max_backfill_hours)

    df_price_win = df_price[df_price["ts_hour"] >= window_start_ts].copy()
    if df_price_win.empty:
        return

    # 2) Прогнози у цьому ж вікні
    con = get_connection(read_only=True)
    df_fc = con.execute(
        """
        SELECT ts_forecast
        FROM forecast_hourly
        WHERE coin_id = ?
          AND vs_currency = ?
          AND model = ?
          AND ts_forecast >= ?
          AND ts_forecast <= ?
        ORDER BY ts_forecast
        """,
        [coin_id, vs_currency, model, window_start_ts, last_fact_ts],
    ).df()
    con.close()

    existing_ts = set(pd.to_datetime(df_fc["ts_forecast"])) if not df_fc.empty else set()

    # 3) Знаходимо "дірки" тільки в межах цього вікна і заповнюємо їх фактом
    for _, row in df_price_win.iterrows():
        ts_hour = row["ts_hour"]
        if ts_hour not in existing_ts:
            fact_price = float(row["price"])
            store_hourly_forecast(
                coin_id=coin_id,
                vs_currency=vs_currency,
                ts_anchor=ts_hour,
                ts_forecast=ts_hour,
                model=model,
                y_pred=fact_price,
                is_backfilled=True,
            )




def store_hourly_forecast(
    coin_id: str,
    vs_currency: str,
    ts_anchor,
    ts_forecast,
    model: str,
    y_pred: float,
    is_backfilled: bool = False,
):
    """
    Зберігає прогноз у forecast_hourly.

    Якщо для (coin_id, vs_currency, ts_forecast, model) вже є запис —
    видаляємо його і вставляємо новий (останній вважається актуальним).
    """
    init_db()
    con = get_connection()

    with con:
        con.execute(
            """
            DELETE FROM forecast_hourly
            WHERE coin_id = ?
              AND vs_currency = ?
              AND ts_forecast = ?
              AND model = ?
            """,
            [coin_id, vs_currency, ts_forecast, model],
        )

        con.execute(
            """
            INSERT INTO forecast_hourly (
                coin_id, vs_currency,
                ts_anchor, ts_forecast,
                model, y_pred,
                is_backfilled,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                coin_id,
                vs_currency,
                ts_anchor,
                ts_forecast,
                model,
                float(y_pred),
                bool(is_backfilled),
            ],
        )

    con.close()


def load_last_hourly_forecast(
    coin_id: str,
    vs_currency: str,
    model: str = "lstm_v0.4",
):
    con = get_connection()
    query = """
        SELECT *
        FROM forecast_hourly
        WHERE coin_id = ?
          AND vs_currency = ?
          AND model = ?
        ORDER BY ts_forecast DESC, created_at DESC
        LIMIT 1
    """
    df = con.execute(query, [coin_id, vs_currency, model]).df()

    con.close()

    return df

def load_hourly_forecasts(
    coin_id: str,
    vs_currency: str,
    limit: Optional[int] = None,
    model: str = "lstm_v0.4",
) -> pd.DataFrame:
    """
    Завантажує історичні прогнози t+1 з таблиці forecast_hourly
    для заданої монети та валюти.

    :param coin_id: id монети (наприклад, "bitcoin")
    :param vs_currency: валюта ("usd")
    :param limit: опціонально — обмеження кількості останніх прогнозів
    :param model: назва моделі (для фільтрації, за замовчуванням 'lstm_v0.4')
    """
    init_db()
    con = get_connection(read_only=True)

    if limit is not None:
        query = """
            SELECT *
            FROM forecast_hourly
            WHERE coin_id = ?
              AND vs_currency = ?
              AND model = ?
            ORDER BY ts_forecast DESC
            LIMIT ?
        """
        df = con.execute(query, [coin_id, vs_currency, model, limit]).df()
        # повертаємо у зростаючому порядку часу
        df = df.sort_values("ts_forecast").reset_index(drop=True)
    else:
        query = """
            SELECT *
            FROM forecast_hourly
            WHERE coin_id = ?
              AND vs_currency = ?
              AND model = ?
            ORDER BY ts_forecast
        """
        df = con.execute(query, [coin_id, vs_currency, model]).df()

    con.close()
    return df
