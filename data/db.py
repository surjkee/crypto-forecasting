from pathlib import Path
from typing import Optional

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



