import sys
import os

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
# -------------------------

from typing import Literal, Iterable

import pandas as pd

from ingest.coingecko_client import CoinGeckoClient
from config.settings import get_settings
from data.db import init_db, get_connection



def market_chart_to_df(
    coin_id: str,
    vs_currency: str,
    chart: dict,
) -> pd.DataFrame:
    """
    Перетворює відповідь CoinGecko /market_chart у табличний формат OHLCV.
    (Тут у нас не повний OHLC, а price + market_cap + volume по таймстемпах.)
    """
    prices = chart.get("prices", [])
    caps = chart.get("market_caps", [])
    vols = chart.get("total_volumes", [])

    # Робимо окремі DataFrame'и
    df_prices = pd.DataFrame(prices, columns=["ts_ms", "price"])
    df_caps = pd.DataFrame(caps, columns=["ts_ms", "market_cap"])
    df_vols = pd.DataFrame(vols, columns=["ts_ms", "volume"])

    # Мерджимо по ts_ms
    df = df_prices.merge(df_caps, on="ts_ms", how="left").merge(df_vols, on="ts_ms", how="left")

    # Конвертуємо мс → datetime (UTC)
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"])

    # Додаємо ідентифікатори
    df["coin_id"] = coin_id
    df["vs_currency"] = vs_currency

    # Вирівнюємо порядок колонок
    df = df[["coin_id", "vs_currency", "ts", "price", "market_cap", "volume"]]

    # Сортуємо за часом
    df = df.sort_values("ts").reset_index(drop=True)

    return df


def store_history_df(df: pd.DataFrame) -> None:
    """
    Зберігає DataFrame з історичними даними в DuckDB.
    Стратегія проста:
    - Видаляємо старі записи для (coin_id, vs_currency)
    - Вставляємо все знову
    """
    if df.empty:
        print("⚠️ DataFrame порожній, нічого зберігати.")
        return

    init_db()
    con = get_connection()

    coin_id = df["coin_id"].iloc[0]
    vs_currency = df["vs_currency"].iloc[0]

    # Видаляємо старі дані для цієї монети в цій валюті
    con.execute(
        """
        DELETE FROM ohlcv_hourly
        WHERE coin_id = ? AND vs_currency = ?
        """,
        [coin_id, vs_currency],
    )

    # Реєструємо df як тимчасову таблицю і вставляємо
    con.register("df_hist", df)
    con.execute(
        """
        INSERT INTO ohlcv_hourly BY NAME
        SELECT * FROM df_hist
        """
    )

    con.close()
    print(f"✅ Збережено {len(df)} рядків для {coin_id} / {vs_currency}.")


def fetch_and_store_history(
    coin_id: str = "bitcoin",
    vs_currency: str | None = None,
    days: int | None = None,
    interval: Literal["hourly", "daily"] | None = None,
) -> None:
    """
    Витягує історичні дані з CoinGecko і зберігає їх у DuckDB.
    """
    settings = get_settings()

    if vs_currency is None:
        vs_currency = settings.default_vs_currency
    if days is None:
        days = settings.history_days_default
    if interval is None:
        interval = settings.history_interval  # type: ignore[assignment]

    client = CoinGeckoClient()

    print(
        f"⬇️ Завантажуємо історію для {coin_id} ({vs_currency}), "
        f"days={days}, interval={interval}..."
    )

    chart = client.get_market_chart(
        coin_id=coin_id,
        vs_currency=vs_currency,
        days=days,
        interval=interval,  # type: ignore[arg-type]
    )

    df = market_chart_to_df(coin_id, vs_currency, chart)

    print(df.head())
    print(f"... {len(df)} рядків у DataFrame")

    store_history_df(df)

def fetch_and_store_all_history(
    coins: Iterable[str] | None = None,
    vs_currency: str | None = None,
    days: int | None = None,
    interval: Literal["hourly", "daily"] | None = None,
) -> None:
    """
    Тягне історію одразу для набору монет (список з Settings.tracked_coins за замовчуванням).
    """
    settings = get_settings()

    if coins is None:
        coins = settings.tracked_coins

    vs_currency = vs_currency or settings.default_vs_currency
    days = days or settings.history_days_default
    interval = interval or settings.history_interval  # type: ignore[assignment]

    coins = list(coins)
    print(
        f"⬇️ Завантажуємо історію для монет: {', '.join(coins)} "
        f"({vs_currency}), days={days}, interval={interval}"
    )

    for coin_id in coins:
        print(f"\n--- {coin_id} ---")
        try:
            fetch_and_store_history(
                coin_id=coin_id,
                vs_currency=vs_currency,
                days=days,
                interval=interval,  # type: ignore[arg-type]
            )
        except Exception as e:
            print(f"⚠️ Помилка при завантаженні {coin_id}: {e}")


if __name__ == "__main__":
    fetch_and_store_all_history()
