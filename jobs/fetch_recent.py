import sys
import os

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# -------------------------

from typing import Iterable

import pandas as pd

from config.settings import get_settings
from data.db import init_db, get_connection, load_ohlcv_hourly
from ingest.coingecko_client import CoinGeckoClient
from jobs.fetch_history import market_chart_to_df


def append_new_ohlcv(df_new: pd.DataFrame) -> None:
    """
    Додає нові рядки в ohlcv_hourly (без перезапису старих).
    Очікується, що df_new вже:
      - має колонки: coin_id, vs_currency, ts, price, market_cap, volume
      - не містить дублікатів по (coin_id, vs_currency, ts)
    """
    if df_new.empty:
        return

    init_db()
    con = get_connection()

    # Реєструємо df_new як тимчасову таблицю
    con.register("df_new", df_new)

    with con:
        con.execute(
            """
            INSERT INTO ohlcv_hourly (coin_id, vs_currency, ts, price, market_cap, volume)
            SELECT coin_id, vs_currency, ts, price, market_cap, volume
            FROM df_new
            """
        )

    con.close()


def fetch_recent_for_coin(
    client: CoinGeckoClient,
    coin_id: str,
    vs_currency: str,
    days: int,
    interval: str,
) -> None:
    """
    Інкрементальне оновлення історії для однієї монети:
    - дивимося останній ts у БД
    - тягнемо /market_chart з тими ж days та interval, що і fetch_history
    - додаємо тільки свічки, у яких ts > last_ts у БД
    """
    df_existing = load_ohlcv_hourly(coin_id, vs_currency)

    if df_existing.empty:
        print(f"⚠️ {coin_id}: в БД немає історії. Спочатку запусти jobs.fetch_history.")
        return

    settings = get_settings()
    LOCAL_TZ = getattr(settings, "timezone", "Europe/Kyiv")

    ts_db = df_existing["ts"]

    if ts_db.dt.tz is None:
        # локалізуємо як локальний час, вирішуємо неоднозначності
        ts_db = ts_db.dt.tz_localize(
            LOCAL_TZ,
            ambiguous="infer",        # спробувати реконструювати DST з монотонності
            nonexistent="shift_forward",  # якщо раптом буде "пропущена" година
        )
    else:
        ts_db = ts_db.dt.tz_convert(LOCAL_TZ)

    # далі все в UTC
    ts_db = ts_db.dt.tz_convert("UTC")
    last_ts_utc = ts_db.max()

    print(f"🔍 {coin_id}: останній ts у БД (UTC) = {last_ts_utc}")

    # Тягнемо такий самий зріз, як у fetch_history
    print(f"⬇️ {coin_id}: тягнемо /market_chart (days={days}, interval={interval})...")
    chart = client.get_market_chart(
        coin_id=coin_id,
        vs_currency=vs_currency,
        days=days,
        interval=interval,
    )

    df_api = market_chart_to_df(coin_id, vs_currency, chart)

    if df_api.empty:
        print(f"⚠️ {coin_id}: API повернув порожній DataFrame.")
        return

    ts_api = df_api["ts"]
    if ts_api.dt.tz is None:
        ts_api = ts_api.dt.tz_localize("UTC")
    else:
        ts_api = ts_api.dt.tz_convert("UTC")

    df_api["ts_utc"] = ts_api

    max_api_ts = df_api["ts_utc"].max()
    print(f"🔎 {coin_id}: останній ts з API (UTC) = {max_api_ts}")

    # Беремо лише свічки, які новіші за last_ts_utc
    df_new = df_api[df_api["ts_utc"] > last_ts_utc].copy()

    if df_new.empty:
        print(f"✅ {coin_id}: нових свічок немає, все актуально.")
        return

    # Перед вставкою прибираємо службову колонку ts_utc
    df_new = df_new.drop(columns=["ts_utc"])

    print(f"✅ {coin_id}: знайдено {len(df_new)} нових свічок, додаємо в БД...")
    append_new_ohlcv(df_new)


def fetch_recent_all(
    coins: Iterable[str] | None = None,
    vs_currency: str | None = None,
    days: int | None = None,
    interval: str | None = None,
) -> None:
    """
    Інкрементальне оновлення історії для набору монет.
    За замовчуванням бере those самі параметри, що й fetch_history:
      - Settings.tracked_coins
      - Settings.history_days_default
      - Settings.history_interval
    """
    settings = get_settings()

    if coins is None:
        coins = settings.tracked_coins

    vs_currency = vs_currency or settings.default_vs_currency
    days = days or settings.history_days_default
    interval = interval or settings.history_interval

    init_db()
    client = CoinGeckoClient()

    coins = list(coins)

    print(
        f"🚀 Інкрементальне оновлення історії для монет: "
        f"{', '.join(coins)} (vs_currency={vs_currency}, days={days}, interval={interval})"
    )

    for coin_id in coins:
        print(f"\n--- {coin_id} ---")
        try:
            fetch_recent_for_coin(
                client=client,
                coin_id=coin_id,
                vs_currency=vs_currency,
                days=days,
                interval=interval,
            )
        except Exception as e:
            print(f"❌ Помилка для {coin_id}: {e}")


def main():
    fetch_recent_all()


if __name__ == "__main__":
    main()
