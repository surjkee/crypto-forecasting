# jobs/run_forecast.py

import os
import sys

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# -------------------------

from config.settings import get_settings
from ui.constants import TRACKED_COINS

from models.lstm.inference import (
    forecast_next_t1_and_store as forecast_next_t1_and_store_lstm,
)
from models.gru.inference import (
    forecast_next_t1_and_store as forecast_next_t1_and_store_gru,
)


def main() -> None:
    settings = get_settings()
    vs_currency = settings.default_vs_currency

    print(f"🚀 Запускаємо прогноз t+1 для всіх монет (vs_currency={vs_currency})...")

    for label, coin_id in TRACKED_COINS:
        print(f"\n• {label} ({coin_id})")

        # --- LSTM ---
        try:
            print("   LSTM ... ", end="", flush=True)
            res_lstm = forecast_next_t1_and_store_lstm(
                coin_id=coin_id,
                vs_currency=vs_currency,
                model_name="lstm_v1.0",
            )
            print(
                f"OK — ts_forecast={res_lstm['ts_forecast']}, "
                f"y_pred={res_lstm['y_pred']:.2f} {vs_currency.upper()}"
            )
        except Exception as e:
            print(f"   ❌ LSTM: помилка для {coin_id}: {e}")

        # --- GRU ---
        try:
            print("   GRU  ... ", end="", flush=True)
            res_gru = forecast_next_t1_and_store_gru(
                coin_id=coin_id,
                vs_currency=vs_currency,
                model_name="gru_v1.0",
            )
            print(
                f"OK — ts_forecast={res_gru['ts_forecast']}, "
                f"y_pred={res_gru['y_pred']:.2f} {vs_currency.upper()}"
            )
        except Exception as e:
            print(f"   ❌ GRU: помилка для {coin_id}: {e}")

    print("\n✅ Готово.")


if __name__ == "__main__":
    main()
