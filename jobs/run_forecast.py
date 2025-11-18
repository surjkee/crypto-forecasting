# jobs/run_forecast.py

import sys
import os

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# -------------------------

from config.settings import get_settings
from ui.constants import TRACKED_COINS
from models.lstm.inference import forecast_next_t1_and_store


def main():
    settings = get_settings()
    vs_currency = settings.default_vs_currency

    print(f"🔄 Запускаємо прогноз t+1 для всіх монет (vs_currency={vs_currency})...")

    for label, coin_id in TRACKED_COINS:
        try:
            print(f"  • {label} ({coin_id}) ...", end=" ", flush=True)
            result = forecast_next_t1_and_store(coin_id, vs_currency=vs_currency)
            print(
                f"OK — ts_forecast={result['ts_forecast']}, "
                f"y_pred={result['y_pred']:.2f} {vs_currency.upper()}"
            )
        except Exception as e:
            print(f"ERROR: {e}")

    print("✅ Готово.")


if __name__ == "__main__":
    main()
