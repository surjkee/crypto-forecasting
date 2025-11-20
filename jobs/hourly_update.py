import os
import sys

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# -------------------------

from config.settings import get_settings
from jobs.fetch_recent import fetch_recent_all
from jobs import run_forecast


def main():
    settings = get_settings()
    vs_currency = settings.default_vs_currency

    print("🚀 Запускаємо погодинне оновлення даних та прогнозів...\n")

    # 1) Оновлюємо історію (тільки нові свічки)
    print("📥 Крок 1: інкрементальне оновлення історії (fetch_recent_all)...")
    fetch_recent_all(vs_currency=vs_currency)
    print("\n✅ Історія оновлена.\n")

    # 2) Рахуємо t+1 прогнози для всіх монет
    print("🔮 Крок 2: побудова t+1 прогнозів для всіх монет (run_forecast)...")
    run_forecast.main()
    print("\n✅ Прогнози оновлені.\n")

    print("🏁 Погодинне оновлення завершено.")


if __name__ == "__main__":
    main()
