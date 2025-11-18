# jobs/train_lstm_all.py

import os
import sys

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.append(PROJECT_ROOT)
# -------------------------

from config.settings import get_settings
from ui.constants import TRACKED_COINS
from models.lstm.train import train_lstm_for_coin


def main():
    settings = get_settings()
    vs_currency = settings.default_vs_currency

    print(f"🚂 Запускаємо тренування LSTM для всіх монет (vs_currency={vs_currency})...")
    print(f"Монети у списку TRACKED_COINS: {', '.join([label for label, _ in TRACKED_COINS])}")

    for label, coin_id in TRACKED_COINS:
        print(f"\n=== {label} ({coin_id}) ===")
        try:
            result = train_lstm_for_coin(coin_id, vs_currency=vs_currency)

            mae = result.test_mae if hasattr(result, "test_mae") else None
            rmse = result.test_rmse if hasattr(result, "test_rmse") else None
            artifact_path = getattr(result, "artifact_path", None)

            print("✅ Тренування завершено.")
            if mae is not None and rmse is not None:
                print(f"   Test MAE:  {mae:.4f} {vs_currency.upper()}")
                print(f"   Test RMSE: {rmse:.4f} {vs_currency.upper()}")
            if artifact_path:
                print(f"   Модель збережено в: {artifact_path}")

        except Exception as e:
            print(f"❌ Помилка при тренуванні для {coin_id}: {e}")

    print("\n🎉 Всі монети оброблені.")


if __name__ == "__main__":
    main()
