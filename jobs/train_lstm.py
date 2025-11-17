# jobs/train_lstm.py

import sys
import os
import argparse

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
# -------------------------

from models.lstm.train import train_lstm_for_coin
from config.settings import get_settings


def main():
    parser = argparse.ArgumentParser(description="Train LSTM model for a coin")
    parser.add_argument("--coin_id", type=str, default="bitcoin", help="CoinGecko coin id")
    parser.add_argument(
        "--vs_currency",
        type=str,
        default=None,
        help="Fiat currency (default from settings)",
    )

    args = parser.parse_args()
    settings = get_settings()
    vs_currency = args.vs_currency or settings.default_vs_currency

    result = train_lstm_for_coin(args.coin_id, vs_currency=vs_currency)

    print("\n=== Training summary ===")
    print(f"Coin:         {args.coin_id}")
    print(f"Currency:     {vs_currency}")
    print(f"Artifact:     {result['artifact_path']}")
    print(f"MAE:          {result['mae']:.4f} {vs_currency.upper()}")
    print(f"RMSE:         {result['rmse']:.4f} {vs_currency.upper()}")


if __name__ == "__main__":
    main()
