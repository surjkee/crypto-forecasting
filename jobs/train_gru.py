import argparse

from config.settings import get_settings
from models.gru.train import train_gru_for_coin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coin_id",
        type=str,
        default="bitcoin",
        help="ID монети (наприклад 'bitcoin')",
    )
    args = parser.parse_args()

    settings = get_settings()
    vs_currency = settings.default_vs_currency

    result = train_gru_for_coin(args.coin_id, vs_currency=vs_currency)
    print(
        f"✅ GRU для {args.coin_id}: MAE={result['mae']:.4f}, "
        f"RMSE={result['rmse']:.4f} {vs_currency.upper()}"
    )


if __name__ == "__main__":
    main()
