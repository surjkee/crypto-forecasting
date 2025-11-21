from config.settings import get_settings
from models.gru.train import train_gru_for_coin


def main():
    settings = get_settings()
    vs_currency = settings.default_vs_currency
    coins = settings.tracked_coins

    print(
        f"🚂 Запускаємо тренування GRU для монет: {', '.join(coins)} "
        f"({vs_currency})"
    )

    for coin_id in coins:
        print(f"\n=== GRU TRAIN: {coin_id} ===")
        try:
            result = train_gru_for_coin(coin_id, vs_currency=vs_currency)
            print(
                f"✅ {coin_id}: MAE={result['mae']:.4f}, "
                f"RMSE={result['rmse']:.4f} {vs_currency.upper()}"
            )
        except Exception as e:
            print(f"❌ Помилка для {coin_id}: {e}")


if __name__ == "__main__":
    main()
