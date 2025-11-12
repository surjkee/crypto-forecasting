import sys
import os

# ---- FIX IMPORT PATH ----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
# -------------------------

import requests
from typing import Any, Dict, List

from config.settings import get_settings


class CoinGeckoClient:
    """
    Клієнт для роботи з CoinGecko API, що використовує Settings.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
        api_key: str | None = None,
    ) -> None:
        settings = get_settings()

        # Якщо параметри не передано — беремо з Settings
        self.base_url = (base_url or "https://api.coingecko.com/api/v3").rstrip("/")
        self.timeout = timeout or 10
        self.api_key = api_key or settings.coingecko_api_key

        if not self.api_key:
            raise RuntimeError(
                "CoinGecko API key is empty. "
                "Перевір .env (COINGECKO_API_KEY) і налаштування Settings."
            )

    def _get(self, path: str, params: Dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"

        if params is None:
            params = {}

        # 🔑 Додаємо API key як query-параметр (рекомендований спосіб для Demo API)
        if self.api_key and "x_cg_demo_api_key" not in params:
            params["x_cg_demo_api_key"] = self.api_key

        # 🔑 І паралельно кладемо в headers (так теж дозволено)
        headers: Dict[str, str] = {}
        if self.api_key:
            headers["x-cg-demo-api-key"] = self.api_key

        response = requests.get(url, params=params, timeout=self.timeout, headers=headers)

        # 🔍 Тимчасове debug-логування при помилках
        if response.status_code != 200:
            try:
                err_json = response.json()
            except Exception:
                err_json = response.text

            print("=== CoinGecko DEBUG ===")
            print("URL:", response.url)
            print("Status code:", response.status_code)
            print("Response:", err_json)
            print("API key prefix:", repr(self.api_key[:6]) + "..." if self.api_key else "<empty>")
            print("API key length:", len(self.api_key) if self.api_key else 0)
            print("=======================")

            response.raise_for_status()

        return response.json()


    # -------- API METHODS --------

    def ping(self) -> Dict[str, Any]:
        """Перевірка доступності API."""
        return self._get("/ping")

    def get_top_coins(
        self,
        vs_currency: str | None = None,
        per_page: int = 50,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        settings = get_settings()
        vs_currency = vs_currency or settings.default_vs_currency

        params = {
            "vs_currency": vs_currency,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "sparkline": False,
        }
        return self._get("/coins/markets", params=params)

    def get_market_chart(
        self,
        coin_id: str,
        vs_currency: str | None = None,
        days: int | None = None,
        interval: str | None = None,
    ) -> Dict[str, Any]:
        settings = get_settings()

        vs_currency = vs_currency or settings.default_vs_currency
        days = days or settings.history_days_default
        interval = interval or settings.history_interval

        # --- Нормалізація інтервалу під обмеження CoinGecko ---
        # Якщо хочемо погодинні дані, але не маємо Enterprise:
        # - для days 2–90: просто не вказуємо interval → CoinGecko сам дає hourly
        # - для days < 2: піднімаємо до 2, і теж без interval
        interval_param: str | None = interval

        if interval == "hourly":
            interval_param = None  # не передаємо interval в API

        params: Dict[str, Any] = {
            "vs_currency": vs_currency,
            "days": days,
        }
        if interval_param is not None:
            params["interval"] = interval_param

        return self._get(f"/coins/{coin_id}/market_chart", params=params)


