from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List


# Корінь проєкту: .../Crypto-forecasting
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Директорія для даних
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Шлях до DuckDB
DUCKDB_PATH = DATA_DIR / "crypto.duckdb"


class Settings(BaseSettings):
    """
    Централізований конфіг проєкту.
    Значення беремо з .env або змінних середовища.
    """

    # === API keys ===
    coingecko_api_key: str = Field(..., alias="COINGECKO_API_KEY")

    # === App defaults ===
    default_vs_currency: str = Field("usd", alias="DEFAULT_VS_CURRENCY")
    history_days_default: int = Field(60, alias="HISTORY_DAYS_DEFAULT")
    history_interval: str = Field("hourly", alias="HISTORY_INTERVAL")

    # 🆕 Список монет, за якими ми тягнемо історію
    tracked_coins: List[str] = Field(
        default_factory=lambda: [
            "bitcoin",
            "ethereum",
            "solana",
            "binancecoin",
            "ripple",
        ],
        alias="TRACKED_COINS",
    )

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,  # ігноруємо регістр в назвах змінних
    )


def get_settings() -> Settings:
    """
    Повертає новий інстанс Settings.

    ВАЖЛИВО:
    - ми не кешуємо settings, щоб у Streamlit зміни .env
      підхоплювалися при кожному rerun'і скрипта.
    """
    return Settings()


# Корисні "шорткати" для інших модулів, якщо треба
# (можемо використовувати їх або безпосередньо get_settings())
def get_default_vs_currency() -> str:
    return get_settings().default_vs_currency


def get_history_days_default() -> int:
    return get_settings().history_days_default


def get_history_interval() -> str:
    return get_settings().history_interval
