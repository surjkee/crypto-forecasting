from dataclasses import dataclass
from pathlib import Path

from config.settings import PROJECT_ROOT


@dataclass
class GRUConfig:
    # Дані / спліт
    window_size: int = 48
    train_ratio: float = 0.8
    target_col: str = "price"

    # Архітектура
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2

    # Тренування
    batch_size: int = 64
    num_epochs: int = 200          # max_epochs
    learning_rate: float = 3e-4

    # EarlyStopping
    early_stopping_patience: int = 40
    early_stopping_min_delta: float = 1e-5

    # ReduceLROnPlateau
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 15
    lr_min: float = 1e-6

    def get_artifact_path(self, coin_id: str, vs_currency: str) -> Path:
        """
        Шлях до файлу з вагами GRU-моделі.
        Аналогічно до LSTM, але з префіксом 'gru_'.
        Приклад:
        models/artifacts/gru_bitcoin_usd.pt
        """
        artifacts_dir = PROJECT_ROOT / "models" / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        fname = f"gru_{coin_id}_{vs_currency}.pt"
        return artifacts_dir / fname
