# models/lstm/config.py

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class LSTMConfig:
    # базові параметри з нашого Colab-патерну
    window_size: int = 48          # довжина history-вікна
    train_ratio: float = 0.8       # частка train

    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.2

    batch_size: int = 64
    num_epochs: int = 200
    learning_rate: float = 3e-4

    target_col: str = "price"

     # Нові параметри:
    early_stopping_patience: int = 40   # скільки епох чекаємо без покращення
    early_stopping_min_delta: float = 1e-5

    lr_reduce_factor: float = 0.5       # у скільки разів зменшувати lr
    lr_reduce_patience: int = 15         # скільки епох без покращення до зменшення lr
    lr_min: float = 1e-6

    def artifact_path(self, coin_id: str, vs_currency: str) -> Path:
        fname = f"lstm_{coin_id}_{vs_currency}.pt"
        return ARTIFACTS_DIR / fname
