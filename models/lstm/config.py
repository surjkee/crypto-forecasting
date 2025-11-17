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

    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    batch_size: int = 64
    num_epochs: int = 60
    learning_rate: float = 1e-3

    target_col: str = "price"

    def artifact_path(self, coin_id: str, vs_currency: str) -> Path:
        fname = f"lstm_{coin_id}_{vs_currency}.pt"
        return ARTIFACTS_DIR / fname
