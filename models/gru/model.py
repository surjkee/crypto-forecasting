import torch
from torch import nn


class GRUForecastModel(nn.Module):
    """
    Аналог LSTMForecastModel, але на базі GRU.
    На виході — скаляр (прогноз ціни) для останнього кроку вікна.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # У GRU dropout працює тільки якщо num_layers > 1
        effective_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, input_size)
        """
        out, _ = self.gru(x)             # (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]      # (batch, hidden_size)
        y = self.fc(last_hidden)         # (batch, 1)
        return y
