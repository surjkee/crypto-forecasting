# models/lstm/model.py

import torch
from torch import nn


class LSTMForecastModel(nn.Module):
    """
    Простий LSTM → Linear для one-step ahead прогнозу.
    Формат вхідних даних: (batch, seq_len, input_size)
    Вихід: (batch, 1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)      # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]  # беремо останній timestep
        y_hat = self.fc(last_hidden) # (batch, 1)
        return y_hat
