import torch
import torch.nn as nn


class ModelGRU(nn.Module):
    def __init__(self, input_size=12, hidden_size=32, output_size=1):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, czas, cechy]
        out, h_n = self.gru(x)

        # ostatni krok czasowy
        last = out[:, -1, :]

        y = self.fc(last)

        return y


model = ModelGRU()

x = torch.randn(32, 10, 12)
y = model(x)

print(y.shape)
