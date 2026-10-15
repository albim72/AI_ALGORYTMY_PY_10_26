import torch
import torch.nn as nn


class ModelLSTM(nn.Module):
    def __init__(self, input_size=12, hidden_size=32, output_size=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [batch, czas, cechy]
        out, (h_n, c_n) = self.lstm(x)

        # bierzemy ostatni stan czasowy
        last = out[:, -1, :]

        # predykcja końcowa
        y = self.fc(last)

        return y


model = ModelLSTM()

x = torch.randn(32, 10, 12)
y = model(x)

print(y.shape)
