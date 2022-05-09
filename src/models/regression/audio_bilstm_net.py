import torch
import torch.nn as nn


class AudioBiLSTMNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1,
        dropout: float = 0.5,
        num_layers: int = 2,
        hidden_size: int = 256,
        embed_size: int = 256
    ):
        super(AudioBiLSTMNet, self).__init__()

        self.attention_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size), nn.ReLU(inplace=True)
        )

        self.gru_layer = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.linear_layer = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(in_features=hidden_size, out_features=num_classes),
            nn.ReLU(),
        )

    def forward(self, x):
        x, _ = self.gru_layer(x)
        x = x.sum(dim=1)
        y = self.linear_layer(x)
        return y


if __name__ == "__main__":
    model = AudioBiLSTMNet()
    x = torch.randn((10, 3, 256))
    y = model(x)

    print(" input_size:{}\noutput_size:{}".format(x.shape, y.shape))
