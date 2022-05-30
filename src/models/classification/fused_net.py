import torch
import torch.nn as nn


class FusedNet(nn.Module):
    def __init__(self,
                 text_hidden_dim, audio_hidden_dim, num_classes):
        super(FusedNet, self).__init__()
        self.fused_layer = nn.Sequential (
            nn.Linear(in_features=text_hidden_dim + audio_hidden_dim, out_features=num_classes, bias=False),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.fused_layer(x)
