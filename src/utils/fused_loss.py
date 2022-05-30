import torch
import torch.nn as nn
import torch.nn.functional as F


class ClsFusedLoss(nn.Module):
    def __init__(self, pos):
        super(ClsFusedLoss, self).__init__()
        self.pos = pos
        self.loss = nn.CrossEntropyLoss()
        
    def forward(self, y_embed_audio, y_embed_text, y_true, model):
        weight = model.fused_layer[0].weight
        y_pred_audio = F.linear(y_embed_audio, weight[:, :self.pos])
        y_pred_text = F.linear(y_embed_text, weight[:, self.pos:])
        
        l_audio = self.loss(y_pred_audio, y_true)
        l_text = self.loss(y_pred_text, y_true)
        return l_audio + l_text

class RegFusedLoss(nn.Module):
    def __init__(self, pos):
        super(RegFusedLoss, self).__init__()
        self.pos = pos
        self.loss = nn.SmoothL1Loss()
        
    def forward(self, y_embed_audio, y_embed_text, y_true, model):
        weight = model.fused_layer[0].weight
        y_pred_audio = F.linear(y_embed_audio, weight[:, :self.pos])
        y_pred_text = F.linear(y_embed_text, weight[:, self.pos:])
        y_true = y_true.view_as(y_pred_audio).float()
        
        l_audio = self.loss(y_pred_audio, y_true)
        l_text = self.loss(y_pred_text, y_true)
        return l_audio + l_text
