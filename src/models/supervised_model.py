import torch
import torch.nn as nn

class SupervisedLinkModel(nn.Module):
    def __init__(self, encoder: nn.Module, predictor: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor

    def forward(self, x, edge_index, edge_attr, edge_label_index):
        # 1. Calcola le embedding dei nodi
        z = self.encoder(x, edge_index, edge_attr)

        # 2. Predici la probabilità per ogni coppia (u,v)
        return self.predictor(z, edge_label_index)