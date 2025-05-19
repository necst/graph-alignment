import torch.nn as nn
import torch.nn.functional as F
import torch

class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels=64, num_layers=3, act_layer='GELU', norm_layer='BatchNorm1d', drop=0.2):
        super().__init__()
        self.norm = getattr(nn, norm_layer)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.act = getattr(nn, act_layer)()

        # Primo layer: input = 2 * in_channels
        self.layers.append(nn.Linear(in_channels * 2, hidden_channels))
        self.norms.append(self.norm(hidden_channels))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.norms.append(self.norm(hidden_channels))

        # Output layer
        self.layers.append(nn.Linear(hidden_channels, 1))

        self.dropout = drop

    def forward(self, z, edge_index, n_id = None):
                # batch.n_id: tensore degli ID globali dei nodi nel batch
        # Costruisci mappa da global_id → local_id
        #global_to_local = {nid.item(): i for i, nid in enumerate(n_id)}

        # edge_index ha shape [2, num_edges], con indici globali
        src, dst = edge_index[0], edge_index[1]

        # Rimappa a indici locali
        #src_local = torch.tensor([global_to_local[int(i)] for i in src], device=z.device)
      #  dst_local = torch.tensor([global_to_local[int(i)] for i in dst], device=z.device)
        
        edge_feat = torch.cat([z[src], z[dst]], dim=1)  # [batch_size, 2*in_channels]

        for i, layer in enumerate(self.layers[:-1]):
            edge_feat = layer(edge_feat)
            edge_feat = self.norms[i](edge_feat)
            edge_feat = self.act(edge_feat)
            edge_feat = F.dropout(edge_feat, p=self.dropout, training=self.training)

        edge_feat = self.layers[-1](edge_feat)
        return edge_feat.squeeze()
        #return torch.sigmoid(edge_feat).squeeze()
