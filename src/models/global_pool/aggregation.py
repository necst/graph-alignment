import torch.nn as nn
import torch_scatter


class Gate(nn.Module):
    def __init__(self,in_features, hidden_features=128, norm_layer='LayerNorm', act_layer='GELU') -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.norm = getattr(nn, norm_layer)(hidden_features)
        self.act = getattr(nn, act_layer)()
        self.fc2 = nn.Linear(hidden_features, 1, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        return self.fc2(x)


class GlobalAttentionPooling(nn.Module):
    def __init__(self, in_features, hidden_features, norm_layer='LayerNorm', act_layer='GELU', temperature=0.4):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            getattr(nn, norm_layer)(hidden_features),
            getattr(nn, act_layer)(),
            nn.Linear(hidden_features, 1, bias=False),            
        )
        self.norm_out = getattr(nn, norm_layer)(hidden_features)
        self.temperature = temperature

    def forward(self, x, batch):
        # x shape: (total_nodes, in_features)
        # batch shape: (total_nodes,)
        
        w = self.project(x)  # (total_nodes, 1)
        
        # softmax over nodes in the same graph
        w_max = torch_scatter.scatter_max(w, batch, dim=0)[0]  # remove the max for stability when doing exponentiation
        w = w - w_max[batch]
        w = (w / self.temperature).exp()
        w_sum = torch_scatter.scatter_add(w, batch, dim=0)
        w = w / (w_sum[batch] + 1e-6)
        
        # weighted sum of node features and normalization
        out = torch_scatter.scatter_add(x * w, batch, dim=0)  # (num_graphs, in_features)
        return self.norm_out(out)