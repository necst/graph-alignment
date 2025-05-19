import torch.nn as nn
from torch_geometric.nn import TransformerConv


class Attention(nn.Module):
    """
    Attention block based on TransformerConv PyG implementation
    """
    def __init__(self, dim=128, num_heads=2, attn_drop=0.2, proj_drop=0.2, edge_dim=5, beta=True, concat=True):
        super().__init__()
        self.attention = TransformerConv(
            in_channels=dim, 
            out_channels = dim // num_heads,
            heads = num_heads, 
            dropout = attn_drop,
            edge_dim = edge_dim, 
            beta = beta, 
            concat = concat
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, edge_index, edge_attr, return_attw=False):
        attn = self.attention(x, edge_index, edge_attr)
        x = self.proj_drop(self.proj(attn))
        return x
