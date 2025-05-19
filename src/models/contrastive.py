import torch
import torch.nn as nn

from .nn.encoder import Encoder
from .nn.head import Head
from ..utils.scheduler import cosine_increase_law


class Network(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.encoder = Encoder(model=config['model'], layer=config['encoder']['layer'], dim=config['encoder']['dim'], num_layers=config['encoder']['n_layers'], num_heads=config['encoder']['n_heads'], 
                               mlp_ratio=config['encoder']['mlp_ratio'], drop_rate=config['encoder']['drop_rate'], attn_drop_rate=config['encoder']['attn_drop_rate'], 
                               embedding = config['embedding'], norm_layer=config['encoder']['norm_layer'], act_layer=config['encoder']['act_layer'], batch_size=config['loader']['batch_size'])
        
        self.head = Head(in_dim=config['encoder']['dim'], out_dim=config['head']['out_dim'], use_bn=config['head']['use_bn'], norm_last_layer=config['head']['norm_last_layer'], 
                         nlayers=config['head']['n_layers'], hidden_dim=config['head']['hidden_dim'], bottleneck_dim=config['head']['bottleneck_dim'], drop_rate=config['encoder']['drop_rate'])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr if isinstance(data.x,torch.Tensor) else None
        x = self.encoder(x, edge_index, edge_attr)
        x = self.head(x)
        return x
        

class GraphCL(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.config = config
        self.branch1 = Network(self.config)
        self.branch2 = Network(self.config)
        

    def forward(self, data, mode=1):
        if mode == 1:
            branch1_out = self.branch1(data)
            return branch1_out
        elif mode == 2:
            branch2_out = self.branch2(data)
            return branch2_out 