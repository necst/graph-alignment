import torch.nn as nn
from torch_geometric.nn import global_max_pool, global_add_pool, global_mean_pool
import torch

# define wrappers to add this poolings as modules in the encoder

class MaxPoolLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_max_pool(x, batch)
    

class MeanPoolLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_mean_pool(x, batch)


class AddPoolLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        return global_add_pool(x, batch)
    

class VarPoolLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, batch):
        mean = global_mean_pool(x, batch)
        squared_diff = (x - mean[batch]) ** 2
        var = global_add_pool(squared_diff, batch) / global_add_pool(torch.ones_like(x), batch)
        return var
