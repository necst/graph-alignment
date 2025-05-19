import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    """
    Traditional MLP implementation
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer='GELU', norm_layer='BatchNorm1d', drop=0.2):
        super().__init__()
        self.in_features = in_features
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        if norm_layer != None:
            self.norm = getattr(nn, norm_layer)(hidden_features)
        self.act = getattr(nn, act_layer)()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)


    def forward(self, x):
        x = self.fc1(x)
        if hasattr(self, 'norm_layer'):
            x = self.norm(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
    
    
class FeedForward(nn.Module):
    """
    FeedForward architecture based on SwiGLU activation from LLama 3
    https://github.com/meta-llama/llama3/

    # TODO 
        - test activation with parametric Beta param in the sigmoid
    """
    def __init__(self, in_dim: int, hidden_dim: int = None, out_dim : int = None, drop : float = 0.2):
        super().__init__()
        self.in_features = in_dim
        hidden_dim = hidden_dim or in_dim 
        out_dim = out_dim or in_dim
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.d2 = nn.Dropout(drop)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.d3 = nn.Dropout(drop)


    def forward(self, x):
        return self.d2(self.w2(F.silu(self.w1(x)) * self.d3(self.w3(x))))    # dropout only after the activation 
                       