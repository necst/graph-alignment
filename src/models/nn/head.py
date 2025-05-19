import torch.nn as nn
import torch


class Head(nn.Module):
    def __init__(self, in_dim=128, out_dim=64, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=64, bottleneck_dim=64, drop_rate=0.2):
        super().__init__()
        self.out_dim = out_dim
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        # the following five lines only work with recent versions of torch D:
        #self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        #with torch.no_grad():
        #    self.last_layer.parametrizations.weight.original0.fill_(1)
        #if norm_last_layer:
        #    self.last_layer.parametrizations.weight.original0.requires_grad = False

        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    

"""
class Head(nn.Module):
    def __init__(self, in_dim=128, out_dim=64, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=64, bottleneck_dim=64, drop_rate=0.2):
        super().__init__()
        self.out_dim = out_dim
        self.mlp1 = nn.Linear(in_dim, hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(drop_rate)
        self.mlp2 = nn.Linear(hidden_dim, bottleneck_dim)
        self.drop2 = nn.Dropout(drop_rate)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False


    def forward(self, x):
        x = self.mlp1(x)
        proj = self.drop1( self.act1(x) )
        x = self.drop2( self.mlp2(proj) )
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x, proj"""