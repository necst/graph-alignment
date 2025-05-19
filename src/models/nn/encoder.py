import torch.nn as nn
import torch_geometric.nn as nng
from torch_geometric.nn import GINEConv, GCNConv, SAGEConv
import torch

from src.models.nn.link_predictor import LinkPredictor

from .attention import Attention
from .mlp import FeedForward, Mlp
from ..utils.regularization import DropPath
from ..utils.embedding import Embedding
from ..global_pool import *

class GIN(nn.Module):
    def __init__(self, dim=384, mlp_ratio=2., drop=0.25, drop_path=0.25, act_layer='GELU', norm_layer='BatchNorm1d'):
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp = FeedForward(in_dim=dim, hidden_dim=mlp_hidden_dim, out_dim=dim)
        self.conv = GINEConv(nn=mlp, train_eps=True, edge_dim=dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm = getattr(nn, norm_layer)(dim)
        self.act = getattr(nn, act_layer)()

    def forward(self, x, edge_index, edge_attr, return_layers=False):
        if not return_layers:
            x = x + self.drop_path( self.act( self.conv( self.norm( x ) , edge_index, edge_attr) ) )
        else:
            self.layers_emb = []
            res = self.drop_path( self.attn( self.norm1(x), edge_index, edge_attr ))
            self.layers_emb.append(res[:self.batch_size,:])
            x = x + res
            self.layers_emb.append(x[:self.batch_size,:])
        return x


class Block(nn.Module):
    """
    Residual transformer block 
    """
    def __init__(self, dim=384, num_heads=6, mlp_ratio=2., drop=0.25, attn_drop=0.25,
                 drop_path=0.25, act_layer='GELU', norm_layer='BatchNorm1d', batch_size=64) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.norm1 = getattr(nn, norm_layer)(dim)
        self.attn = Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop, edge_dim=dim, beta=False, concat=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = getattr(nn, norm_layer)(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x, edge_index, edge_attr, return_layers=False):
        if not return_layers:
            x = x + self.drop_path( self.attn( self.norm1(x), edge_index, edge_attr ) )
            x = x + self.drop_path( self.mlp( self.norm2(x) ) )
        elif return_layers:
            self.layers_emb = []
            attn = self.drop_path( self.attn( self.norm1(x), edge_index, edge_attr ) )
            self.layers_emb.append(attn[:self.batch_size,:])
            x = x + attn
            self.layers_emb.append(x[:self.batch_size,:])
            mlp =  self.drop_path( self.mlp( self.norm2(x) ) )
            self.layers_emb.append(mlp[:self.batch_size,:])
            x = x + mlp
            self.layers_emb.append(x[:self.batch_size,:])
        return x
    
class Encoder(nn.Module):
    def __init__(self, model='VGAE',layer='Transformer', dim=128, num_layers=6, num_heads=2, mlp_ratio=2., drop_rate=0.2, 
                 attn_drop_rate=0.2, embedding: dict = {}, norm_layer='BatchNorm1d', act_layer='GELU', node_feature_dim=5,
                 input_dim=1, output_dim=128, use_embedding=True, use_pre_norm=True, residual_connections=True, 
                 batch_size=64, scale_factor='none', dataset='BioKG') -> None:
        super().__init__()
        self.num_layers = num_layers
        self.model = model
        self.batch_size = batch_size
        self.layer_type = layer
        self.layers_emb = []
        self.dataset = dataset

        if layer == 'Transformer' or layer == "GIN":
            if dataset == 'BioKG':
                self.node_embedding = nn.Embedding(6, dim)
                self.edge_embedding = nn.Embedding(52, dim) 
            elif dataset == 'Proteins':
                self.node_embedding = nn.Linear(1, dim)
                self.edge_embedding = nn.Linear(8, dim) 
            if layer == 'Transformer':
                self.blocks = nn.ModuleList([
                        Block(
                            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.2, norm_layer=norm_layer, 
                            act_layer=act_layer, batch_size=batch_size
                        )
                        for _ in range(num_layers)
                    ]
                )
                
            elif layer == 'GIN':
                self.blocks = nn.ModuleList([
                        GIN(
                            dim=dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=0.2, norm_layer=norm_layer, act_layer=act_layer
                        )
                        for _ in range(num_layers)
                    ]
                )
            if model == "VGAE":
                if layer == "GIN":
                    self.conv_mu = GIN(
                            dim=dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=0.2, norm_layer=norm_layer, act_layer=act_layer
                        )
                    self.conv_logstd = GIN(
                            dim=dim, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=0.2, norm_layer=norm_layer, act_layer=act_layer
                        )
                else:
                    self.conv_mu = Block(
                            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.2, norm_layer=norm_layer, act_layer=act_layer, batch_size=batch_size
                        )
                    self.conv_logstd = Block(
                            dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0.2, norm_layer=norm_layer, act_layer=act_layer, batch_size=batch_size
                        )
        elif model == "Supervised":
            self.conv1 = getattr(nng, layer)(input_dim, dim)
            self.conv2 = getattr(nng, layer)(dim, output_dim)
            self.norm1 = getattr(nn, norm_layer)(dim)
            self.norm2 = getattr(nn, norm_layer)(output_dim)
            self.activation = getattr(nn, act_layer)()

        else:
            self.layers = nn.ModuleList()
            self.normalizations = nn.ModuleList()
            self.activation = getattr(nn, act_layer)()
            self.ConvLayer = getattr(nng, layer)
            self.scale_factor = scale_factor
            self.use_embedding = use_embedding
            self.use_pre_norm = use_pre_norm
            self.residual_connections = residual_connections
            self.input_dim = input_dim
            self.embedding_dim = dim
            self.output_dim = output_dim
            last = False

            if use_embedding:
                self.class_embedding = nn.Embedding(node_feature_dim, dim)
                self.input_dim = dim

            self.layers.append(self.ConvLayer(self.input_dim, self.compute_new_dim(self.input_dim, last)))
            curr_dim = self.compute_new_dim(self.input_dim, last)
            if use_pre_norm and use_embedding:
                self.pre_norm = getattr(nn,norm_layer)(dim)
                          
            for i in range(num_layers - 1):
                self.normalizations.append(getattr(nn,norm_layer)(curr_dim))
                if (i == num_layers - 2):
                    last = True
            self.layers.append(self.ConvLayer(curr_dim, self.compute_new_dim(curr_dim,last)))
            if self.residual_connections:
                self.normalizations.append(getattr(nn,norm_layer)(curr_dim))
            curr_dim = self.compute_new_dim(curr_dim, last)
            self.conv_mu = self.ConvLayer(output_dim, int(curr_dim / 2))
            self.conv_logstd = self.ConvLayer(output_dim, int(curr_dim / 2))

            # TODO local/global poolings

        self._init_weights()

    def compute_new_dim(self, current_dim, last):
        if last:
            return self.output_dim
        if current_dim < self.output_dim:
            return self.embedding_dim
        if self.scale_factor == "half":
            return int((current_dim + self.output_dim)/2)
        else:
            return current_dim    
        

    def _init_weights(self):
        # TODO define kaiming/xavier from config
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    

    def forward(self, x, edge_index, edge_attr, return_layers=False):
        if return_layers:
            layers_emb = []

        if self.layer_type == "GIN" or self.layer_type == "Transformer":
            # embedding phase
            x = self.node_embedding(x.to(torch.int64).squeeze()+1 if self.dataset == "BioKG" else x.to(torch.float32)) 
            edge_attr = self.edge_embedding(edge_attr.to(torch.int64).squeeze()+1 if self.dataset == "BioKG" else edge_attr.to(torch.float32))
            if return_layers:
                layers_emb.append(x[:self.batch_size,:])
            
            # convolutions
            for layer, blk in enumerate(self.blocks):
                x = blk(x, edge_index, edge_attr, return_layers=return_layers)
                if return_layers:
                    layers_emb.extend(blk.layers_emb) 
            
            if return_layers:
                if not hasattr(self, 'layers_emb') or len(self.layers_emb) == 0:
                    self.layers_emb = layers_emb  # Se è il primo batch, inizializza con i layer del batch corrente
                else:
                    self.layers_emb = [torch.cat([self.layers_emb[i], layers_emb[i]], dim=0) for i in range(len(self.layers_emb))]

            if self.model == "VGAE":
                return self.conv_mu(x, edge_index, edge_attr), self.conv_logstd(x, edge_index, edge_attr) 
            return x
        
        elif self.model == "VGAE" or self.model == "DeepGraphInfomax":
            if self.use_embedding:
                if x.size(dim = 1) == 1:
                    node_classes,_ = torch.max(x, dim = 1)
                else:
                    node_classes = torch.argmax(x, dim=1)
                x = self.class_embedding(node_classes.to(torch.int64))
                if return_layers:
                    self.layers_emb.append(x[:self.batch_size,:])
            if self.use_pre_norm and self.use_embedding:
                x = self.pre_norm(x)
                if return_layers:
                    self.layers_emb.append(x[:self.batch_size,:])
            if self.residual_connections:
                for i, conv in enumerate(self.layers):
                    if not return_layers:
                        x = x + self.activation(conv(self.normalizations[i](x), edge_index))
                    else:
                        res = self.activation(conv(self.normalizations[i](x), edge_index))
                        self.layers_emb.append(res[:self.batch_size,:])
                        x = x + res
                        self.layers_emb.append(x[:self.batch_size,:])
            else:
                for i, conv in enumerate(self.layers):
                    x = conv(x, edge_index)
                    if return_layers:
                        self.layers_emb.append(x[:self.batch_size,:])
                    if i < len(self.layers) - 1:
                        if self.normalizations[i] is not None:  # Normalizzazione, se definita
                            x = self.normalizations[i](x)
                            if return_layers:
                                self.layers_emb.append(x[:self.batch_size,:])
                        x = self.activation(x)  # Attivazione
            if (self.model == "DeepGraphInfomax"):
                return x
            return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
        elif self.model == "Supervised":
            x = self.conv1(x, edge_index)
            if return_layers:
                self.layers_emb.append(x[:self.batch_size,:])
            x = self.norm1(x)
            if return_layers:
                self.layers_emb.append(x[:self.batch_size,:])
            x = self.activation(x)
            if return_layers:
                self.layers_emb.append(x[:self.batch_size,:])
            
            x = self.conv2(x, edge_index)
            if return_layers:
                self.layers_emb.append(x[:self.batch_size,:])
            x = self.norm2(x)
            if return_layers:
                self.layers_emb.append(x[:self.batch_size,:])
            x = self.activation(x)
            if return_layers:
                self.layers_emb.append(x[:self.batch_size,:])
            
            return x




