import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import degree


from ..nn.mlp import FeedForward, Mlp


class Embedding(nn.Module):
    def __init__(self, dim, options: dict = {}, method: str = 'cat', use_norm: bool = True, norm_layer: str = 'BatchNorm1d'):
        super().__init__()
        self.method = method
        self.dim = dim
        self.use_norm = use_norm
        self.node_features = options['node_features']
        self.edge_features = options['edge_features']

        # iterate to add learnable embeddings for each feature
        # number of classes for each feature has +1 since we also need to consider 
        # that we need a learnable token when we mask the corresponding feature
        if method == 'cat' and self.node_features != 1:
            node_feature_size = dim//self.node_features   
            self.node_embeddings = nn.ModuleList([
                nn.Embedding(int(torch.round(options['max_node_features'][i]))+1, node_feature_size)
                for i in range(len(options['max_node_features']))])
        elif method == 'sum':
            self.node_embeddings = nn.ModuleList([
                nn.Embedding(int(torch.round(options['max_node_features'][i]))+1, dim)
                for i in range(len(options['max_node_features']))])
        elif self.node_features == 1:
            self.node_embeddings = nn.Embedding(int(options['max_node_features']+1), dim)
        
        if method == 'cat' and self.edge_features != 1:
            edge_feature_size = dim//self.edge_features 
            self.edge_embeddings = nn.ModuleList([
                nn.Embedding(int(torch.round(options['max_edge_features'][i]))+1, edge_feature_size)
                for i in range(len(options['max_edge_features']))])
        elif method == 'sum':
            self.edge_embeddings = nn.ModuleList([
                nn.Embedding(int(torch.round(options['max_edge_features'][i]))+1, dim)
                for i in range(len(options['max_edge_features']))])
        elif self.edge_features == 1:
            self.edge_embeddings = nn.Embedding(int(options['max_edge_features']+1), dim)

        if use_norm:
            self.node_norm = getattr(nn, norm_layer)(dim)
            self.edge_norm = getattr(nn, norm_layer)(dim)

    def forward(self, x, edge_attr):
        if self.method == 'cat':
            x, edge_attr = self._concat_embedding(x, edge_attr)
        elif self.method == 'sum':
            x, edge_attr = self._sum_embedding(x, edge_attr)
        return x, edge_attr
    
    
    def _concat_embedding(self, x, edge_attr):
        if self.node_features != 1:
            x = torch.cat([
                embedding(
                    torch.where(
                        x[:, i] == -1,
                        torch.tensor(0, device=x.device),  # Use 0 as padding index
                        (x[:, i] * embedding.num_embeddings).long() + 1
                    )
                ) if embedding.num_embeddings > 3 else
                embedding(
                    torch.where(
                        x[:, i] == -1,
                        torch.tensor(0, device=x.device),  # Use 0 as padding index
                        (x[:, i]).long() + 1
                    )
                )
                for i, embedding in enumerate(self.node_embeddings)
            ], dim=-1)
            if self.use_norm:
                x = self.node_norm(x)
        else:
            x = self.node_embeddings(x.squeeze())

        if self.edge_features != 1:
            edge_attr = torch.cat([
                (
                    embedding(
                        torch.where(
                            edge_attr[:, i] == -1,
                            torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                            (edge_attr[:, i] * embedding.num_embeddings).long() +1
                        )
                    ) if embedding.num_embeddings > 3
                    else embedding(
                        torch.where(
                            edge_attr[:, i] == -1,
                            torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                            edge_attr[:, i].long() + 1
                        )
                    )
                ) if isinstance(embedding, nn.Embedding)
                else embedding(edge_attr[:, i].view(-1, 1)) if isinstance(embedding, nn.Linear)
                else embedding(edge_attr[:]) if isinstance(embedding, FeedForward)
                else embedding(
                    torch.where(
                        edge_attr[:, i] == -1,
                        torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                        edge_attr[:, i].long() + 1
                    )
                )
                for i, embedding in enumerate(self.edge_embeddings)
            ], dim=-1)
            if self.use_norm:
                edge_attr = self.edge_norm(edge_attr)
        else:
            edge_attr = self.edge_embeddings(edge_attr.squeeze())
        return x, edge_attr
    
    def _sum_embedding(self, x, edge_attr):
        x = torch.sum(
            torch.stack(
                [
                    embedding(
                        torch.where(
                            x[:, i] == -1,
                            torch.tensor(0, device=x.device),  # Use 0 as padding index
                            (x[:, i] * embedding.num_embeddings).long() + 1
                        )
                    ) if embedding.num_embeddings > 3 else
                    embedding(
                        torch.where(
                            x[:, i] == -1,
                            torch.tensor(0, device=x.device),  # Use 0 as padding index
                            (x[:, i]).long() + 1
                        )
                    )
                    for i, embedding in enumerate(self.node_embeddings)
                ], dim = 0
            ), dim = 0
        )
        if self.use_norm:
            x = self.edge_norm(x)

        edge_attr = torch.sum(
            torch.stack(
                [
                    (
                        embedding(
                            torch.where(
                                edge_attr[:, i] == -1,
                                torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                                (edge_attr[:, i] * embedding.num_embeddings).long() +1
                            )
                        ) if embedding.num_embeddings > 3
                        else embedding(
                            torch.where(
                                edge_attr[:, i] == -1,
                                torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                                edge_attr[:, i].long() + 1
                            )
                        )
                    ) if isinstance(embedding, nn.Embedding)
                    else embedding(edge_attr[:, i].view(-1, 1)) if isinstance(embedding, nn.Linear)
                    else embedding(edge_attr[:]) if isinstance(embedding, FeedForward)
                    else embedding(
                        torch.where(
                            edge_attr[:, i] == -1,
                            torch.tensor(0, device=edge_attr.device),  # Use 0 as padding index
                            edge_attr[:, i].long() + 1
                        )
                    )
                    for i, embedding in enumerate(self.edge_embeddings)
                ], dim = 0
            ), dim = 0
        )
        if self.use_norm:
            edge_attr = self.edge_norm(edge_attr)
        return x, edge_attr