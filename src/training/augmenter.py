import torch
import random
from torch_geometric.data import Batch


class Augmenter():
    def __init__(self, node_drop_rate=0.2, edge_drop_rate=0.2, subgraph_size=0.2, n_views:int = 2, n_global_views:int = 1, device='cuda'):
        self.node_drop_rate = node_drop_rate
        self.edge_drop_rate = edge_drop_rate
        self.subgraph_size = subgraph_size
        self.n_views = n_views
        self.n_global_views = n_global_views
        self.device = device

    @staticmethod
    def _mask_node_features(batch: Batch, feature_mask_rate: float, device):
        """
        Masks a random percentage of all node features for graph data augmentation by replacing
        them with -1.

        Args:
        - batch (Batch): A PyTorch Geometric Batch object containing the graph data, cloned from DataLoader generated batch.
        - feature_mask_rate (float): The percentage of all node features to mask, between 0 and 1.

        Returns:
        - batch (Batch): The augmented graph batch with masked node features.
        """
        if batch.x is None:
            return batch

        total_features = batch.x.numel()
        num_features_to_mask = int(feature_mask_rate * total_features)

        # Create a flat mask tensor
        mask = torch.zeros(total_features, dtype=torch.bool, device=device)
        mask[:num_features_to_mask] = True
        mask = mask[torch.randperm(total_features, device=device)]

        # Reshape the mask to match batch.x shape
        mask = mask.reshape(batch.x.shape)

        # Apply the mask
        batch.x[mask] = -1

        return batch
    
    @staticmethod
    def _mask_edge_features(batch : Batch, feature_mask_rate : float, device):
        """
        Masks a random percentage of all edge features for graph data augmentation by replacing
        them with -1.

        Args:
        - batch (Batch): A PyTorch Geometric Batch object containing the graph data, cloned from DataLoader generated batch.
        - feature_mask_rate (float): The percentage of all edge features to mask, between 0 and 1.

        Returns:
        - batch (Batch): The augmented graph batch with masked edge features.
        """
        if batch.edge_attr is None:
            return batch

        total_features = batch.edge_attr.numel()
        num_features_to_mask = int(feature_mask_rate * total_features)

        # Create a flat mask tensor
        mask = torch.zeros(total_features, dtype=torch.bool, device=device)
        mask[:num_features_to_mask] = True
        mask = mask[torch.randperm(total_features, device=device)]

        # Reshape the mask to match batch.edge_attr shape
        mask = mask.reshape(batch.edge_attr.shape)

        # Apply the mask
        batch.edge_attr[mask] = -1

        return batch
    
    @staticmethod
    def _subgraph_removal(graph_batch, p_subgraph_rem=0.1, device='cuda', protected_nodes=None):
        """
        removes subgraphs while preserving specified protected nodes and their connections.
        
        parameters:
            graph_batch (data): a pytorch geometric data object.
            p_subgraph_rem (float): maximum fraction of nodes to be considered for removal.
            protected_nodes (torch.tensor): indices of nodes that should not be removed.
            device (str): device on which to perform operations.
        
        returns:
            data: a new pytorch geometric data object with subgraphs removed.
        """
        # randomize removal percentage
        p_subgraph_rem = random.uniform(0.0, p_subgraph_rem)
        
        graph_batch = graph_batch.to(device)
        num_nodes = graph_batch.num_nodes
        
        # initialize protection mask
        protected_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        if protected_nodes is not None:
            protected_nodes = protected_nodes.to(device)
            protected_mask[protected_nodes] = True
        
        # select random start nodes (excluding protected nodes)
        available_nodes = torch.arange(num_nodes, device=device)[~protected_mask]
        num_start_nodes = max(1, int(len(available_nodes) * 0.01))  # sample ~1% of available nodes
        if len(available_nodes) > 0:
            start_node_indices = torch.randint(0, len(available_nodes), (num_start_nodes,), device=device)
            start_nodes = available_nodes[start_node_indices]
        else:
            start_nodes = torch.tensor([], device=device)
        
        # find neighbors of selected nodes
        if len(start_nodes) > 0:
            mask = (graph_batch.edge_index[0].unsqueeze(1) == start_nodes).any(dim=1) | \
                (graph_batch.edge_index[1].unsqueeze(1) == start_nodes).any(dim=1)
            neighbor_nodes = torch.unique(torch.cat([
                graph_batch.edge_index[0][mask],
                graph_batch.edge_index[1][mask]
            ]))
            
            # remove protected nodes from neighbor_nodes
            neighbor_nodes = neighbor_nodes[~protected_mask[neighbor_nodes]]
            
            # determine number of nodes to remove
            num_to_remove = int(neighbor_nodes.size(0) * p_subgraph_rem)
            if num_to_remove > 0:
                to_remove = neighbor_nodes[torch.randperm(neighbor_nodes.size(0), device=device)[:num_to_remove]].long()
            else:
                to_remove = torch.tensor([], device=device, dtype=torch.long)
        else:
            to_remove = torch.tensor([], device=device, dtype=torch.long)
        
        # create a mask for remaining nodes
        remaining_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
        remaining_mask[to_remove] = False
        
        # ensure protected nodes are kept
        remaining_mask[protected_mask] = True
        
        # filter node features
        new_x = graph_batch.x[remaining_mask] if hasattr(graph_batch, 'x') else None
        
        # filter edges where both nodes remain
        remaining_nodes = torch.arange(num_nodes, device=device)[remaining_mask]
        edge_mask = remaining_mask[graph_batch.edge_index[0]] & remaining_mask[graph_batch.edge_index[1]]
        new_edge_index = graph_batch.edge_index[:, edge_mask]
        
        # adjust edge indices after node removal
        new_edge_index = torch.cat([
            torch.searchsorted(remaining_nodes, new_edge_index[0]).unsqueeze(0),
            torch.searchsorted(remaining_nodes, new_edge_index[1]).unsqueeze(0)
        ], dim=0)
        
        # filter edge attributes if they exist
        new_edge_attr = graph_batch.edge_attr[edge_mask] if hasattr(graph_batch, 'edge_attr') else None
        
        # create new data object
        new_data = Batch(
            edge_index=new_edge_index,
            num_nodes=remaining_mask.sum().item()
        )
        
        if new_x is not None:
            new_data.x = new_x
        if new_edge_attr is not None:
            new_data.edge_attr = new_edge_attr
        

        return new_data.to(device)

    @staticmethod
    def _edge_dropping(graph, ratio, device="cuda", min_edges_per_protected=1, protected_nodes=None):
        """
        Randomly drops edges while ensuring protected nodes maintain connectivity.
        
        Parameters:
            graph (Batch): A PyTorch Geometric Batch object.
            ratio (float): The maximum fraction of edges to remove.
            protected_nodes (torch.Tensor): Indices of nodes that should maintain connectivity.
            min_edges_per_protected (int): Minimum number of edges to maintain for each protected node.
            device (str): Device to perform computations on (default: 'cuda').
        
        Returns:
            Batch: A new PyTorch Geometric Batch object with some edges removed.
        """
        # Randomize drop ratio
        ratio = random.uniform(0.0, ratio)
        graph = graph.to(device)
        total_edges = graph.edge_index.size(1)
        
        # Initialize mask for all edges
        mask = torch.rand(total_edges, device=device) > ratio
        
        if protected_nodes is not None:
            protected_nodes = protected_nodes.to(device)
            
            # Find edges connected to protected nodes
            protected_edges_mask = torch.zeros_like(mask, dtype=torch.bool)
            
            for node in protected_nodes:
                # Find edges where protected node is either source or target
                node_edges = (graph.edge_index[0] == node) | (graph.edge_index[1] == node)
                
                if node_edges.any():
                    # Get indices of edges connected to this protected node
                    node_edge_indices = torch.where(node_edges)[0]
                    
                    # If we have more edges than minimum required, randomly select some
                    if len(node_edge_indices) > min_edges_per_protected:
                        num_edges_to_keep = max(
                            min_edges_per_protected,
                            int(len(node_edge_indices) * (1 - ratio))
                        )
                        kept_edges = node_edge_indices[
                            torch.randperm(len(node_edge_indices))[:num_edges_to_keep]
                        ]
                    else:
                        # Keep all edges if we have fewer than minimum
                        kept_edges = node_edge_indices
                    
                    protected_edges_mask[kept_edges] = True
            
            # Combine random mask with protected edges mask
            mask = mask | protected_edges_mask
        
        # Update edge index
        new_edge_index = graph.edge_index[:, mask]
        
        # Create new Batch object
        new_data = Batch(
            edge_index=new_edge_index,
            batch=graph.batch,
            num_nodes=graph.num_nodes
        )
        
        # Preserve node features
        if hasattr(graph, 'x'):
            new_data.x = graph.x
        
        # Preserve edge attributes if they exist
        if hasattr(graph, 'edge_attr'):
            new_data.edge_attr = graph.edge_attr[mask]
        
        return new_data.to(device)

    @staticmethod
    def _node_dropping(graph, p_node_drop, device='cuda', protected_nodes=None):
        """
        Randomly removes a percentage of nodes while preserving specified nodes.
        
        Parameters:
            graph (Data): A PyTorch Geometric Data object.
            p_node_drop (float): Fraction of nodes to drop.
            protected_nodes (torch.Tensor): Indices of nodes that should not be dropped.
            device (str): Device to perform computations on (default: 'cuda').
        
        Returns:
            Data: A new PyTorch Geometric Data object with dropped nodes.
        """
        graph = graph.to(device)
        num_nodes = graph.num_nodes
        
        # Initialize node mask with random values
        node_mask = torch.rand(num_nodes, device=device) > p_node_drop
        
        if protected_nodes is not None:
            # Ensure protected_nodes is on the correct device
            protected_nodes = protected_nodes.to(device)
            # Set mask to True for protected nodes
            node_mask[protected_nodes] = True
        
        # Get the remaining nodes
        remaining_nodes = torch.where(node_mask)[0].to(torch.long)
        
        # Create mapping from old to new indices
        old_to_new_nodes = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
        old_to_new_nodes[remaining_nodes] = torch.arange(remaining_nodes.size(0), device=device)
        
        # Filter node features
        new_x = graph.x[node_mask] if hasattr(graph, 'x') else None
        
        # Filter edges where both nodes remain
        edge_mask = node_mask[graph.edge_index[0]] & node_mask[graph.edge_index[1]]
        new_edge_index = graph.edge_index[:, edge_mask]
        
        # Update edge indices
        new_edge_index[0] = old_to_new_nodes[new_edge_index[0]]
        new_edge_index[1] = old_to_new_nodes[new_edge_index[1]]
        
        # Filter edge attributes if they exist
        new_edge_attr = graph.edge_attr[edge_mask] if hasattr(graph, 'edge_attr') else None
        
        # Create new Data object
        new_data = Batch(
            edge_index=new_edge_index,
            num_nodes=remaining_nodes.size(0)
        )
        
        if new_x is not None:
            new_data.x = new_x
        if new_edge_attr is not None:
            new_data.edge_attr = new_edge_attr
        
        return new_data.to(device)

    @staticmethod
    def corruption_shuffle_nodes(x, edge_index, edge_attr):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        perm = torch.randperm(x.size(0), device=device)
        return x[perm], edge_index, edge_attr
    
    @staticmethod
    def corruption_shuffle_edges(x, edge_index):
        num_edges = edge_index.size(1)
        perm = torch.randperm(num_edges)
        return x, edge_index[:, perm]
    
    def generate_augmentations(self, data, protected_nodes):
        views = []
        batch = data.clone().to(self.device)

        for i in range(self.n_views):
            if i < self.n_global_views:
                # for the global views remove randomly only 10% or leave full data.
                # for global views the augmentation is always subraph removal
                # if round(random.random()):
                #     # views.append(batch)
                #     views.append(self._edge_dropping(batch, 0.15, self.device))
                #     # views.append(self._mask_edge_features(batch, 0.15, self.device))  
                #     # views.append(self._subgraph_removal(batch, 0.1, self.device))              
                # else:
                #     # views.append(batch)
                #     # views.append(self._mask_node_features(batch, 0.15, self.device))
                #     views.append(self._subgraph_removal(batch, 0.15, self.device))              
                #     # views.append(self._edge_dropping(batch, 0.15, self.device))  
                #     # views.append(self._mask_node_features(batch, 0.15, self.device))
                g_aug = [
                    lambda input: input, # full non distorted input view
                    lambda input: self._subgraph_removal(input, 0.15, self.device, protected_nodes=protected_nodes),
                    lambda input: self._edge_dropping(input, 0.15, self.device, protected_nodes=protected_nodes),
                    lambda input: self._node_dropping(input, 0.15, self.device, protected_nodes=protected_nodes),
                    # lambda input: self._mask_node_features(input, 0.15, self.device),
                    # lambda input: self._mask_edge_features(input, 0.15, self.device),
                ]
                selected_augmentations = random.sample(g_aug, 1)
                data_aug = batch
                for aug in selected_augmentations:
                    data_aug = aug(data_aug)
                views.append(data_aug)
            else:
                # random local augmentations, with higher perturbation ratios.
                # in this case multiple augmentation rounds are performed
                augmentations1 = [
                    lambda input: self._subgraph_removal(input, 0.25, self.device, protected_nodes=protected_nodes),
                    lambda input: self._edge_dropping(input, 0.25, self.device, protected_nodes=protected_nodes),
                    lambda input: self._node_dropping(input, 0.25, self.device, protected_nodes=protected_nodes),
                ]
                augmentations2 = [
                    lambda input: self._mask_node_features(input, 0.15, self.device),
                    lambda input: self._mask_edge_features(input, 0.15, self.device),
                ]
                num_augmentations = 1 # random.randint(1,3) # randomly select up to 3 augmentations
                selected_augmentations = random.sample(augmentations1, num_augmentations)
                data_aug = batch # copy to leave batch unmodified
                for aug in selected_augmentations:
                    data_aug = aug(data_aug)
                selected_augmentations = random.sample(augmentations2, num_augmentations)
                for aug in selected_augmentations:
                    data_aug = aug(data_aug)
                views.append(data_aug)
        return views
        
    @staticmethod
    def get_common_nodes(batch1, batch2, device="cuda"):
        """
        Finds common nodes between two PyG batches using edge_index (GPU-efficient).

        Parameters:
            batch1 (Data): First PyG batch.
            batch2 (Data): Second PyG batch.
            device (str): Device for computation.

        Returns:
            common_nodes (Tensor): Node indices that exist in both batches.
        """
        batch1, batch2 = batch1.to(device), batch2.to(device)

        # Get unique node indices present in each batch
        nodes1 = torch.unique(batch1.edge_index)  # Nodes in batch1
        nodes2 = torch.unique(batch2.edge_index)  # Nodes in batch2

        # Find intersection using torch.isin
        common_mask1 = torch.isin(nodes1, nodes2)  # Mask of common nodes in batch1
        common_mask2 = torch.isin(nodes2, nodes1)  # Mask of common nodes in batch2

        common_nodes = nodes1[common_mask1]  # The actual node indices in the original graph

        # Map common nodes back to their positions in batch.x
        idx_t = torch.nonzero(common_mask1, as_tuple=True)[0]  # Positions in batch1.x
        idx_s = torch.nonzero(common_mask2, as_tuple=True)[0]  # Positions in batch2.x

        return idx_t, idx_s