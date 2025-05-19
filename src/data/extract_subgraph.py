import yaml

from src.training.trainer import Trainer
from src.data.load_ogbl_biokg_homo import load_ogbl_biokg_homo
from src.utils.model_loader import ModelLoader
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.loader import NeighborLoader
from torch.cuda.amp import autocast
import torch


def extract_subgraph(data, node_type_src, node_type_dst):
    if node_type_src is None or node_type_dst is None:
        return data
    class_to_idx = {
    "disease": 0,
    "drug": 1,
    "function": 2,
    "protein": 3,
    "sideeffect" : 4
    }
    node_type_src = class_to_idx[node_type_src]
    node_type_dst = class_to_idx[node_type_dst]
    node_classes = data.x.squeeze()
    src_nodes = (node_classes == node_type_src).nonzero(as_tuple=True)[0]
    dst_nodes = (node_classes == node_type_dst).nonzero(as_tuple=True)[0]  
    all_nodes = torch.cat([src_nodes, dst_nodes]).unique()
    mapping = {old.item(): new for new, old in enumerate(all_nodes)}
    
        # 3. Crea maschera basata sui nodi (non sull'edge type)
    src_index = data.edge_index[0]
    dst_index = data.edge_index[1]

    # Nuova maschera: archi che partono da src e arrivano a dst
    mask = torch.isin(src_index, src_nodes) & torch.isin(dst_index, dst_nodes)
     
    filtered_edge_index = data.edge_index[:, mask]
    
    src, dst = filtered_edge_index[0], filtered_edge_index[1]
    valid_src = torch.isin(src, src_nodes)
    valid_dst = torch.isin(dst, dst_nodes)
    final_mask = valid_src & valid_dst
    src_dst_edges = filtered_edge_index[:, final_mask]

    # Rimap edge_index
    remapped_edge_index = torch.stack([
        torch.tensor([mapping[s.item()] for s in src_dst_edges[0]]),
        torch.tensor([mapping[d.item()] for d in src_dst_edges[1]])
    ], dim=0)

    # Rimappa le feature dei nodi
    sub_x = data.x[all_nodes]
    
    from torch_geometric.data import Data

    sub_data = Data(
        x=sub_x,
        edge_index=remapped_edge_index,
        edge_attr=torch.ones(remapped_edge_index.shape[1], 1) * 1,
        num_nodes=sub_x.size(0)# oppure salva i veri tipi se ne hai
    )


    return sub_data, mapping
