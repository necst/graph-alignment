from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
import torch

def remove_edges (data, interval, drug_disease = True, drug_side_effect = True, disease_protein = True, function_function = True, p = 0.2):

    node_types = ['disease', 'drug', 'function', 'protein', 'sideeffect']
    masks = {}

    for node_type in node_types:
        masks[node_type + '_source'] = (data.edge_index[0] >= interval[node_type][0]) & (data.edge_index[0] <= interval[node_type][1])
        masks[node_type + '_dest'] = (data.edge_index[1] >= interval[node_type][0]) & (data.edge_index[1] <= interval[node_type][1])

    link_masks = {}
    if drug_disease:
        link_masks['drug_disease'] = masks["drug_source"] & masks["disease_dest"]
    if drug_side_effect:
        link_masks['drug_sideeffect'] = masks["drug_source"] & masks["sideeffect_dest"]
    if disease_protein:
        link_masks['disease_protein'] = masks["disease_source"] & masks["protein_dest"]
    if function_function:
        link_masks['function_function'] = masks["function_source"] & masks["function_dest"]

    subsets = {}
    subsets_attr = {}
    edge_index_mask = torch.ones(data.edge_index.size(1), dtype=torch.bool)
    for key,mask in link_masks.items():
        edges = torch.nonzero(mask, as_tuple=True)[0]
        num_edge_to_remove = int(p * edges.size(0))
        random_indices = torch.randperm(edges.size(0))[:num_edge_to_remove]
        edge_to_remove = edges[random_indices]
        subsets[key] = data.edge_index[:, edge_to_remove]
        subsets_attr[key] = data.edge_attr[edge_to_remove, :]
        edge_index_mask[edge_to_remove] = False
        
    new_data = Data(x=data.x, edge_index=data.edge_index[:, edge_index_mask], edge_attr=data.edge_attr[edge_index_mask, :])
    return new_data, subsets, subsets_attr
