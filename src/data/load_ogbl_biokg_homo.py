from ogb.linkproppred import PygLinkPropPredDataset
from torch_geometric.data import Data
from .remove_edges import remove_edges
import torch
import os

#version could be 'unique' or 'duplicates'
#type_feature could be 'one_hot' or 'class'
#unique along with class is not possible

def load_ogbl_biokg_homo(version, type_feature, processed):
    versions = ["unique", "duplicates"]
    type_features = ["one_hot", "class"]
    if version not in versions or type_feature not in type_feature:
        raise ValueError(f"version or type_feature not available")
    if version == 'unique' and type_feature == 'class':
        raise ValueError(f"Unique-class is not a possible couple, please change version or type_feature")
    folder_path = f'src/data/dataset/ogbl_biokg/load_data'
    if os.path.exists(os.path.join(folder_path, f'ogbl_{version}_{type_feature}{"_processed" if processed else ""}.pt')):
        data = torch.load(os.path.join(folder_path, f'ogbl_{version}_{type_feature}{"_processed" if processed else ""}.pt'))
        return data
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    dataset = PygLinkPropPredDataset("ogbl-biokg",  root = 'src/data/dataset/')
    data = dataset[0]
    node_types = list(data.num_nodes_dict.keys())
    num_node_types = len(node_types)
    node_type_to_idx = {node_type: i for i, node_type in enumerate(node_types)}
    class_features = []
    if type_feature == 'one_hot':
        for node_type in node_types:
            num_nodes = data.num_nodes_dict[node_type]
            #torch.zeros((num_righe,num_colonne)) crea una matrice delle dimensioni indicate con tutti 0
            features = torch.zeros((num_nodes, num_node_types))
            #per tutte le righe, nella colonna specificata metti 1
            features[:, node_type_to_idx[node_type]] = 1
            #aggiungi matrice alla lista
            class_features.append(features)
    else:
        for node_type in node_types:
            num_nodes = data.num_nodes_dict[node_type]
            features = torch.zeros((num_nodes, 1))
            features[:,0] = node_type_to_idx[node_type]
            class_features.append(features)
    data.x = torch.cat(class_features, dim=0)
    edge_types = list(data.edge_index_dict.keys())
    num_edge_types = len(edge_types)
    edge_type_to_idx = {edge_type: i for i, edge_type in enumerate(edge_types)}
    node_offset = {}
    current_offset = 0
    for node_type in node_types:
        node_offset[node_type] = current_offset
        current_offset += data.num_nodes_dict[node_type]
    
    interval = {}
    for node_type in node_types:
        interval[node_type] = (node_offset[node_type], node_offset[node_type] + data.num_nodes_dict[node_type] - 1)
        
    torch.save (interval, 'src/data/dataset/ogbl_biokg/load_data/interval.pt')
        
    new_edge_index_list = []
    edge_features_list = []

    for edge_type in edge_types:
        src_offset = node_offset[edge_type[0]]
        dst_offset = node_offset[edge_type[2]]
        new_edge_index_list.append(data.edge_index_dict[edge_type] + torch.tensor([[src_offset], [dst_offset]]))
        if type_feature == 'one_hot':
            edge_features = torch.zeros((data.edge_index_dict[edge_type].shape[1], num_edge_types))
            edge_features[:, edge_type_to_idx[edge_type]] = 1
        else:
            edge_features = torch.zeros((data.edge_index_dict[edge_type].shape[1], 1))
            edge_features[:, 0] = edge_type_to_idx[edge_type]
        edge_features_list.append(edge_features)

    homogeneous_edge_index = torch.cat(new_edge_index_list, dim=1)
    edge_features = torch.cat(edge_features_list, dim=0)
    if version == "duplicates":
        data = Data(x=data.x, edge_index=homogeneous_edge_index, edge_attr=edge_features)
        if processed:
            data, subsets, subsets_attr = remove_edges(data, interval, drug_disease=True, drug_side_effect=True, disease_protein=True, function_function=True)
            torch.save(subsets, f'src/data/dataset/ogbl_biokg/load_data/ogbl_{version}_{type_feature}_subsets.pt')
            torch.save(subsets_attr, f'src/data/dataset/ogbl_biokg/load_data/ogbl_{version}_{type_feature}_subsets_attr.pt')
        torch.save(data, f'src/data/dataset/ogbl_biokg/load_data/ogbl_{version}_{type_feature}{"_processed" if processed else ""}.pt')
        return data

    unique_edges, inverse_indices = torch.unique(homogeneous_edge_index, dim=1, return_inverse=True)
    aggregated_edge_features = torch.zeros(unique_edges.shape[1], num_edge_types)
    

    for i, index in enumerate(inverse_indices):
        aggregated_edge_features[index] += edge_features[i]
        
    data = Data(x=data.x, edge_index=unique_edges, edge_attr=aggregated_edge_features)
    if processed:
        data, subsets, subsets_attr = remove_edges(data, interval, drug_disease=True, drug_side_effect=True, disease_protein=True, function_function=True)
        torch.save(subsets, f'src/data/dataset/ogbl_biokg/load_data/ogbl_{version}_{type_feature}_subsets.pt')
        torch.save(subsets_attr, f'src/data/dataset/ogbl_biokg/load_data/ogbl_{version}_{type_feature}_subsets_attr.pt')
    torch.save (data, f'src/data/dataset/ogbl_biokg/load_data/ogbl_{version}_{type_feature}{"_processed" if processed else ""}.pt')
    return data

# torch.save (data_unique, 'dataset/ogbl_biokg/load_data/ogbl_unique.pt')
# torch.save (data_with_duplicates, 'dataset/ogbl_biokg/load_data/ogbl_duplicates.pt')
# torch.save (interval, 'dataset/ogbl_biokg/load_data/interval.pt')

    