import torch
from torch_geometric.data import Data
def split_negative_edges(edge_index_dict, edge_attr_dict):
    edge_index = torch.tensor(list(edge_index_dict.values())[0]).T
    edge_attr = torch.tensor(list(edge_attr_dict.values())[0])
    num_edges = edge_index.shape[1]
    perm = torch.randperm(num_edges)
    
    split_idx = int(0.8 * num_edges)

    combined = list(zip(edge_index, edge_attr))

    train_idx = perm[:split_idx]
    test_idx = perm[split_idx:]

    train_neg_edges = edge_index[:, train_idx]
    train_neg_attrs = edge_attr[train_idx]

    test_neg_edges = edge_index[:, test_idx]
    test_neg_attrs = edge_attr[test_idx]
    
    return train_neg_edges, train_neg_attrs, test_neg_edges, test_neg_attrs

def concatenate_negatives(data, neg_edges, neg_attrs, exclude_edge_index=None):
    pos_edge_index = data.edge_index         # shape [2, N_pos]
    pos_edge_attr = data.edge_attr           # shape [N_pos, D]
    y_pos = torch.ones(pos_edge_index.size(1))
    y_neg = torch.zeros(neg_edges.size(1))
 # Inizializza la maschera con True per tutti gli archi positivi
    exclude_mask = torch.zeros(pos_edge_index.size(1), dtype=torch.bool).to(data.x.device)

    # Confronta ogni arco positivo con gli archi da escludere
    # for i in range(pos_edge_index.size(1)):
    #     src, dst = pos_edge_index[:, i]  # Nodo di origine e nodo di destinazione per ogni arco positivo
        
    #     # Verifica se (src, dst) è presente in exclude_edge_index
    #     exclude_mask[i] = not(((exclude_edge_index[0] == src) & (exclude_edge_index[1] == dst)).any())
    exclude_mask = filter_edges(pos_edge_index, exclude_edge_index)

    # Imposta l'etichetta a -1 per gli archi che devono essere esclusi
    y_pos[exclude_mask] = -1
    edge_index = torch.cat([pos_edge_index, neg_edges], dim=1)
    edge_attr = torch.cat([pos_edge_attr,neg_attrs.unsqueeze(-1)], dim=0)
    y = torch.cat([y_pos, y_neg])
    new_data = Data(x=data.x, edge_index=edge_index, edge_attr=edge_attr, edge_label=y)
    return new_data

    
def generate_edge_labels(data, neg_edge_index, idx):
    if idx >= neg_edge_index.size(1):  # Se idx supera il numero di archi negativi disponibili, ricomincia da capo
        idx = 0
    edge_index_pos = data.edge_index  # Archi positivi
    num_pos = edge_index_pos[:, data.edge_label == 1].size(1)
    edge_index_neg = neg_edge_index[:,idx:num_pos]
    edge_label_index = torch.cat([edge_index_pos, edge_index_neg], dim=1)
    edge_label = torch.cat([data.edge_label, torch.zeros(num_pos)])
    

    return edge_label_index, edge_label 

def filter_edges(pos_edge_index, exclude_edge_index):
    # Crea una rappresentazione efficiente degli archi da escludere
    exclude_edges = set(zip(exclude_edge_index[0].tolist(), exclude_edge_index[1].tolist()))
    
    # Estrai le sorgenti e destinazioni
    src = pos_edge_index[0]
    dst = pos_edge_index[1]
    
    # Crea una maschera booleana usando operazioni vettoriali
    edges = list(zip(src.tolist(), dst.tolist()))
    exclude_mask = torch.tensor([edge not in exclude_edges for edge in edges], dtype=torch.bool)
    
    return exclude_mask
    
    
    