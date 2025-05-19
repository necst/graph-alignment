import random
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader, NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.nn import VGAE, DeepGraphInfomax
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.data import Data

from src.data.remove_edges import remove_edges
from src.data.preprocessing.supervised.handle_supervised_data import concatenate_negatives, split_negative_edges
from src.models.nn.link_predictor import LinkPredictor
from src.models.supervised_model import SupervisedLinkModel

from ..models.nn.encoder import Encoder
from ..models.dino import DINO
from ..models.contrastive import GraphCL
from ..training.augmenter import Augmenter 
from ..training.loss import DINOLoss
from ..data.load_ogbl_biokg_homo import load_ogbl_biokg_homo
from .model_loader import ModelLoader
from ..training.loss import NTXentLoss
from torch_geometric.utils import subgraph

#TODO implement dataset

def create_shared_loader(data, n_samples, n_neighbors, batch_size, seed=42):
    interval = torch.load("src/data/dataset/ogbl_biokg/load_data/interval.pt")
    torch.manual_seed(seed)
    class_counts = {cls: end - start for cls, (start, end) in interval.items()}
    total = sum(class_counts.values())
    node_indices = []

    # Calcola quanti sample per classe
    samples_per_class = {cls: int((count / total) * n_samples) for cls, count in class_counts.items()}

    for cls, (start, end) in interval.items():
        available = torch.arange(start, end)
        sampled = available[torch.randperm(len(available))[:samples_per_class[cls]]]
        node_indices.append(sampled)

    # Concatena tutti i sample
    node_indices = torch.cat(node_indices)  # sample shared nodes
    node_indices = node_indices[torch.randperm(len(node_indices))]

    loader = NeighborLoader(
        data,
        num_neighbors=n_neighbors,
        batch_size=batch_size,
        input_nodes=node_indices,
        shuffle=False  # don't reshuffle!
    )
    return loader, node_indices


class ConfigParser():
    def get_training_setup(self, config:dict):
        assert isinstance(config, dict),  f"Expected a dict, got a {type(config).__name__}"
        self.config = config
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data = self._load_dataset(self, self.config['dataset'], processed=True)
        if self.config['dataset'] == 'Proteins':
            # Costruisci il sottografo: edge solo tra nodi di train
            edge_index_sub, edge_attr_sub = subgraph(
                subset=data.train_idx,               # i nodi che vuoi tenere
                edge_index=data.edge_index,
                edge_attr= data.edge_attr,
                # il grafo originale
                relabel_nodes=True,     
                # rilabel dei nodi per creare un grafo "compatto"
            )
            
            x_sub = data.x[data.train_idx]
            y_sub = data.y[data.train_idx]

            old_to_new = {old.item(): new for new, old in enumerate(data.train_idx)}

            data = Data(
                x=x_sub,
                edge_index=edge_index_sub,
                edge_attr=edge_attr_sub,
                y=y_sub
            )
        else:    
            interval = torch.load(f'src/data/dataset/ogbl_biokg/load_data/interval.pt')

        if self.config['model'] == 'VGAE':
            data.train_mask = data.val_mask = data.test_mask = None
            if (self.config['dataset'] == 'BioKG'):
                transform = T.RandomLinkSplit(num_val=0.1, num_test=0.1, is_undirected=True, add_negative_train_samples=False, neg_sampling_ratio = 1.0, split_labels = True)
            else:
                transform = T.RandomLinkSplit(num_val=0.0, num_test=0.0, is_undirected=True, add_negative_train_samples=False, neg_sampling_ratio = 1.0, split_labels = True)
            train_data, val_data, test_data = transform(data)
        # compute number of node and edge features and the max value of each feature 
        # to create learnable embeddings
        self.config.update({'embedding':{}})
        self.config['embedding']['node_features'] = data.x.size(dim = 1)
        self.config['embedding']['edge_features'] = data.edge_attr.size(dim = 1)
        self.config['embedding']['max_node_features'], _ = data.x.max(dim=0)
        self.config['embedding']['max_edge_features'], _ = data.edge_attr.max(dim=0)
        if self.config['model'] == 'VGAE': 
            self.config.update({'data': (data, train_data, val_data, test_data)})
            train_edges_set = set(map(tuple, train_data.pos_edge_label_index.T.tolist()))
            val_edges_set = set(map(tuple, val_data.pos_edge_label_index.T.tolist()))
            test_edges_set = set(map(tuple, test_data.pos_edge_label_index.T.tolist()))
            self.config.update({'edges_sets': {}})
            self.config['edges_sets']['train'] = train_edges_set
            self.config['edges_sets']['valid'] = val_edges_set
            self.config['edges_sets']['test'] = test_edges_set
        if self.config['model'] == 'Supervised':
            drug_disease = self.config['drug_disease']
            drug_side_effect = self.config['drug_side_effect']
            disease_protein = self.config['disease_protein']
            function_function = self.config['function_function']
            if self.config['dataset'] == 'BioKG':
                _, whole_edge_index_dict, edge_attr_dict = remove_edges(data, interval, drug_disease=drug_disease, drug_side_effect=drug_side_effect, disease_protein=disease_protein, function_function=function_function, p = 1)
                neg_edges, neg_attrs = self.generate_negative_samples(data, whole_edge_index_dict, interval, edge_attr_dict)
                train_data, edge_index_dict, edge_attr_dict = remove_edges(data, interval, drug_disease=drug_disease, drug_side_effect=drug_side_effect, disease_protein=disease_protein, function_function=function_function, p = 0.2)
                test_edge_index = list(edge_index_dict.values())[0]
                test_edge_attr = list(edge_attr_dict.values())[0]
                train_neg_edges, train_neg_attrs, test_neg_edges, test_neg_attrs = split_negative_edges(neg_edges, neg_attrs)
                train_data  = concatenate_negatives(train_data, train_neg_edges, train_neg_attrs, list(whole_edge_index_dict.values())[0])
                self.train_data = train_data
                test_data = Data(x=train_data.x, edge_index=test_edge_index, edge_attr=test_edge_attr)
                test_data = concatenate_negatives(test_data, test_neg_edges, test_neg_attrs, list(edge_index_dict.values())[0])
                self.test_data = test_data
            else:
                raise RuntimeError("supervised for proteins still to be implemented")
                    
        model = self._get_model(self.config) 
        optimizer = self._get_optimizer(self.config, model)
        criterion = self._get_loss(self.config, device)
        loader = self._get_loader(self, self.config, data if (self.config['model'] != "VGAE" and self.config['model'] != 'Supervised') else train_data)
        return model, optimizer, criterion, loader, device, self.config
    

    def load_run(self, run:str, checkpoint:str, n_samples=5000, return_layers=False, test=False, shuffle_samples=True, shared_loader=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_loader = ModelLoader(run=run, checkpoint=checkpoint, device=device)
        config = model_loader.config        
        model = model_loader.get_encoder()
        data = self._load_dataset(self, config['dataset'], processed=True)
        train_data = data
        if (config['model'] == 'Supervised' and test==True):
            interval = torch.load("src/data/dataset/ogbl_biokg/load_data/interval.pt")
            _, whole_edge_index_dict, edge_attr_dict = remove_edges(data, interval, drug_disease=config['drug_disease'], drug_side_effect=config['drug_side_effect'], disease_protein=config['disease_protein'], function_function=config['function_function'], p = 1)
            neg_edges, neg_attrs = self.generate_negative_samples(data, whole_edge_index_dict, interval, edge_attr_dict)
            train_data, edge_index_dict, edge_attr_dict = remove_edges(data, interval, drug_disease=config['drug_disease'], drug_side_effect=config['drug_side_effect'], disease_protein=config['disease_protein'], function_function=config['function_function'], p = 0.2)
            test_edge_index = list(edge_index_dict.values())[0]
            test_edge_attr = list(edge_attr_dict.values())[0]
            _, _, test_neg_edges, test_neg_attrs = split_negative_edges(neg_edges, neg_attrs)
            self.train_data = train_data
            test_data = Data(x=train_data.x, edge_index=test_edge_index, edge_attr=test_edge_attr)
            test_data = concatenate_negatives(test_data, test_neg_edges, test_neg_attrs, list(edge_index_dict.values())[0])
            self.test_data = test_data              
            
        # seed for the shuffle in neighbor loader
        # need shuffle beacuse witohut it there will only be the first class if with few samples
        seed = 42
        torch.manual_seed(seed)
        if shared_loader:
            data_loader = shared_loader
        else:
            data_loader = NeighborLoader(data if (config['model'] != 'Supervised' and test == True) else train_data, num_neighbors=config['loader']['n_neighbors'], batch_size=config['loader']['batch_size'], shuffle = shuffle_samples)
        samples, labels = self._encode_samples(model, data_loader, device, n_samples, config['model'], config['loader']['batch_size'] if not shared_loader else 512, return_layers, config['dataset'])
        if return_layers:
            if config['model'] == 'Supervised':
                self.intermediate_layers = model.encoder.layers_emb
            else:
                self.intermediate_layers = model.layers_emb 

        print(f'Model: {config["model"]}')
        print(f'Samples: {samples.shape}')
        return samples, labels

    def generate_negative_samples (self, data, edge_index_dict, interval, edge_attr_dict):
        edge_index_set = set(map(tuple, data.edge_index.T.tolist()))
        edge_tuples = {}
        for key, value in edge_index_dict.items():
            edge_tuples[key] = set(map(tuple, value.T.tolist()))
        negative_edges = {}
        negative_edges_attr = {}
        for key, value in edge_tuples.items():
            num_negatives = len(value)  # Tanti quanti gli archi positivi
        # num_negatives = drug_disease_set.size(1)  # Tanti quanti gli archi positivi
            negative_edges[key] = []
            negative_edges_attr[key] = []
            if key in edge_attr_dict:
                attr_dim = edge_attr_dict[key].shape[1]  # Prende la dimensione della feature
            else:
                attr_dim = 1
            while len(negative_edges[key]) < num_negatives:
                u, v = random.randint(interval[key.split('_')[0]][0], interval[key.split('_')[0]][1]), random.randint(interval[key.split('_')[1]][0], interval[key.split('_')[1]][1])
                if (u, v) not in edge_index_set and not any((u, v) in t for t in edge_tuples.values()):  # Assicurati che non sia un arco esistente
                    negative_edges[key].append((u, v))
                    negative_edges_attr[key].append(torch.zeros(attr_dim))
        return negative_edges, negative_edges_attr    
    
    @staticmethod
    def _encode_samples(encoder, loader, device, n_samples, model_type, batch_size, return_layers, dataset='BioKG'):
        z = torch.Tensor().to(device) 
        labels=[]
        with torch.no_grad():
            for batch in loader:
                # with autocast(device, dtype=torch.bfloat16):
                if model_type == 'DINO':
                    z_batch = encoder(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device), return_layers=return_layers)
                elif model_type == 'GraphCL':
                    z_batch = encoder(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device), return_layers=return_layers)
                elif model_type == 'VGAE':
                    z_batch = encoder.encode(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device), return_layers=return_layers)
                elif model_type == 'DeepGraphInfomax':
                    z_batch = encoder(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device), return_layers=return_layers)
                elif model_type == 'Supervised':
                    z_batch = encoder.encoder(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device), return_layers=return_layers)
                z = torch.cat((z,z_batch[:batch_size,:]), dim=0)  
                if dataset == 'BioKG':
                    labels.extend(batch.x[:batch_size,:].squeeze().to(torch.int).tolist())  # for biokg
                else:
                    labels.extend(batch.y[:batch_size,:].squeeze().to(torch.int).tolist())  # for ogbn-proteins

                # check requested number of samples 
                if n_samples != None: 
                    if len(z) >= n_samples:
                        z = z[:n_samples]
                        break

        z = z.cpu()
        labels = np.array(labels)
        return z, labels

    @staticmethod
    def _get_model(config:dict) -> torch.nn.Module:
        encoder = Encoder(model=config['model'],
                                           layer=config['encoder']['layer'],
                                           dim=config['encoder']['dim'],
                                           num_layers=config['encoder']['n_layers'],
                                           norm_layer=config['encoder']['norm_layer'],
                                           act_layer=config['encoder']['act_layer'],
                                           node_feature_dim=config['encoder']['num_classes'],
                                           input_dim=config['embedding']['node_features'],
                                           output_dim=config['encoder']['output_dim'],
                                           use_embedding=config['encoder']['use_embedding'],
                                           use_pre_norm=config['encoder']['use_pre_norm'],
                                           residual_connections=config['encoder']['residual_connections'],
                                           scale_factor=config['encoder']['scale_factor']
                                           )
        if config['model'] == 'DINO':
            model = DINO(config)
        elif config['model'] == 'GraphCL':
            model = GraphCL(config)
        elif config['model'] == 'VGAE':
            if config['encoder']['layer'] != 'GIN' and config['encoder']['layer'] != 'Transformer':
                model = VGAE(encoder = encoder)
            else:
                model = VGAE(encoder=Encoder(layer=config['encoder']['layer'], dim=config['encoder']['dim'], num_layers=config['encoder']['n_layers'], num_heads=config['encoder']['n_heads'], 
                               mlp_ratio=config['encoder']['mlp_ratio'], drop_rate=config['encoder']['drop_rate'], attn_drop_rate=config['encoder']['attn_drop_rate'], 
                               embedding = config['embedding'], norm_layer=config['encoder']['norm_layer'], act_layer=config['encoder']['act_layer'], dataset=config['dataset']))
        elif config['model'] == "DeepGraphInfomax":
            model = DeepGraphInfomax(encoder=encoder,
                                    hidden_channels=config['encoder']['dim'],
                                    summary=lambda z, *args, **kwargs: z.mean(dim=0).sigmoid(), #Calcola la media di tutti gli embedding dei nodi lungo la dimensione dei nodi (dim=0), ottenendo una rappresentazione globale del grafo.
                                    corruption=Augmenter.corruption_shuffle_nodes)
        elif config['model'] == 'Supervised':
            predictor = LinkPredictor(
                in_channels= config['encoder']['dim'],
                hidden_channels=config['encoder']['dim'],
                num_layers=3,
                act_layer=config['encoder']['act_layer'],
                norm_layer=config['encoder']['norm_layer'],
                drop=0.3        
            )
            model = SupervisedLinkModel(encoder=encoder, predictor=predictor)
        
        assert isinstance(model, torch.nn.Module),  f"Returning {type(model).__name__} instead of torch.nn.Module"
        return model
    
    @staticmethod
    def _load_dataset(self, dataset, type_feature='class', processed=False):
        if dataset == 'BioKG':
            data = load_ogbl_biokg_homo(version='duplicates', type_feature=type_feature, processed=processed)
        elif dataset == 'Proteins':
            dataset = PygNodePropPredDataset(name = 'ogbn-proteins')
            split_idx = dataset.get_idx_split()
            train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
            data = dataset[0]
            #data.x = torch.ones((data.num_nodes, 1), dtype=torch.int64)
            # Supponiamo che i tuoi ID siano in un tensore data.x di shape [N, 1] o [N]
            original_ids = data.node_species.squeeze()  # [N] vettore di ID tipo 3702, 10000, ecc.

            # Trova i valori unici e crea una mappatura
            unique_ids, inverse_indices = torch.unique(original_ids, sorted=True, return_inverse=True)

            # Rimappa gli ID a interi da 0 in su
            data.x = inverse_indices.unsqueeze(1)  # shape [N, 1] se ti serve così
            data.train_idx = train_idx
            data.valid_idx = valid_idx
            data.test_idx = test_idx
        elif dataset == 'ArXiv':
            dataset = PygNodePropPredDataset(name = 'ogbn-arxiv') 
            data = dataset[0]
            data.edge_attr = torch.ones((data.edge_index.shape[1], 1), dtype=torch.int64)
        return data   

    @staticmethod
    def _get_optimizer(config, model):
        if config['training']['optim'] == 'AdamW':
            optimizer = optim.AdamW(model.student.parameters() if config["model"] == "DINO" else model.parameters(), lr=float(config['training']['lr']), weight_decay=float(config['training']['init_wd']))
        return optimizer

    @staticmethod
    def _get_loss(config, device) -> torch.nn.Module:
        if config['model'] == 'DINO':
            loss = DINOLoss(config, device)
        elif config['model'] == 'GraphCL':
            loss = NTXentLoss
        elif config['model'] == 'VGAE' or config['model'] == 'DeepGraphInfomax' or config['model'] == 'Supervised':
            return None
        return loss
    
    @staticmethod
    def _get_loader(self, config, data) -> DataLoader:
        loader = NeighborLoader(data, num_neighbors=config['loader']['n_neighbors'], batch_size=config['loader']['batch_size'], shuffle = True)
        return loader
    
    
            