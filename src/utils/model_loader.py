import yaml
import os
from pathlib import Path

import torch
from torch_geometric.nn import DeepGraphInfomax, VGAE

from src.models.nn.link_predictor import LinkPredictor
from src.models.supervised_model import SupervisedLinkModel

from ..models.dino import DINO
from ..models.contrastive import GraphCL
from ..models.nn.encoder import Encoder


class ModelLoader():
    def __init__(self, run:str='', checkpoint:str='', device='cuda') -> None:
        self.run = run
        self.checkpoint = checkpoint
        self.device = device
        self.model = self._load_model(self.run, self.checkpoint)
        
    
    def get_encoder(self) -> torch.nn.Module:
        if self.config['model'] == 'DINO':
            return self.model.teacher.encoder.to(self.device)
        elif self.config['model'] == 'GraphCL':
            return self.model.branch1.encoder.to(self.device)
        elif self.config['model'] == 'DeepGraphInfomax':
            return self.model.encoder.to(self.device)    
        elif self.config['model'] == 'VGAE':
            return self.model.to(self.device)
        elif self.config['model'] == 'Supervised':
            return self.model.to(self.device)

    def _load_model(self, run:str, checkpoint:str) -> torch.nn.Module:
        root = Path(os.getcwd())
        with open(root.joinpath('runs',run, 'config.yaml'), 'r') as f:
            config = yaml.safe_load(f)     
        print(config['model'])
        config.update({'embedding':{}})
        config['embedding']['node_features'] = 1
        config['embedding']['edge_features'] = 1
        config['embedding']['max_node_features'] = 6
        config['embedding']['max_edge_features'] = 52

        state_dict=torch.load(root.joinpath('runs', run, f'{checkpoint}.pt'), map_location=torch.device('cpu'))
        #to not repeat the code
        encoder_mock = Encoder(model=config['model'],
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
                                    scale_factor=config['encoder']['scale_factor'],
                                    dataset=config['dataset'],
                                    batch_size=config['loader']['batch_size']
                                    )
        encoder_gin = Encoder(model=config['model'], layer=config['encoder']['layer'], dim=config['encoder']['dim'],
                num_layers=config['encoder']['n_layers'], num_heads=config['encoder']['n_heads'],
                mlp_ratio=config['encoder']['mlp_ratio'], drop_rate=config['encoder']['drop_rate'],
                attn_drop_rate=config['encoder']['attn_drop_rate'],
                embedding=config['embedding'], norm_layer=config['encoder']['norm_layer'],
                act_layer=config['encoder']['act_layer'], dataset=config['dataset'])

        if config['model'] == 'byocc':
            # TODO
            # model = BYOCC(config)
            pass
        if config['model'] == 'DINO':
            model = DINO(config)
        if config['model'] == 'GraphCL':
            model = GraphCL(config)
        if config['model'] == 'VGAE':
            encoder = encoder_mock
            if config['encoder']['layer'] != 'GIN' and config['encoder']['layer'] != 'Transformer':
                model = VGAE(encoder = encoder)
            else:
                model = VGAE(encoder=encoder_gin)
        if config['model'] == 'DeepGraphInfomax':
            encoder = encoder_mock
            model = DeepGraphInfomax(encoder=encoder,
                        hidden_channels=config['encoder']['dim'],
                        summary=lambda z, *args, **kwargs: z.mean(dim=0).sigmoid(), #Calcola la media di tutti gli embedding dei nodi lungo la dimensione dei nodi (dim=0), ottenendo una rappresentazione globale del grafo.
                        corruption=None)
        if config['model'] == 'Supervised':
            predictor = LinkPredictor(
                in_channels= config['encoder']['dim'],
                hidden_channels=config['encoder']['dim'],
                num_layers=3,
                act_layer=config['encoder']['act_layer'],
                norm_layer=config['encoder']['norm_layer'],
                drop=0.3        
            )
            model = SupervisedLinkModel(encoder=encoder_mock if config['encoder']['layer'] != 'GIN' and config['encoder']['layer'] != 'Transformer' else encoder_gin, predictor=predictor)
        self.config = config
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

