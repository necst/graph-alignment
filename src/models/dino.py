import torch
import torch.nn as nn

from .nn.encoder import Encoder
from .nn.head import Head
from ..utils.scheduler import cosine_increase_law


class Network(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.encoder = Encoder(model=config['model'], layer=config['encoder']['layer'], dim=config['encoder']['dim'], num_layers=config['encoder']['n_layers'], num_heads=config['encoder']['n_heads'], 
                               mlp_ratio=config['encoder']['mlp_ratio'], drop_rate=config['encoder']['drop_rate'], attn_drop_rate=config['encoder']['attn_drop_rate'], 
                               embedding = config['embedding'], norm_layer=config['encoder']['norm_layer'], act_layer=config['encoder']['act_layer'], batch_size=config['loader']['batch_size'], dataset=config['dataset'])
        
        self.head = Head(in_dim=config['encoder']['dim'], out_dim=config['head']['out_dim'], use_bn=config['head']['use_bn'], norm_last_layer=config['head']['norm_last_layer'], 
                         nlayers=config['head']['n_layers'], hidden_dim=config['head']['hidden_dim'], bottleneck_dim=config['head']['bottleneck_dim'], drop_rate=config['encoder']['drop_rate'])

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr if isinstance(data.x,torch.Tensor) else None
        x = self.encoder(x, edge_index, edge_attr)
        x = self.head(x)
        return x
        

class DINO(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.config = config
        self.student = Network(self.config)
        self.teacher = Network(self.config)
        # create lookup table to vary the momentum coefficient along the training
        self.tau_lut = cosine_increase_law(self.config['training']['init_momentum'], self.config['training']['final_momentum'], float(self.config['training']['n_steps'])+1)
        
        # initialization with equal params
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data.copy_(param_s.data)
            param_t.requires_grad = False
    
    @torch.no_grad()
    def update_teacher(self, step:int):
        self.tau = self.tau_lut[step-1]
        for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
            param_t.data = self.tau * param_t.data + (1 - self.tau) * param_s.data

    def forward(self, data, mode="student"):
        if mode == "student":
            student_out = self.student(data)
            return student_out
        elif mode == "teacher":
            teacher_out = self.teacher(data)
            return teacher_out 

