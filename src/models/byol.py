import torch
import torch.nn as nn
from .encoder import Encoder
from .head import Head
from ..utils.scheduler import cosine_increase_law

#TODO make byol work again

class Network(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.encoder = Encoder(dim=config['dim_e'], num_layers=config['n_layers_e'], num_heads=config['n_heads'],
                               mlp_ratio=config['mlp_ratio'], drop_rate=config['drop_rate'], attn_drop_rate=config['attn_drop_rate'], 
                               norm_layer=config['norm_layer'], act_layer=config['act_layer'], local_pool=config['local_pool'], pool_step=config['pool_step'])
        
        self.head = Head(in_dim=config['dim_e']*4, out_dim=config['dim_h'], use_bn=config['use_bn'], norm_last_layer=config['norm_last_layer'], 
                         nlayers=config['n_layers_h'], hidden_dim=config['hidden_dim'], bottleneck_dim=config['bottleneck_dim'], drop_rate=config['drop_rate'])

    def forward(self, data):
        x = self.encoder(data)
        x = self.head(x)
        return x
    

class Predictor(nn.Module):    
    def __init__(self, config) -> None:
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(config['dim_h'], int(config['dim_h'] * config['mlp_ratio'])),
            nn.GELU(),
            nn.Dropout(config['drop_rate']),
            nn.Linear(int(config['dim_h'] * config['mlp_ratio']), config['dim_h']),
            nn.Dropout(config['drop_rate'])
        )

    def forward(self, x):
        return self.projection(x)
    

class BYOCC(nn.Module):
    def __init__(self, config:dict):
        super().__init__()
        self.config = config
        self.student = Network(self.config)
        self.predictor = Predictor(self.config)
        self.teacher = Network(self.config)
        # create lookup table to vary the momentum coefficient along the training
        self.tau_lut = cosine_increase_law(self.config['init_momentum'], self.config['final_momentum'], float(self.config['n_steps'])+1)
        
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
            prediction = self.predictor(student_out)
            return prediction
        elif mode == "teacher":
            teacher_out = self.teacher(data)
            return teacher_out 
