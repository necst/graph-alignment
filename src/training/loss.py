import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.scheduler import linear_increase_law


class DINOLoss(nn.Module):
    def __init__(self, config, device='cuda'):
        super().__init__()
        self.config = config
        self.student_temp = config['training']['student_temp']
        self.register_buffer('center', torch.zeros(1, config['head']['out_dim']))
        self.center_momentum = config['training']['center_momentum']
        self.device = device
        # create lookup table to vary the teacher temperature coefficient along the training
        self.teacher_temp_lut = linear_increase_law(self.config['training']['init_teacher_temp'], self.config['training']['final_teacher_temp'], float(self.config['training']['warmup'])+1)
        
    
    def forward(self, student_output, teacher_output, step): #idx_t, idx_s, step) -> float:
        # update teacher temperature
        if step <= float(self.config['training']['warmup']):
            self.teacher_temp = self.teacher_temp_lut[step-1]

        student_out = student_output / self.student_temp
        # detach since teacher representations should not influence the loss
        # is the student that should adapt to match, distillation is an asymmetric process
        # teacher_out = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach() for t in teacher_output]  
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1).detach() 
        
        # for multiple augmentations
        # n_loss_terms = 0 
        # for iq, q in enumerate(teacher_out):
        #     for v in range(len(student_out)):
        #         # if v == iq:   # enable this piece only when some inputs are shared across the two networks
        #         #     continue
        #         loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
        #         total_loss += loss.mean()
        #         n_loss_terms += 1
        # total_loss /= n_loss_terms
        total_loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=1), dim=1).mean()
        self.update_center(teacher_output)
        return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output):
        # update center of the teacher representations
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)  / len(teacher_output)
        self.center.data = self.center.data * 0.9 + batch_center * 0.1


class BYOCCLoss(nn.Module):
    def __init__(self, config, device='cuda') -> None:
        super().__init__()
        self.config = config
        self.student_temp = config['student_temp']
        self.register_buffer('center', torch.zeros(1, config['dim_h']))
        self.center_momentum = config['center_momentum']
        self.device = device
        # create lookup table to vary the teacher temperature coefficient along the training
        self.teacher_temp_lut = linear_increase_law(self.config['init_teacher_temp'], self.config['final_teacher_temp'], float(self.config['warmup'])+1)


    def forward(self, student_output, teacher_output, step) -> float:
        r"""
        Compute cosine similarity between online prediction and target projection.
        """
        cos_sim = nn.CosineSimilarity()

        # BYOCC approach with representation normalization
        # normalize the predictions
        student_norm = nn.functional.normalize(student_output, dim=1)
        teacher_norm = nn.functional.normalize(teacher_output, dim=1).detach()

        # averaging the similarity (2-2*sim amplifies differences) 
        sim = cos_sim(student_norm, teacher_norm)
        loss = 2 - 2 * torch.mean(sim, dim=0)
        return loss
        
        """
        student_norm = [nn.functional.normalize(s, dim=1) for s in student_output]
        teacher_norm = [nn.functional.normalize(t, dim=1).detach() for t in teacher_output]
        n_loss_terms = 0
        for iq, q in enumerate(teacher_norm):
            for v in range(len(student_norm)):
                if v == iq:   # enable this piece only when some inputs are shared across the two networks
                    continue
                sim = cos_sim(q, student_norm[v])
                total_loss += 2 - 2 * torch.mean(sim, dim=0)
                n_loss_terms += 1
                """

        """
        # DINO approach but with CosineSimilarity
        # update teacher temperature
        if step <= float(self.config['warmup']):
            self.teacher_temp = self.teacher_temp_lut[step-1]
        student_out = [s / self.student_temp for s in student_output]
        teacher_out = [((t - self.center) / self.teacher_temp).detach() for t in teacher_output]  
        
        total_loss = torch.tensor(0.0, device=self.device)
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:   # enable this piece only when some inputs are shared across the two networks
                    continue
                sim = cos_sim(q, student_out[v])
                total_loss += 2 - 2 * torch.mean(sim, dim=0)
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(torch.cat(teacher_output, dim=0))
        return total_loss
        """

    @torch.no_grad()
    def update_center(self, teacher_output):
        # update center of the teacher representations
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)  / len(teacher_output)
        self.center.data = self.center.data * 0.9 + batch_center * 0.1


def NTXentLoss(z1, z2, temperature=0.1):
    """
    Compute NT-Xent loss for contrastive learning.
    
    Args:
        z1: Tensor of shape (N, D) containing first set of embeddings
        z2: Tensor of shape (N, D) containing second set of embeddings
            z1[i] and z2[i] are different augmentations of the same input
        temperature: Softmax temperature parameter
    
    Returns:
        Scalar loss value
    """
    # Check input dimensions
    assert z1.size(0) == z2.size(0), "Batch sizes must be equal"
    batch_size = z1.size(0)
    
    # Normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # Concatenate embeddings from both augmentations
    z = torch.cat([z1, z2], dim=0)  # Shape: (2N, D)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T) / temperature  # Shape: (2N, 2N)
    
    # Create mask for positive pairs
    pos_mask = torch.zeros_like(sim_matrix)
    
    # Mark positive pairs: (i, i+N) and (i+N, i)
    for i in range(batch_size):
        pos_mask[i, i + batch_size] = 1
        pos_mask[i + batch_size, i] = 1
    
    # We need to exclude self-similarity (diagonal) from the denominator
    diag_mask = torch.eye(2 * batch_size, device=z.device)
    
    # For each sample i, compute loss
    loss = 0
    for i in range(2 * batch_size):
        # Numerator: similarity with positive pair (excluding self)
        # For each sample, there's exactly one positive pair
        numerator = torch.exp(torch.sum(sim_matrix[i] * pos_mask[i]))
        
        # Denominator: sum of similarities with all samples except self
        denominator = torch.sum(torch.exp(sim_matrix[i]) * (1 - diag_mask[i]))
        
        # Add to loss
        loss -= torch.log(numerator / denominator)
    
    # Average loss over all samples
    return loss / (2 * batch_size)