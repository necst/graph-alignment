from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler, autocast
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from torch_geometric.nn import VGAE

from ..models.utils.regularization import KoLeoLoss
from ..models.dino import DINO
from ..utils.scheduler import cosine_decay_lr, exponential_decay_lr, cosine_increase_law
from .augmenter import Augmenter
from .logger import Logger
import torch.nn.functional as F

idx = 0
class Trainer():
    def __init__(self, model: torch.nn.Module, loader: DataLoader, criterion:nn.Module,
                 optimizer: torch.optim.Optimizer, device: str, config: dict) -> None:
        # settings
        self.config = config
        self.device = device
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        self.model = model.to(self.device)
        self.loader = loader
        self.optimizer = optimizer
        self.criterion = criterion.to(device) if isinstance(model, DINO) else criterion
        self.is_vgae = isinstance(model, VGAE)
        self.uses_edge_attr = False if self.config["encoder"]["layer"] == "GCNConv" or self.config["encoder"]["layer"] == "SAGEConv" else True
        if config['training']['decay'] == 'cosine':
            self.scheduler = LambdaLR(optimizer, lambda epoch: cosine_decay_lr(epoch, float(config['training']['warmup']), float(config['training']['n_steps'])))
        elif config['training']['decay'] == 'exponential':
            self.scheduler = LambdaLR(optimizer, lambda epoch: exponential_decay_lr(epoch, config['warmup'], config['n_steps'])) # not implemented yet
        # create lookup table to vary the weight decay coefficient along the training
        self.wd_lut = cosine_increase_law(float(self.config['training']['init_wd']), float(self.config['training']['final_wd']), float(self.config['training']['n_steps'])+1)
        self.save_every = float(self.config['training']['save_every'])
        if config['training']['amp']:
            self.grad_scaler = GradScaler() # for cuda.amp  (automatic mixed precision)

        self.koleo = KoLeoLoss() # used as a regularizer

        self.augmenter = Augmenter(device=self.device)
        self.logger = Logger(self.config)
        self.created_output_dir = False


    def train(self):
        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.model.train()
        self.logger.start(self.device, num_params)
        step = 1
        train = True

        # trainig loop
        while train:

            for batch in self.loader:
                batch = batch.to(self.device)

                # perform the training step
                if self.config['model'] == 'DINO':
                    loss = self._step_DINO(batch, step)
                if self.config['model'] == 'GraphCL':
                    loss = self._step_CL(batch, step)
                if self.config['model'] == 'VGAE':
                    loss = self._step_VGAE(batch)
                if self.config['model'] == 'DeepGraphInfomax':
                    loss = self._step_DGI(batch)
                if self.config['model'] == 'Supervised':
                    loss = self._step_SUPERVISED(batch)

                # if training still going update LR and WD schedulers and log, otherwise stop the loop 
                if step <= float(self.config['training']['n_steps']):
                    self.logger.monitor(loss, step)
                    if step % self.save_every == 0:
                        self.logger.save_checkpoint(self.model, step)
                    self._update_schedulers(step) 
                    step += 1   
                else:
                    train = False
                    break

        self.logger.end(self.config)
        print('Done! :D')


    def _step_DINO(self, batch:Batch, step:int):
        # TODO check grad-data ratio (Karpathy tutorial)
        # grad reset
        self.optimizer.zero_grad(set_to_none=True) # set to none should be more efficient
        # generate augmentations
        protected_nodes = torch.arange(self.config['loader']['batch_size']) 
        views = self.augmenter.generate_augmentations(batch, protected_nodes=protected_nodes)
        # idx_t, idx_s = self.augmenter.get_common_nodes(views[0], views[1])
        # forward/backward pass  
        if self.config['training']['amp']:
            with autocast('cuda', dtype=torch.bfloat16):
                student_out = [self.model(view, mode="student") for view in views[1:]]
                teacher_out = [self.model(view, mode="teacher") for view in views[:1]]
                dino_loss = self.criterion(student_out[0][:self.config['loader']['batch_size'],:], teacher_out[0][:self.config['loader']['batch_size'],:], step)
                koleo_loss = 0.1*self.koleo(student_out[0])  # koleo regularization from dinov2
                loss = dino_loss + koleo_loss
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.student.parameters(), self.config['training']['grad_clip'])
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            student_out = [self.model(view, mode="student") for view in views[1:]]
            teacher_out = [self.model(view, mode="teacher") for view in views[:1]]
            dino_loss = self.criterion(student_out, teacher_out, step)
            koleo_loss = self.koleo(student_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.student.parameters(), self.config['training']['grad_clip'])
            self.optimizer.step()
        # update teacher params
        self.model.update_teacher(step)
        return loss


    def _step_CL(self, batch:Batch, step:int):
        # TODO check grad-data ratio (Karpathy tutorial)
        # grad reset
        self.optimizer.zero_grad(set_to_none=True) # set to none should be more efficient
        # generate augmentations
        protected_nodes = torch.arange(self.config['loader']['batch_size']) 
        views = self.augmenter.generate_augmentations(batch, protected_nodes=protected_nodes)
        # forward/backward pass  
        if self.config['training']['amp']:
            with autocast('cuda', dtype=torch.bfloat16):
                student_out = [self.model(view, mode=1) for view in views[1:]]
                teacher_out = [self.model(view, mode=2) for view in views[:1]]
                cl_loss = self.criterion(student_out[0][:self.config['loader']['batch_size'],:], teacher_out[0][:self.config['loader']['batch_size'],:])
                koleo_loss = 0.1*self.koleo(student_out[0])  # koleo regularization from dinov2
                loss = cl_loss + koleo_loss
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.branch1.parameters(), self.config['training']['grad_clip'])
            torch.nn.utils.clip_grad_norm_(self.model.branch2.parameters(), self.config['training']['grad_clip'])
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            student_out = [self.model(view, mode="student") for view in views[1:]]
            teacher_out = [self.model(view, mode="teacher") for view in views[:1]]
            cl_loss = self.criterion(student_out, teacher_out, step)
            koleo_loss = self.koleo(student_out)
            loss = cl_loss + koleo_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.student.parameters(), self.config['training']['grad_clip'])
            self.optimizer.step()
        return loss

    def _step_VGAE(self, batch:Batch):
        # grad reset
        self.optimizer.zero_grad()

        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr 
        # negative sampling
        neg_edge_index = negative_sampling(
            edge_index=edge_index,
            num_neg_samples=edge_index.size(1)
        )
        # check valid negatives 
        neg_edges_list = [
            edge for edge in neg_edge_index.T.tolist()
            if tuple(edge) not in self.config['edges_sets']['train']
            and tuple(edge) not in self.config['edges_sets']['valid']
            and tuple(edge) not in self.config['edges_sets']['test']
        ]
        neg_edge_index = torch.tensor(neg_edges_list, dtype=torch.long).T.to(self.device)
        # forward/backward pass  
        if self.config['training']['amp']:
            with autocast('cuda', dtype=torch.bfloat16):
                z = self.model.encode(x, edge_index, edge_attr if self.uses_edge_attr else None)
                recon_loss = self.model.recon_loss(z, edge_index, neg_edge_index)
                kl_loss = self.model.kl_loss() / x.size(0)
                loss = recon_loss + kl_loss
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            z = self.model.encode(x, edge_index, edge_attr if self.uses_edge_attr else None)
            recon_loss = self.model.recon_loss(z, edge_index, neg_edge_index)
            kl_loss = self.model.kl_loss() / x.size(0)
            loss = recon_loss + kl_loss
            loss.backward()
            self.optimizer.step()
        return loss


    def _step_DGI(self, batch:Batch):
        # grad reset
        self.optimizer.zero_grad(set_to_none=True)

        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        # forward/backward pass  
        if self.config['training']['amp']:
            with autocast('cuda', dtype=torch.bfloat16):
                pos_z, neg_z, summary = self.model(x, edge_index, edge_attr)
                loss = self.model.loss(pos_z, neg_z, summary)
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            pos_z, neg_z, summary = self.model(x, edge_index, edge_attr)
            loss = self.model.loss(pos_z, neg_z, summary)
            loss.backward()
            self.optimizer.step()
        return loss

    def _step_SUPERVISED(self, batch: Batch):
        global idx
        self.optimizer.zero_grad(set_to_none=True)

        x = batch.x.to(self.device)
        edge_index = batch.edge_index.to(self.device)
        edge_attr = batch.edge_attr.to(self.device) if hasattr(batch, 'edge_attr') else None
        n_id = batch.n_id.to(self.device)
        edge_label = batch.edge_label.to(self.device)
        #edge_label_index, edge_label = generate_edge_labels(batch, batch.neg_edge_index, idx)
        idx += 1
        conv_mask = edge_label != 0

        mask_of_interest = (edge_label != -1)
        self.criterion = torch.nn.BCEWithLogitsLoss()

        if self.config['training']['amp']:
            with autocast('cuda', dtype=torch.bfloat16):
                z = self.model.encoder(x, edge_index[:, conv_mask], edge_attr[conv_mask])  # GCN forward
                pred = self.model.predictor(z, edge_index, n_id)  # MLP link predictor
                loss = self.criterion(pred[mask_of_interest], edge_label[mask_of_interest])
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            z = self.model.encoder(x, edge_index[:, conv_mask], edge_attr[conv_mask])
            pred = self.model.predictor(z, edge_index, n_id)
            loss = F.binary_cross_entropy(pred[mask_of_interest], edge_label[mask_of_interest])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
            self.optimizer.step()

        return loss


    def _update_schedulers(self, step):
        # LR scheduling
        if self.config['training']['decay'] == 'cosine':
            self.scheduler.step()
        elif self.config['training']['decay'] == 'warm_restarts':
            if step <= self.config['warmup']:
                self.warmup_scheduler.step()
            else:
                self.restart_scheduler.step()
        # WD scheduling
        self._update_weight_decay(step)


    def _update_weight_decay(self, step):
        for param_group in self.optimizer.param_groups:
            param_group['weight_decay'] = self.wd_lut[step-1]

    @staticmethod
    def test_supervised(model, train_data, test_data, device):
            # Impostiamo il modello in modalità di test
        model.eval()

        # Esegui l'encoding utilizzando il grafo di addestramento
        # Usa i nodi e gli archi di train_data per l'encoding
        x_train = train_data.x.to(device)
        edge_index_train = train_data.edge_index.to(device)
        edge_attr_train = train_data.edge_attr.to(device) if hasattr(train_data, 'edge_attr') else None

        with torch.no_grad():
            # Calcola la rappresentazione dei nodi per il grafo di addestramento
            z_train = model.encoder(x_train, edge_index_train, edge_attr_train)

        # Ora, testiamo usando gli archi di test
        edge_index_test = test_data.edge_index.to(device)  # Gli archi di test
        edge_attr_test = test_data.edge_attr.to(device) if hasattr(test_data, 'edge_attr') else None

        # Predizione sugli archi di test
        with torch.no_grad():
            # Calcola la predizione sugli archi di test
            pred_test = model.predictor(z_train, edge_index_test)

        # Calcola la loss di link prediction (utilizzando le etichette di test)
        test_labels = test_data.edge_label.to(device)  # Etichette per gli archi di test (1 per i positivi, 0 per i negativi)
        test_loss = F.binary_cross_entropy(pred_test, test_labels)
        
        pred_test_binary = (pred_test > 0.5).int()

        # Calcolare AUC (Area Under Curve) come metrica di valutazione
        acc = accuracy_score(test_labels.cpu().numpy(), pred_test_binary.cpu().numpy())
        auc_score = roc_auc_score(test_labels.cpu().numpy(), pred_test_binary.cpu().numpy())
        print(f"Test Loss: {test_loss.item():.4f}")
        print(f"[Test] Accuracy: {acc:.4f} | AUC: {auc_score:.4f}")

        return test_loss, auc_score, acc
