import torch.nn.functional as F
from torch_geometric.nn import VGAE
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
class VGAEWithEdgeAttr(VGAE):
    def __init__(self, encoder):
        super().__init__(encoder)

    def decode(self, z, edge_index, edge_attr):
        """
        Decodifica la struttura del grafo e integra edge_attr.
        """
        i, j = edge_index
        edge_logits = (z[i] * z[j]).sum(dim=-1)  # Classico VGAE decoder
        edge_attr_logits = torch.sum(edge_attr, dim=-1)  # Semplice somma

        return edge_logits + edge_attr_logits  # Combina struttura ed edge_attr

    def recon_loss(self, z, pos_edge_index, neg_edge_index, edge_attr):
        """
        Calcola la loss considerando edge_attr.
        """
        # Ricostruzione degli archi positivi
        pos_loss = -torch.log(torch.sigmoid(self.decode(z, pos_edge_index, edge_attr)) + 1e-15).mean()

        # Ricostruzione degli archi negativi (edge_attr impostato a 0 perché non conosciuto)
        neg_edge_attr = torch.zeros(neg_edge_index.shape[1], edge_attr.shape[1], device=edge_attr.device)
        neg_loss = -torch.log(1 - torch.sigmoid(self.decode(z, neg_edge_index, neg_edge_attr)) + 1e-15).mean()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index, edge_attr):
        """
        Valuta il modello su dati di test, considerando sia la ricostruzione della struttura sia degli edge_attr.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))


            # Edge attributes
        neg_edge_attr = torch.zeros(neg_edge_index.shape[1], edge_attr.shape[1], device=edge_attr.device) if edge_attr is not None else None

            # Ricostruzione
        pos_pred = torch.sigmoid(self.decode(z, pos_edge_index, edge_attr))
        neg_pred = torch.sigmoid(self.decode(z, neg_edge_index, neg_edge_attr))

            # Concatenazione per valutazione
        y_true = torch.cat([pos_y, neg_y], dim=0).cpu().numpy()
        y_pred = torch.cat([pos_pred, neg_pred], dim=0).cpu().numpy()

        # Calcolo metriche
        auc = roc_auc_score(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)

        return auc,ap