import torch
from sklearn.metrics import accuracy_score, roc_auc_score
import torch.nn.functional as F

def test_supervised(model, z_train, test_data, device):
    # Impostiamo il modello in modalità di test
    model.eval()

    edge_index_test = test_data.edge_index.to(device)  # Gli archi di test
    edge_attr_test = test_data.edge_attr.to(device) if hasattr(test_data, 'edge_attr') else None

    # Predizione sugli archi di test
    with torch.no_grad():
        # Calcola la predizione sugli archi di test
        pred_test = model.predictor(z_train.to(device), edge_index_test)

    # Calcola la loss di link prediction (utilizzando le etichette di test)
    test_labels = test_data.edge_label.to(device)  # Etichette per gli archi di test (1 per i positivi, 0 per i negativi)
    test_loss = F.binary_cross_entropy_with_logits(pred_test, test_labels)

    # Calcolare AUC (Area Under Curve) come metrica di valutazione
    pred_test = torch.sigmoid(pred_test)
    pred_test_binary = (pred_test > 0.5).int()
    acc = accuracy_score(test_labels.cpu().numpy(), pred_test_binary.cpu().numpy())
    auc_score = roc_auc_score(test_labels.cpu().numpy(), pred_test_binary.cpu().numpy())
    print(f"Test Loss: {test_loss.item():.4f}")
    print(f"[Test] Accuracy: {acc:.4f} | AUC: {auc_score:.4f}")

    return test_loss, auc_score, acc, pred_test_binary