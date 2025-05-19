import numpy as np
import torch
import seaborn as sns
from matplotlib import pyplot as plt

from src.postprocessing.alignment.metrics import AlignmentMetrics
from src.utils.config_parser import ConfigParser
from src.utils.model_loader import ModelLoader


def setup(run, checkpoint, n_samples, return_layers, shuffle_samples, test=False):
    parser = ConfigParser()
    # to get all nodes, set n_sample to None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_loader = ModelLoader(run, checkpoint, device)
    z, labels = parser.load_run(run=run, checkpoint=checkpoint, n_samples=n_samples, return_layers=return_layers, shuffle_samples=shuffle_samples, test=test)
    return z, labels, parser, model_loader, device


def compute_alignment_matrix(z, metric='cka', trials=5):
    num_layers = len(z)
    alignment_matrix = np.zeros((num_layers, num_layers))

    for i in range(num_layers):
        for j in range(num_layers):
            scores = []
            for _ in range(trials):
                score = AlignmentMetrics.measure(metric, z[i], z[j])
                scores.append(score)
            alignment_matrix[i, j] = np.mean(scores)

    return alignment_matrix


def plot_alignment_matrix(matrix, title="Layer Alignment Matrix"):
    plt.figure(figsize=(6, 6))
    ax = sns.heatmap(matrix[::-1], cmap='magma', square=True, vmin=0, vmax=1, cbar_kws={'shrink': 0.8})
    plt.xlabel("Layer")
    plt.ylabel("Layer")
    plt.title(title)

    # Fix y-axis tick labels
    num_layers = matrix.shape[0]
    ax.set_yticks(np.arange(num_layers) + 0.5)
    ax.set_yticklabels(reversed(range(num_layers)))

    plt.show()


def compute_norm_ratios(skip_connections, long_branch_outputs):
    """
    Computes the norm ratio ||z_i||/||f(z_i)|| for each layer

    Parameters:
    skip_connections: List of tensors [n_samples, embedding_dim] for each layer's skip connection
    long_branch_outputs: List of tensors [n_samples, embedding_dim] for each layer's long branch output

    Returns:
    norm_ratios: List of average norm ratios for each layer
    """
    norm_ratios = []

    for i in range(len(skip_connections)):
        z_i = skip_connections[i]  # hidden representation from skip connection
        f_z_i = long_branch_outputs[i]  # transformation from long branch

        # Compute norms
        if isinstance(z_i, torch.Tensor):
            # For PyTorch tensors
            z_i_norms = torch.norm(z_i, dim=1)
            f_z_i_norms = torch.norm(f_z_i, dim=1)

            # Calculate ratios and mean
            ratio = (z_i_norms / f_z_i_norms).mean().item()
        else:
            # For NumPy arrays
            z_i_norms = np.linalg.norm(z_i, axis=1)
            f_z_i_norms = np.linalg.norm(f_z_i, axis=1)

            # Calculate ratios and mean
            ratio = np.mean(z_i_norms / f_z_i_norms)

        norm_ratios.append(ratio)

    return norm_ratios

def plot_norm_ratios(norm_ratios):
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(norm_ratios) + 1), norm_ratios)
    plt.xlabel('Layer')
    plt.ylabel('Norm Ratio ||z_i||/||f(z_i)||')
    plt.title('Norm Ratios Across ViT Layers')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(range(1, len(norm_ratios) + 1))
    plt.show()


def plot_embedding_norms(mlp_embeddings, attention_embeddings, cls_mlp_embeddings=None,
                         cls_attention_embeddings=None, resnet_embeddings=None):
    """
    Plots the average norm of embeddings for different components across blocks

    Parameters:
    mlp_embeddings: List of tensors containing MLP embeddings for each block
    attention_embeddings: List of tensors containing self-attention embeddings for each block
    cls_mlp_embeddings: Optional list of tensors for CLS token MLP embeddings
    cls_attention_embeddings: Optional list of tensors for CLS token attention embeddings
    resnet_embeddings: Optional list of tensors for ResNet embeddings
    """

    # Calculate average norms for each type of embedding
    def calculate_avg_norms(embedding_list):
        avg_norms = []
        for embedding in embedding_list:
            if isinstance(embedding, torch.Tensor):
                # For PyTorch tensors
                norms = torch.norm(embedding, dim=1).mean().item()
            else:
                # For NumPy arrays
                norms = np.mean(np.linalg.norm(embedding, axis=1))
            avg_norms.append(norms)
        return avg_norms

    mlp_norms = calculate_avg_norms(mlp_embeddings)
    attention_norms = calculate_avg_norms(attention_embeddings)

    # Optional norm calculations
    cls_mlp_norms = calculate_avg_norms(cls_mlp_embeddings) if cls_mlp_embeddings else None
    cls_attention_norms = calculate_avg_norms(cls_attention_embeddings) if cls_attention_embeddings else None
    resnet_norms = calculate_avg_norms(resnet_embeddings) if resnet_embeddings else None

    # Create x-axis indices
    block_indices = np.arange(max(len(mlp_norms), len(attention_norms)))

    # Create plot
    plt.figure(figsize=(10, 6))

    # Plot lines for each type of embedding
    plt.plot(block_indices[:len(attention_norms)], attention_norms, 'b-', linewidth=2, label='Self-Attention')
    plt.plot(block_indices[:len(mlp_norms)], mlp_norms, 'purple', linewidth=2, label='MLP')

    # Plot optional lines if data was provided
    if cls_attention_norms:
        plt.plot(block_indices[:len(cls_attention_norms)], cls_attention_norms, 'b--',
                 linewidth=1.5, label='Self-Attention CLS')

    if cls_mlp_norms:
        plt.plot(block_indices[:len(cls_mlp_norms)], cls_mlp_norms, 'purple',
                 linestyle='dotted', linewidth=1.5, label='MLP CLS')

    if resnet_norms:
        plt.plot(block_indices[:len(resnet_norms)], resnet_norms, 'g-',
                 linewidth=2, label='ResNet50')

    # Setup plot aesthetics
    plt.grid(True)
    plt.xlabel('Block Index')
    plt.ylabel('Average norm of representation')
    plt.title('Ratio of Norms (aggregated)')
    plt.legend()
    plt.ylim(bottom=0)  # Start y-axis at 0
    plt.tight_layout()

    return plt

def compute_embedding_distances(embeddings_per_layer):
    """
    Computes the average Euclidean distance between embeddings in consecutive layers.

    embeddings_per_layer: list of tensors, where embeddings_per_layer[i]
    is the embedding tensor at layer i.
    """
    distances = []
    for i in range(len(embeddings_per_layer) - 1):
        emb1 = embeddings_per_layer[i]
        emb2 = embeddings_per_layer[i + 1]

        # Compute Euclidean distance between embeddings of the same node in consecutive layers
        dist = torch.norm(emb1 - emb2, dim=1)

        # Save the average distance
        distances.append(dist.mean().item())

    return distances

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import torch

def select_representative_nodes(labels):
    """
    Selects one random node per class from the label vector.

    Args:
    - labels (numpy array or torch tensor): A vector of node labels.

    Returns:
    - List of selected node indices.
    """
    unique_classes = np.unique(labels)  # Get unique class labels
    selected_nodes = []

    for cls in unique_classes:
        # Get indices of nodes belonging to class `cls`
        class_indices = np.where(labels == cls)[0]

        # Randomly select one node from this class
        if len(class_indices) > 0:
            selected_nodes.append(np.random.choice(class_indices))

    return selected_nodes

def plot_embedding_trajectory(embeddings_per_layer, num_nodes=5, labels=None):
    """
    Projects embeddings to 2D using PCA and plots the trajectory of selected nodes.

    embeddings_per_layer: list of tensors, where each tensor represents embeddings at a layer.
    num_nodes: number of nodes to visualize (randomly selected).
    """
    # Concatenate all embeddings from all layers to compute PCA
    all_embeddings = torch.cat(embeddings_per_layer, dim=0).cpu().numpy()
    pca = TSNE(n_components=2)
    reduced_embeddings = pca.fit_transform(all_embeddings)

    # Split reduced embeddings layer by layer after PCA
    num_layers = len(embeddings_per_layer)
    num_total_nodes = embeddings_per_layer[0].shape[0]
    reduced_per_layer = np.array_split(reduced_embeddings, num_layers)

    # Select a few random nodes to track their trajectory
    selected_nodes = select_representative_nodes(labels)

    plt.figure(figsize=(8, 6))

    # Plot trajectories
    for node in selected_nodes:
        trajectory = np.array([reduced_per_layer[i][node] for i in range(num_layers)])
        plt.plot(trajectory[:, 0], trajectory[:, 1], marker="o", linestyle="-", label=f"Node {node}")

    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Embedding Trajectories Across Layers")
    plt.legend()
    plt.show()
