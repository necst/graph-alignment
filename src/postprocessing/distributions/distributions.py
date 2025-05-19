import pandas as pd
import numpy as np
import seaborn as sns
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt


def distance_distributions(data, num_samples=1000):
    selected_samples = np.random.choice(data.shape[0], num_samples, replace=False)
    z = data[selected_samples]
    
    euclidean_distances = pdist(z, metric='euclidean')
    cosine_distances = pdist(z, metric='cosine')

    fig, ax = plt.subplots(1, 2, figsize=(12, 6)) 
    sns.histplot(euclidean_distances, kde=False, bins=50, color='skyblue', stat='density', ax=ax[0], alpha=0.7)
    ax[0].set_title('Distribution of Pairwise Euclidean Distances')
    ax[0].set_xlabel('Distance', fontsize=16)
    ax[0].set_ylabel('Density', fontsize=16)

    sns.histplot(cosine_distances, kde=False, bins=50, color='salmon', stat='density', ax=ax[1], alpha=0.7)
    ax[1].set_title('Distribution of Pairwise Cosine Distances')
    ax[1].set_xlabel('Distance', fontsize=16)
    ax[1].set_ylabel('Density', fontsize=16)
    plt.tight_layout()
    return fig
    

def feature_distributions(data, num_features=128, num_samples=1000):
    """
    Selects random features from the input array and plots their distribution across samples.

    Parameters:
        data (np.ndarray): Input array of shape [samples, embedding size].
        num_features (int): Number of random features to select.
        num_samples (int): Number of samples to consider for the plot.
"""
    selected_samples = np.random.choice(data.shape[0], num_samples, replace=False)
    selected_features = np.random.choice(data.shape[1], num_features, replace=False)
    subset_data = data[selected_samples][:, selected_features]

    representations = pd.DataFrame(subset_data, columns=[f'Feature {i}' for i in selected_features])
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=representations, ax=ax)
    ax.set_title('Distribution of Each Feature Across the Samples')
    ax.set_ylabel('Values Distribution')
    ax.set_xticks(range(0, len(selected_features)+8, 8))
    ax.set_xticklabels(range(0, len(selected_features)+8, 8))
    ax.set_xlabel('Features')
    return fig
    

def feature_variance_dstribution(data, num_samples=1000):
    selected_samples = np.random.choice(data.shape[0], num_samples, replace=False)
    subset_data = data[selected_samples]
    subset_data = subset_data.cpu().numpy()
    feature_variances = np.var(subset_data, axis=0)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(feature_variances, kde=False, ax=ax, stat='density', bins=50, alpha=0.6)
    plt.xlabel("Variance")
    plt.ylabel("Density")
    plt.show()
    return fig    