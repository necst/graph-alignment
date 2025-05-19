import pandas as pd
import numpy as np
from umap import UMAP
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_reductions(data, labels, num_samples=5000, method='t-SNE', metric='cosine', downstream=False, axes=None, iteration=0):

    selected_samples = np.random.choice(data.shape[0], num_samples if num_samples <= data.shape[0] else data.shape[0], replace=False)
    features = data[selected_samples]
    labels = labels[selected_samples]

    if method == 't-SNE':
        reducer = TSNE(n_components=2, metric=metric)  
    elif method == 'UMAP':
        reducer = UMAP(n_components=2, n_neighbors=50, metric=metric)
    compressed_emb = reducer.fit_transform(features)

    if not downstream:
        class_names = {
            0: "disease",
            1: "drug",
            2: "function",
            3: "protein",
            4: "sideeffect"
        }
    else:
        class_names = {0: "Negative samples", 1: "Positive samples"}
        links_enumeration = {
            0: "drug_disease",
            1: "drug_sideeffect",
            2: "disease_protein",
            3: "function_function"
        }
    if not downstream:
        labels = np.array([class_names[label] for label in labels])

        fig, ax = plt.subplots(figsize=(6.3,4))
        palette = sns.color_palette("Set1", n_colors=len(set(labels)))
        df = pd.DataFrame({"x": compressed_emb[:, 0], "y": compressed_emb[:, 1], "label": labels})
        sns.scatterplot(data=df, x="x", y="y", ax=ax, hue="label", palette=palette, alpha=0.7, s=10)

        plt.legend(title="Classes")
        plt.xlabel(f'{method}', fontsize=16)
        plt.ylabel(f'{method}', fontsize=16)
        plt.tight_layout()

        fig.savefig("embedding_plot.png", bbox_inches='tight')

        return fig
    else:
        ax = axes[iteration]
        for class_label in np.unique(labels):
            indices = labels == class_label
            class_name = class_names.get(class_label, f"Class {class_label}")
            ax.scatter(compressed_emb[indices, 0], compressed_emb[indices, 1], label=class_name, alpha=0.7)

        ax.legend()
        ax.set_title(f't-SNE Plot ({links_enumeration[iteration]})')
    return None

def plot_tsne_umap_side_by_side(data, labels, num_samples=5000, metric='cosine'):
    selected_samples = np.random.choice(data.shape[0], num_samples, replace=False)
    features = data[selected_samples]
    labels = labels[selected_samples]

    class_names = {
        0: "disease",
        1: "drug",
        2: "function",
        3: "protein",
        4: "sideeffect"
    }
    # Mappa le etichette numeriche ai nomi
    labels_named = np.array([class_names[l] for l in labels])

    # Palette condivisa
    class_names = sorted(np.unique(labels_named))
    palette = sns.color_palette("Set1", n_colors=len(class_names))
    color_map = dict(zip(class_names, palette))

    # Riduzioni
    tsne_emb = TSNE(n_components=2, metric=metric).fit_transform(features)
    umap_emb = UMAP(n_components=2, n_neighbors=50, metric=metric).fit_transform(features)

    # Crea DataFrame per entrambe le riduzioni
    df_tsne = pd.DataFrame({
        'x': tsne_emb[:, 0],
        'y': tsne_emb[:, 1],
        'label': labels_named
    })

    df_umap = pd.DataFrame({
        'x': umap_emb[:, 0],
        'y': umap_emb[:, 1],
        'label': labels_named
    })

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.2))

    # Primo plot con legenda disattivata
    sns.scatterplot(data=df_tsne, x='x', y='y', hue='label', palette=color_map, ax=axes[0], s=10, alpha=0.7, legend=False)
    axes[0].set_title("t-SNE")

    # Secondo plot con legenda disattivata
    sns.scatterplot(data=df_umap, x='x', y='y', hue='label', palette=color_map, ax=axes[1], s=10, alpha=0.7, legend=False)
    axes[1].set_title("UMAP")

    # Crea i handle (etichette + colori) manualmente
    handles = [mpatches.Patch(color=color_map[name], label=name) for name in color_map]

    # Legenda globale sotto i grafici
    fig.legend(handles=handles, title="Classes", loc='lower center', ncol=5, frameon=False)

    # Aggiusta layout per fare spazio alla legenda sotto
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # lascia spazio in basso

    fig.savefig("embedding_plot.png", dpi=300, bbox_inches='tight')

    return fig