# -*- coding: utf-8 -*-
"""
Generate alignment plots similar to Figures 2 & 3 of the
"Platonic Representation Hypothesis" paper – but using *your* models.

**Changes 2025‑05‑15**
---------------------
* La repo non espone più `mutual_knn_overlap`. Ora usiamo la funzione
  `AlignmentMetrics.measure('mutual_knn', …)` presente in `metrics.py` del
  progetto ufficiale.
* Di conseguenza, l'allineamento si calcola direttamente sugli embedding
  normalizzati invece che sulle matrici di similarità.

Come funziona (high‑level)
--------------------------
1. **Configuration** – YAML file con:
   * lista dei modelli (nome, path, metrica di competenza).
   * dataset per gli embedding (single‑modal) e facoltativo paired dataset
     (cross‑modal).
2. Embed di tutti i dati → cache su disco (`*.npy`).
3. Calcolo dell'allineamento con `metrics.AlignmentMetrics` → CSV e JSON.
4. Riproduzione di due plot (Fig‑2 & Fig‑3 analoghi).

Installazione rapide dipendenze
-------------------------------
```bash
pip install torch torchvision  # o torch‑geometric ecc.
pip install numpy scipy pandas scikit-learn umap-learn matplotlib tqdm
pip install git+https://github.com/minyoungg/platonic-rep.git  # contiene metrics.py
```

Esecuzione tipica
-----------------
```bash
python generate_alignment_plots.py configs/my_models.yaml \
       --output_folder results/
```

La sezione finale del file contiene un **template YAML** di esempio.
"""

import argparse
import json
import math
import os
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import umap
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from src.downstream.downstream_utils import find_all_runs, read_config_info
from src.postprocessing.alignment.metrics import AlignmentMetrics
from src.utils.config_parser import ConfigParser, create_shared_loader
import seaborn as sns
from src.utils.model_loader import ModelLoader

import src.alignment.AlignmentMetrics

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

from pathlib import Path
from datetime import date
import os
import math

def _get_output_path(base_path: Path, metric: str) -> Path:
    metric_dir = base_path.joinpath(metric)
    os.mkdir(metric_dir)
    return metric_dir

def get_base_plot_path() -> Path:
    root = Path(os.getcwd())
    today = date.today()
    plots_dir = root / 'plots'
    plots_day = plots_dir / f'{today.month}_{today.day}_{today.year}'

    if not plots_dir.exists():
        os.mkdir(plots_dir)

    if not plots_day.exists():
        os.mkdir(plots_day)
        num_iter = 1
    else:
        existing = [d for d in os.listdir(plots_day) if os.path.isdir(plots_day / d)]
        num_iter = (len(existing) + 1)

    curr_plot = plots_day / f'plots_{num_iter}'
    os.makedirs(curr_plot, exist_ok=True)
    return curr_plot


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ------------------------------------------------------------
# Alignment (mutual‑kNN)
# ------------------------------------------------------------

def kernel_alignment(emb1: np.ndarray, emb2: np.ndarray, k: int = 25, metric="mutual_knn") -> float:
    """Alignment basato su *mutual k‑NN* (repo PRH).

    Converte gli embedding in tensor, li normalizza (L2) e chiama
    `metrics.AlignmentMetrics.measure('mutual_knn', …)`.
    """
    import torch.nn.functional as F
    if not isinstance(emb1, torch.Tensor):
        emb1 = torch.from_numpy(emb1).to(torch.float32)
    if not isinstance(emb2, torch.Tensor):
        emb2 = torch.from_numpy(emb2).to(torch.float32)

    emb1 = F.normalize(emb1, dim=-1)
    emb2 = F.normalize(emb2, dim=-1)

    return AlignmentMetrics.mutual_knn(emb1, emb2, k) if metric=="mutual_knn" else  AlignmentMetrics.cka(emb1, emb2)


def build_alignment_matrix(model_names: List[str], embs: Dict[str, np.ndarray], k: int = 25, metric="mutual_knn") -> pd.DataFrame:
    mat = np.zeros((len(model_names), len(model_names)), dtype=np.float32)
    for i, m1 in enumerate(model_names):
        for j, m2 in enumerate(model_names):
            if i >= j:
                continue
            mat[i, j] = mat[j, i] = kernel_alignment(embs[m1], embs[m2], k=k, metric=metric)
    return pd.DataFrame(mat, index=model_names, columns=model_names)


# ------------------------------------------------------------
# Plotting – Figure 2 analogue
# ------------------------------------------------------------

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_fig2(alignment_df: pd.DataFrame, perf: Dict[str, float], out_dir: Path, metric="mutual_knn"):
    # Calcola le medie di allineamento
    mean_alignments = alignment_df.mean(axis=1)
    model_names = alignment_df.index.tolist()
    model_perfs = [perf[m] for m in model_names]

    # Crea dataframe per plotting
    df = pd.DataFrame({
        "model": model_names,
        "alignment": mean_alignments.values,
        "performance": model_perfs,
    })
    df["type"] = df["model"].apply(lambda m: "Supervised" if m.startswith("Supervised") else "Unsupervised")

    # Colori con Seaborn ma plot con matplotlib
    sns.set(style="whitegrid")
    palette = {"Supervised": "#4C72B0", "Unsupervised": "#55A868"}

    plt.figure(figsize=(6, 4))
    ax = plt.gca()

    for cat in df["type"].unique():
        sub = df[df["type"] == cat]
        ax.scatter(sub["performance"], sub["alignment"], s=50, label=cat, color=palette[cat])

    # Regressione globale (unica retta)
    m, b = np.polyfit(df["performance"], df["alignment"], deg=1)
    xs = np.linspace(df["performance"].min(), df["performance"].max(), 100)
    ax.plot(xs, m * xs + b, color="red", lw=2)

    # Zoom out
    x_min, x_max = df["performance"].min(), df["performance"].max()
    y_min, y_max = df["alignment"].min(), df["alignment"].max()
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_xlabel("Model competence")
    ax.set_ylabel(f"Mean alignment to others ({metric})")
    ax.set_title("Alignment vs Competence")
    ax.legend(title="Model type")
    plt.tight_layout()
    plt.savefig(out_dir / f"fig2_alignment_vs_competence_{metric}.png", dpi=300)
    plt.close()


    # # UMAP
    # with np.errstate(divide='ignore'):
    #     dist = -np.log(alignment_df.values + 1e-9)
    #
    # reducer = umap.UMAP(
    #     metric="precomputed",
    #     random_state=0,
    #     spread=1.0,  # default: 1.0
    #     min_dist=0.05,  # più piccolo = punti più vicini
    #     n_neighbors=10  # meno smoothing globale
    # )
    # xy = reducer.fit_transform(dist)
    # perf_vals = np.array([perf[m] for m in alignment_df.index])
    # norm = (perf_vals - perf_vals.min()) / (np.ptp(perf_vals) + 1e-9)
    # plt.figure(figsize=(5, 5))
    # sc = plt.scatter(
    #     xy[:, 0], xy[:, 1],
    #     c=perf_vals,
    #     cmap="viridis",  # oppure "viridis_r" se vuoi che blu significhi alta competenza
    #     s=60,
    #     edgecolors="k"
    # )
    # for i, m in enumerate(alignment_df.index):
    #     plt.text(xy[i, 0], xy[i, 1], m, fontsize=8)
    # # Riquadro (bordo) attorno al plot
    # ax = plt.gca()
    # for spine in ax.spines.values():
    #     spine.set_visible(True)
    #     spine.set_linewidth(1.5)
    #     spine.set_edgecolor("gray")
    #
    # ax.set_xticks([])
    # ax.set_yticks([])
    # plt.title(f"UMAP of representations ({metric})")
    # plt.colorbar(sc, label="Model competence")
    # plt.tight_layout()
    # plt.savefig(out_dir / f"fig2_umap_{metric}.png", dpi=300)
    # plt.close()


# ------------------------------------------------------------
# Plotting – Figure 3 analogue
# ------------------------------------------------------------

def plot_fig3(cross_align: Dict[Tuple[str, str], float], perf: Dict[str, float], out_dir: Path, metric="mutual_knn"):
    rows = []
    for (lm, rm), align in cross_align.items():
        score = perf.get(rm)
        if score is not None:
            rows.append({
                "left_model": lm,
                "right_model": rm,
                "align": align,
                "perf": float(score),
                "marker_group": "Supervised" if rm.startswith("Supervised") else rm.split("_")[0]
            })
    df = pd.DataFrame(rows)

    # Setup colori e marker
    unique_left = df["left_model"].unique()
    color_map = {lm: get_cmap("tab10")(i % 10) for i, lm in enumerate(unique_left)}

    unique_markers = df["marker_group"].unique()
    marker_styles = ["o", "s", "^", "D", "X", "P", "v"]
    marker_map = {k: marker_styles[i % len(marker_styles)] for i, k in enumerate(unique_markers)}

    # Setup plot
    plt.figure(figsize=(6.3, 3.2))
    sns.set(style="whitegrid")
    ax = plt.gca()

    # Disegna i punti con colore = left_model, forma = marker_group
    for (lm, marker_group), group in df.groupby(["left_model", "marker_group"]):
        color = color_map[lm]
        marker = marker_map[marker_group]
        ax.scatter(
            group["perf"],
            group["align"],
            color=color,
            marker=marker,
            s=70,
            label=f"{lm} / {marker_group}"
        )

    for lm, group in df.groupby("left_model"):
        if len(group) >= 2:
            m, b = np.polyfit(group["perf"], group["align"], deg=1)
            xs = np.linspace(group["perf"].min(), group["perf"].max(), 100)
            ax.plot(xs, m * xs + b, color=color_map[lm], linewidth=2)

    ax.set_xlabel("Model competence")
    ax.set_ylabel(f"Alignment to DINO transformer ({metric})")
    ax.set_title("")

    # Zoom out
    x_pad = (df["perf"].max() - df["perf"].min()) * 0.1
    y_pad = (df["align"].max() - df["align"].min()) * 0.1
    ax.set_xlim(df["perf"].min() - x_pad, df["perf"].max() + x_pad)
    ax.set_ylim(df["align"].min() - y_pad, df["align"].max() + y_pad)

    # Costruisci legenda separata: colori per left_model, forme per marker_group
    color_legend = [
        Patch(facecolor=color_map[lm], label=lm) for lm in unique_left
    ]
    marker_legend = [
        Line2D([0], [0], marker=marker_map[m], color="gray", label=m,
               markerfacecolor="gray", markersize=8, linestyle="None")
        for m in unique_markers
    ]

    first_legend = ax.legend(handles=color_legend, title="Left model (color)", loc="upper left", bbox_to_anchor=(1.0, 1))
    second_legend = ax.legend(handles=marker_legend, title="Right model (marker)", loc="lower left", bbox_to_anchor=(1.0, 0))
    ax.add_artist(first_legend)

    plt.tight_layout()
    plt.savefig(out_dir / f"fig3_alignment_to_DINO_{metric}.png", dpi=300)
    plt.close()

def plot_intra_group_alignment(alignment_df: pd.DataFrame, supervised_names: list[str], out_path: Path, metric="mutual_knn"):
    # Tutti i modelli nel plot
    all_models = alignment_df.index.tolist()

    # Divide supervised e unsupervised
    supervised_set = set(supervised_names)
    unsupervised_set = set(all_models) - supervised_set

    def average_alignment_within(group):
        group = list(group)
        total, count = 0.0, 0
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = alignment_df.loc[group[i], group[j]]
                total += a
                count += 1
        return total / count if count > 0 else 0.0

    supervised_score = average_alignment_within(supervised_set)
    unsupervised_score = average_alignment_within(unsupervised_set)

    # Bar plot
    plt.figure(figsize=(3, 3))
    plt.bar(["Supervised", "Unsupervised"], [supervised_score, unsupervised_score], color=["#4C72B0", "#55A868"])
    plt.ylabel("Average intra-group alignment")
    plt.title(f"{metric}")
    plt.tight_layout()
    plt.savefig(out_path / f"intra_group_alignment_{metric}.png", dpi=300)
    plt.close()


def load_yaml(path: Path) -> Dict:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)



# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def main(cfg_path: Path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_yaml(cfg_path)

    models_cfg = cfg["models"]
    k = cfg.get("knn_k", 25)
    parser = ConfigParser()
    data = parser._load_dataset(parser, "BioKG")
    shared_loader, shared_indices = create_shared_loader(
        data,
        n_samples=5000,
        n_neighbors=[30, 30],
        batch_size=256
    )
    embeddings, perf = {}, {}
    for m in models_cfg:
        name = m["name"]
        perf[name] = float(m["score"])
        embeddings[name] = parser.load_run(run=m['run'], checkpoint=m['checkpoint'], n_samples=1024, return_layers=False,
                                    shuffle_samples=True, shared_loader=shared_loader)[0]


    # to get all nodes, set n_sample to None



    model_names = list(embeddings.keys())
    base_path = get_base_plot_path()
    for metric in ['mutual_knn', 'cka']:
        output_folder = _get_output_path(base_path, metric)
        align_df = build_alignment_matrix(model_names, embeddings, k=k, metric=metric)
        align_df.to_csv(output_folder / f"alignment_matrix_{metric}.csv")
        plot_fig2(align_df, perf, output_folder, metric=metric)

        # ⬇️ Nuova chiamata: intra-group alignment plot
        supervised_names = [m for m in model_names if m.startswith("Supervised")]
        plot_intra_group_alignment(align_df, supervised_names, output_folder, metric=metric)

    # cross‑modal
    if "align_to_one" in cfg:
        cross_cfg = cfg["align_to_one"]
        lm_names = cross_cfg["left_model"]
        rm_names = cross_cfg["right_models"]
        for metric in ['mutual_knn', 'cka']:
            cross_align = {
                (lm, rm): kernel_alignment(embeddings[lm], embeddings[rm], k=cfg.get("knn_k", 10), metric=metric)
                for lm in lm_names for rm in rm_names
            }
            #json.dump(cross_align, open(output_folder / "alignment_to_DGI.json", "w"), indent=2)
            plot_fig3(cross_align, perf, base_path, metric)
    print("Done. All plots saved to", base_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Reproduce PRH alignment plots on custom models")
    p.add_argument("config", type=Path)
    args = p.parse_args()
    main(args.config)

#run this from termimal python alignment_plots.py config_alignment.yaml
