import itertools
from copy import deepcopy
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import tqdm
from sklearn.decomposition import PCA
from torch.nn import functional as F

from icl.figures.colors import PRIMARY, SECONDARY

sns.set_style("whitegrid")


def plot_loss_trace(batch_losses, likelihoods, title=None):
    fig, ax = plt.subplots(figsize=(10, 5))

    batch_losses['mean'] = [float(x.mean()) for x in batch_losses['mean']]  
    sns.lineplot(data=batch_losses, x="draw", y="mean", hue="chain", palette="tab20", ax=ax, alpha=0.8)

    twin_ax = ax.twinx()

    likelihoods = likelihoods.groupby("draw").mean().reset_index()
    likelihoods.sort_values(by="draw", inplace=True)

    if "llc/mean/mean" in likelihoods.columns:
        sns.lineplot(data=likelihoods, x="draw", y="llc/mean/mean", ax=twin_ax, alpha=1., color="black")
        twin_ax.fill_between(likelihoods['draw'], likelihoods['llc/mean/mean'] - likelihoods['llc/mean/std'], likelihoods['llc/mean/mean'] + likelihoods['llc/mean/std'], alpha=0.2, color="black")
    else:
        sns.lineplot(data=likelihoods, x="draw", y="llc/mean", ax=twin_ax, alpha=1., color="black")
        twin_ax.fill_between(likelihoods['draw'], likelihoods['llc/mean'] - likelihoods['llc/std'], likelihoods['llc/mean'] + likelihoods['llc/std'], alpha=0.1, color="black")

    ax.set_ylabel(r"Batch Loss. $L^{(\tau)}_m$")
    # twin_ax.set_ylabel(r"LLC, $\hat\lambda_\tau$", color=PRIMARY)
    twin_ax.set_ylabel(r"LLC, $\hat\lambda_\tau$")

    # for label in twin_ax.get_yticklabels():
    #     label.set_color(PRIMARY)

    ax.set_xlabel(r"Draw, $\tau$")
    ax.legend().remove()

    if title is not None:
        ax.set_title(title)
        plt.tight_layout()

    return fig



def plot_explained_variance(pca, title="Explained Variance", ax: Optional[plt.Axes] = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 8))

    ax.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)

    for i, ratio in enumerate(pca.explained_variance_ratio_):
        ax.text(i, ratio, f"{ratio:.2f}", fontsize=12, ha='center', va='bottom')

    ax.set_title(title)
    ax.set_xlabel('PC')
    ax.set_ylabel('Explained Variance')

    ax.set_xticks(range(len(pca.explained_variance_ratio_)), range(1, len(pca.explained_variance_ratio_) + 1))


def plot_weights_trace(model, deltas, xs, ys, device='cpu', num_components=3, num_points=10):
    model.to(device)
    xs.to(device)
    ys.to(device)

    num_chains = deltas.shape[0]
    num_draws = deltas.shape[1]

    pca = PCA(n_components=num_components)

    weights_reduced = pca.fit_transform(deltas.reshape(num_chains * num_draws, -1))

    def get_pc_landscape(pca, fn, pc1, pc2, pc1_lim: Tuple[int, int], pc2_lim: Tuple[int, int], num_points=100, ax=None):
        xx, yy = np.meshgrid(np.linspace(*pc1_lim, num_points), np.linspace(*pc2_lim, num_points))

        # Compute function values for the grid
        Z = np.zeros(xx.shape)
        for i in tqdm.tqdm(range(xx.shape[0]), "Iterating over rows"):
            for j in range(xx.shape[1]):
                u = xx[i, j] * pc1 + yy[i, j] * pc2
                Z[i, j] = fn(u)

        # Plot the density map
        Z = (Z - Z.min()) / (Z.max() - Z.min()) # rescale
        Z = np.log(1e-3 + Z)
        
        im = ax.imshow(Z, interpolation='bilinear', origin='lower',
            extent=(*pc1_lim, *pc2_lim), cmap='Blues', alpha=1., aspect='auto')
        
        return Z

    def weights_to_model(weights):
        m = deepcopy(model)
        m.to(device)

        i = 0
        for n, p in m.named_parameters():
            p.data += torch.from_numpy(weights[i:i+p.numel()]).view(p.shape).to(device)
            i += p.numel()
        
        return m


    def weights_to_loss(weights):
        m = weights_to_model(weights)
        yhats = m(xs, ys)
        return F.mse_loss(yhats, ys).item()

    xs.to(device)
    ys.to(device)

    pc_combos = list(itertools.combinations(range(num_components), 2))

    fig, axes = plt.subplots(1, len(pc_combos) + 1, figsize=(20, 5))

    for ax,  (pc1_idx, pc2_idx) in zip(axes, pc_combos):
        pc1 = pca.components_[pc1_idx]
        pc2 = pca.components_[pc2_idx]

        min_pc1, max_pc1 = weights_reduced[:, pc1_idx].min(), weights_reduced[:, pc1_idx].max()
        min_pc2, max_pc2 = weights_reduced[:, pc2_idx].min(), weights_reduced[:, pc2_idx].max()

        pc1_lims = (min_pc1 - 0.1 * (max_pc1 - min_pc1), max_pc1 + 0.1 * (max_pc1 - min_pc1))
        pc2_lims = (min_pc2 - 0.1 * (max_pc2 - min_pc2), max_pc2 + 0.1 * (max_pc2 - min_pc2))

        get_pc_landscape(pca, weights_to_loss, pc1, pc2, pc1_lims, pc2_lims, num_points=num_points, ax=ax)

        for chain in range(num_chains):
            # _weights = pca.transform(deltas[chain])
            _weights = weights_reduced[chain * num_draws:(chain + 1) * num_draws] 
            sns.scatterplot(x=_weights[:, pc1_idx], y=_weights[:, pc2_idx], ax=ax)

        ax.set_xlim(*pc1_lims)
        ax.set_ylim(*pc2_lims)

        ax.set_title(f"PC {pc1_idx + 1} vs PC {pc2_idx + 1}")
        ax.set_xlabel(f"PC {pc1_idx + 1}")
        ax.set_ylabel(f"PC {pc2_idx + 1}")

    # Plot explained variance
    plot_explained_variance(pca, title="Explained Variance", ax=axes[-1])        

    return fig