"""
Based on Michaud et al. (2023)'s multitask sparse parity dataset.

"""

import copy
import itertools
import os
from copy import deepcopy
from math import isnan
from pprint import pp
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
import seaborn as sns
import torch
from devinterp.slt.forms import get_osculating_circle
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
from scipy.stats import zipf
from skimage.measure import EllipseModel
from sklearn.decomposition import PCA
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm, trange

import wandb
from icl.analysis.smoothing import gaussian_filter1d_variable_sigma
from icl.constants import DATA, FIGURES
from infra.utils.device import move_to_


class MultitaskSparseParity(IterableDataset):
    def __init__(
        self,
        n: int,
        k: int,
        num_tasks: int,
        alpha: float = 0.4,
    ):
        """
        Initialize a multitask sparse parity dataset.

        Args:
            n: The length of the task bits.
            k: The size of the subset of bits used to compute the parity.
            num_tasks: The number of tasks.
            alpha: The parameter for the Zipfian distribution.
            replace: Whether to sample tasks with replacement. I.e., whether any task bits can be reused. 
        """
        super().__init__()
        self.n = n
        self.k = k
        self.num_tasks = num_tasks
        self.alpha = alpha

        # Choose subsets for tasks
        self.tasks = [torch.randperm(n)[:k] for _ in range(num_tasks)]

        # Set up the Zipfian distribution
        task_weights = 1 / torch.arange(1, num_tasks + 1) ** alpha
        self.task_frequencies = task_weights / task_weights.sum()

    def _generate_sample(self, task_idx: Optional[int] = None, merge_input=True) -> Union[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate a single sample for the dataset.

        Returns:
            A tuple containing the control bits, the task bits, and the parity.
        """
        # Select a task
        if task_idx is None:
            task_idx = torch.multinomial(self.task_frequencies, 1).item()

        task = self.tasks[task_idx]

        # Create the control bits
        control_bits = torch.zeros(self.num_tasks, dtype=torch.float)
        control_bits[task_idx] = 1

        # Create the task bits
        # TODO: allow different distribution over task bits?
        task_bits = torch.randint(0, 2, (self.n,), dtype=torch.float)

        # Compute the parity
        parity = task_bits[task].sum() % 2

        if merge_input:
            return (torch.cat([control_bits, task_bits]), parity.long())

        return (control_bits, task_bits), parity.long()
    
    def generate_batch(self, batch_size: int, task_idx: Optional[int] = None, merge_input=True) -> Union[Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate a batch of samples.

        Args:
            batch_size: The number of samples in the batch.

        Returns:
            A tuple containing the batched control bits, task bits, and parities.
        """
        control_bits_batch = []
        task_bits_batch = []
        parities_batch = []

        for _ in range(batch_size):
            (control_bits, task_bits), parity = self._generate_sample(task_idx=task_idx, merge_input=False)
            control_bits_batch.append(control_bits)
            task_bits_batch.append(task_bits)
            parities_batch.append(parity)

        control_bits_batch = torch.stack(control_bits_batch)
        task_bits_batch = torch.stack(task_bits_batch)
        parities_batch = torch.stack(parities_batch)

        if merge_input:
            return (torch.cat([control_bits_batch, task_bits_batch], dim=1), parities_batch)

        return (control_bits_batch, task_bits_batch), parities_batch

    def __iter__(self):
        while True:
            yield self._generate_sample()

    def __str__(self) -> str:
        return f"MultitaskSparseParity(n={self.n}, k={self.k}, num_tasks={self.num_tasks}, alpha={self.alpha})"

    def __repr__(self) -> str:
        return self.__str__()
    

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def to_color_string(color):
    # return (256 * color[0], 256 * color[1], 256 * color[2], color[3])
    return f"rgb({int(256 * color[0])}, {int(256 * color[1])}, {int(256 * color[2])}, {color[3]})"


def plot_ed(pca, reduced, reduced_smooth, task_references_reduced, model_id, form_cmap='rainbow', evolute_cmap='Spectral', num_components=3, title="", slug="pca.html"):
    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    subplot_titles = []
    fig = make_subplots(rows=num_components, cols=num_components, subplot_titles=subplot_titles)

    if isinstance(form_cmap, str):
        form_cmap = sns.color_palette(form_cmap, as_cmap=True)
    if isinstance(evolute_cmap, str):
        evolute_cmap = sns.color_palette(evolute_cmap, as_cmap=True)

    # colors = np.array([to_color_string(form_cmap(c)) for c in np.linspace(0, 1, reduced.shape[0])])   
    colors = np.array([to_color_string(form_cmap(c)) for c in np.linspace(0, 1, task_references_reduced.shape[0])])   
    evolute_colors = np.array([to_color_string(evolute_cmap(c)) for c in np.linspace(0, 1, len(reduced_smooth)-4)])

    for i, j in tqdm(itertools.product(range(num_components), range(num_components)), total=num_components ** 2): 
        row, col = i + 1, j + 1
            
        ymin, ymax = (
            reduced[:, i].min(),
            reduced[:, i].max(),
        )
        xmin, xmax = (
            reduced[:, j].min(),
            reduced[:, j].max(),
        )

        ts = np.array(range(2, len(reduced_smooth) - 2))
        centers = np.zeros((len(ts), 2))

        # Circles
        # for ti, t in enumerate(ts):
        #     center, radius = get_osculating_circle(
        #         reduced_smooth[:, (j, i)], t
        #     )
        #     if ti % 3 == 0:
        #         # This seems to be cheaper than directly plotting a circle
        #         circle = go.Scatter(
        #             x=center[0] + radius * np.cos(np.linspace(0, 2 * np.pi, 100)),
        #             y=center[1] + radius * np.sin(np.linspace(0, 2 * np.pi, 100)),
        #             mode="lines",
        #             line=dict(color="rgba(0.1, 0.1, 1, 0.05)", width=1),
        #             showlegend=False,
        #         )
        #         fig.add_trace(circle, row=row, col=col)

        #     centers[ti] = center

        # Centers
        fig.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode="markers",
                marker=dict(size=4, symbol="x", color=evolute_colors),
                name="Centers",
            ),
            row=row,
            col=col,
        )

        # Original samples
        # fig.add_trace(
        #     go.Scatter(
        #         x=reduced[:, j],
        #         y=reduced[:, i],
        #         mode="markers",
        #         marker=dict(color=colors, size=3),
        #         showlegend=False,
        #     ),
        #     row=row,
        #     col=col,
        # )

        # Smoothed trajectory
        fig.add_trace(
            go.Scatter(
                x=reduced_smooth[:, j],
                y=reduced_smooth[:, i],
                mode="lines",
                line=dict(color="black", width=2),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Task references
        fig.add_trace(
            go.Scatter(
                x=task_references_reduced[:, j],
                y=task_references_reduced[:, i],
                mode="markers",
                marker=dict(color=colors, size=10),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        if j == 0:
            fig.update_yaxes(title_text=labels[str(i)], row=row, col=col)

        fig.update_xaxes(title_text=labels[str(j)], row=row, col=col)

        fig.update_xaxes(
            range=(xmin * 1.25, xmax * 1.25),
            row=row,
            col=col,
        )
        fig.update_yaxes(
            range=(ymin * 1.25, ymax * 1.25),
            row=row,
            col=col,
        )

    fig.update_layout(width=2000, height=2000)  # Adjust the size as needed
    fig.update_layout(title_text=title, showlegend=False)

    # Save as html
    pyo.plot(fig, filename=str(FIGURES / model_id / f"{slug}.html"), auto_open=False)
    fig.write_image(str(FIGURES / model_id / f"{slug}.png"))

    return fig


def train(config): 
    config.setdefault("eval_batch_size", 256)
    config.setdefault("batch_size", 1024)
    config.setdefault("num_steps", 1_000)
    config.setdefault("num_tasks", 8)
    config.setdefault("num_features", 16)
    config.setdefault("num_task_bits", 3)
    config.setdefault("hidden_dim", 50)
    config.setdefault("alpha", 0.5)
    config.setdefault("num_checkpoints", min(2_500, config["num_steps"]))
    config.setdefault("lr", 0.003)
    config.setdefault("seed", 0)
    config.setdefault("num_components", 5)
    config.setdefault("init_smoothing", 0.1)
    config.setdefault("final_smoothing", 40.0)
    config.setdefault("log_wandb", True)

    # Read the values from the updated config dictionary
    eval_batch_size = config["eval_batch_size"]
    batch_size = config["batch_size"]
    num_steps = config["num_steps"]
    num_tasks = config["num_tasks"]
    num_features = config["num_features"]
    num_task_bits = config["num_task_bits"]
    num_bits = num_tasks + num_features
    hidden_dim = config["hidden_dim"]
    alpha = config["alpha"]
    num_checkpoints = config["num_checkpoints"]
    lr = config["lr"]
    seed = config["seed"]
    num_components = config["num_components"]
    init_smoothing = config["init_smoothing"]
    final_smoothing = config["final_smoothing"]
    log_wandb = config["log_wandb"]

    model_id = f"multitask-parity-{num_tasks}-{num_features}-{num_task_bits}-{hidden_dim}-{alpha}-{num_steps}-{lr}-{seed}"

    torch.manual_seed(seed)

    dataset = MultitaskSparseParity(n=num_features, k=num_task_bits, num_tasks=num_tasks, alpha=alpha)

    eval_sets = [dataset.generate_batch(eval_batch_size, task_idx=t) for t in trange(num_tasks, desc="Generating eval sets")]

    log_interval = list(np.linspace(0, num_steps, num_checkpoints).astype(int))
    model = MLP(input_dim=num_bits, hidden_dim=hidden_dim, output_dim=2)

    checkpoints = []

    if log_wandb:
        for k, v in config.items():
            wandb.config[k] = v
        wandb.run.name = model_id
        wandb.watch(model)

    fig_folder = FIGURES / "multitask-parity" / model_id

    if not (FIGURES / "multitask-parity" / model_id).exists():
        (FIGURES / "multitask-parity" / model_id).mkdir(parents=True)

    if not (DATA / "multitask-parity" / model_id).exists():
        (DATA / "multitask-parity" / model_id).mkdir(parents=True)

    model.to('mps')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    dataloader = DataLoader(dataset, batch_size=batch_size)

    def accuracy_score(labels, predictions):
        return (labels == predictions).sum() / len(labels)

    def eval_model(model, eval_sets):
        accuracies = []
        losses = []

        for task_idx, (inputs, labels) in enumerate(eval_sets):
            inputs = inputs.to('mps')
            labels = labels.to('mps')
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            _, predictions = torch.max(outputs, 1)
            accuracies.append(accuracy_score(labels, predictions).item())

        accuracies = np.array(accuracies)
        losses = np.array(losses)
        task_freqs = dataset.task_frequencies.detach().cpu().numpy()

        results = {
            "Loss": (losses @ task_freqs),
            "Accuracy": (accuracies @ task_freqs),
        }

        for task_idx in range(num_tasks):
            results[f"Loss/{task_idx}"] = losses[task_idx]
            results[f"Accuracy/{task_idx}"] = accuracies[task_idx]

        return results

    def log_to_wandb(step=None):
        with torch.no_grad():
            results = eval_model(model, eval_sets)
            wandb.log(results, step=step)

    step = 0
    log_to_wandb()

    for i, (inputs, labels) in enumerate(tqdm(dataloader, total=num_steps, desc="Training")):
        if step > num_steps:
            break

        inputs = inputs.to('mps')
        labels = labels.to('mps')

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (step in log_interval) and log_wandb:
            log_to_wandb(step=step)
            checkpoint = {
                "step": step,
                "model": deepcopy(model.state_dict()),
                "optimizer": deepcopy(optimizer.state_dict()),
                "rng_state": torch.get_rng_state(),
            }
            move_to_(checkpoint, "cpu")
            checkpoints.append(checkpoint)

        step += 1


    print("Saving model")
    torch.save(checkpoints, str(DATA / "multitask-parity" / model_id / "checkpoints.pt"))

    models = [c["model"] for c in checkpoints]
    # PCs       
        
    ed_outputs = []
    eval_inputs = torch.cat([x for x, _ in eval_sets], dim=0).to('mps')
    eval_labels = torch.cat([y for _, y in eval_sets], dim=0).to('mps')

    losses = []

    eval_criterion = nn.CrossEntropyLoss(reduction='none')

    for state_dict in tqdm(models, desc="Evaluating models"):
        model.load_state_dict(state_dict)
        preds = model(eval_inputs)
        loss = eval_criterion(preds, eval_labels)
        losses.append(loss.mean().item())
        ed_outputs.append(loss.cpu().detach().numpy().flatten())
        # ed_outputs.append(preds.cpu().detach().numpy().flatten())

    ed_outputs_np = np.stack(ed_outputs)
    pca = PCA(n_components=num_components)

    ed_projections = pca.fit_transform(ed_outputs_np)
    ed_projections_smooth = gaussian_filter1d_variable_sigma(ed_projections, sigma=np.linspace(init_smoothing, final_smoothing, ed_projections.shape[0]), axis=0)

    fig = make_subplots(rows=1, cols=1+num_components, subplot_titles=["Loss"] + [f"Component {i} ({pca.explained_variance_ratio_[i-1]:.2f})" for i in range(1, num_components+1)])

    fig.add_trace(go.Scatter(x=log_interval, y=losses, mode="lines", name="Loss", showlegend=False), row=1, col=1)

    for i in range(1, num_components+1):
        fig.add_trace(go.Scatter(x=log_interval, y=ed_projections[:, i-1], mode="markers", marker=dict(color='black', opacity=0.1), name=f"Component {i}", showlegend=False), row=1, col=i+1)
        fig.add_trace(go.Scatter(x=log_interval, y=ed_projections_smooth[:, i-1], mode="lines", line=dict(color='red', width=4), name=f"Component {i} (Smooth)", showlegend=False), row=1, col=i+1)
        fig.update_xaxes(title_text="Step", row=1, col=i+1)

    fig.update_layout(height=500, width=2000, title_text="PCA Components")
    # fig.show()

    wandb.log({"PCA/Components": fig}, step=step, commit=True)

    # Potentials

    task_references = []

    for t in range(num_tasks):
        task_reference = np.ones(eval_batch_size * num_tasks) * -np.log(0.5)
        task_reference[0:(t+1)* eval_batch_size] = 0
        task_references.append(task_reference)
        
    task_references = np.stack(task_references)
    task_references_reduced = pca.transform(task_references)
    
    form_potentials = np.zeros((num_tasks, len(log_interval)))

    for t in range(num_tasks):
        form_potentials[t, :] = np.sum((ed_outputs_np - task_references[t, :]) ** 2, axis=1)

    fig = make_subplots(rows=1, cols=num_tasks, subplot_titles=[f"Task {i}" for i in range(1, num_tasks+1)])

    for i in range(num_tasks):
        fig.add_trace(go.Scatter(x=log_interval, y=form_potentials[i], mode="lines", name=f"Task {i+1}", showlegend=False), row=1, col=i+1)

    fig.update_layout(height=500, width=2000, title_text="Reference Value Potentials")
    # fig.show()

    wandb.log({"PCA/Form Potentials": fig}, commit=True)

    fig = plot_ed(pca, ed_projections, ed_projections_smooth, task_references_reduced, f"multitask-parity/{model_id}", title="Multitask Parity PCA", slug="pca", num_components=5)

    # table = wandb.Table(columns=["figure"])
    # table.add_data(wandb.Html(str(FIGURES / "multitask-parity" / "pca.html")))
    wandb.log({"PCA/Essential Dynamics": fig}, commit=True)

    return models



if __name__ == "__main__":
    wandb.init(entity='devinterp', project='multitask-parity')
    models = train(dict(wandb.config))