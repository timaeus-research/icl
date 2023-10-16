
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
from devinfra.utils.device import get_default_device
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from torch.nn import functional as F

import wandb
from icl.analysis.sample import sample
from icl.config import get_config
from icl.experiments.utils import *
from icl.train import Run
from icl.utils import pyvar_dict_to_latex, pyvar_dict_to_slug

app = typer.Typer()


class ObservedOnlineLLCEstimator:
    def __init__(self, num_chains: int, num_draws: int, n: int, device="cpu", threshold=0.05):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.llcs = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        
        # If a chain has a loss < init_loss for more than `threshold` draws, do not include it in the final estimate.
        self.threshold = threshold
        self.num_draws_in_chain_below_init = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.chain_below_threshold = torch.zeros(num_chains, dtype=torch.bool).to(device)
        self.thresholded_llcs = torch.zeros(num_draws, dtype=torch.float32).to(device)

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss 

        if draw == 0:  # TODO: We can probably drop this and it still works (but harder to read)
            self.llcs[0, draw] = 0.
        else:
            t = draw + 1
            prev_llc = self.llcs[chain, draw - 1]

            with torch.no_grad():
                self.llcs[chain, draw] = (1 / t) * (
                    (t - 1) * prev_llc + (self.n / self.n.log()) * (loss - self.init_loss)
                )

        if self.threshold:
            if loss < self.init_loss:
                self.num_draws_in_chain_below_init[chain] += 1

            if (self.num_draws_in_chain_below_init[chain] / draw) > self.threshold:
                self.chain_below_threshold[chain] = 0.
            else:
                self.chain_below_threshold[chain] = 1.


        # Assumes this is run serially
        if chain == self.num_chains - 1:
            thresholded_llcs = self.llcs[self.chain_below_threshold, draw]
            self.thresholded_llcs[draw] = thresholded_llcs.mean()

            wandb.log({
                "llc/mean": self.llcs[:, draw].mean().item(), 
                "llc/std": self.llcs[:, draw].std().item(),
                "llc/max": self.llcs[:, draw].max().item(),
                "llc/min": self.llcs[:, draw].min().item(),
                "thresholded-llc/mean": thresholded_llcs.mean().item(),
                "thresholded-llc/std": thresholded_llcs.std().item(),
                "thresholded-llc/max": thresholded_llcs.max().item(),
                "thresholded-llc/min": thresholded_llcs.min().item(),
                **{
                    f"chain-llcs/{i}": self.llcs[i, draw].item() for i in range(self.num_chains)
                },    
            }, step=draw)

    @property
    def init_loss(self):
        return self.losses[0, 0]

    def sample(self):
        return {
            # "llc/mean": self.llcs[:, -1].mean().cpu().numpy(),
            # "llc/std": self.llcs[:, -1].std().cpu().numpy(),
            # "llc/thresholded-mean": self.llcs[self.chain_below_threshold, :].mean(axis=0).cpu().numpy(),
            "llc/means": self.llcs.mean(axis=0).cpu().numpy(),
            "llc/stds": self.llcs.std(axis=0).cpu().numpy(),
            "llc/trace": self.llcs.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy(),

        }
    
    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)


def estimate_llc_at_end(
    config: dict,
    sampler_config: dict,
):      
    cores = int(os.environ.get("CORES", 1))
    config = get_config(**config)

    print("Loaded configs")

    device = get_default_device()
    num_layers = config.task_config.num_layers
    num_heads = config.task_config.num_heads
    num_tasks = config.task_config.num_tasks

    print("\n")
    print("-" * 30 + f" M={num_tasks} " + "-" * 30)
    # Retrieve the last available checkpoint from AWS
    run = Run.create_and_restore(config)

    xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
    dataset = torch.utils.data.TensorDataset(xs, ys)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset)) 
    
    num_chains = sampler_config.pop("num_chains", 25)
    num_draws = sampler_config.pop("num_draws", 1000)

    llc_estimator = ObservedOnlineLLCEstimator(num_chains, num_draws, len(dataset), device=device)
    callbacks=[llc_estimator]

    sampling_method = sampler_config.pop("sampling_method", "sgld")

    if sampling_method == "sgld":
        optimizer_class = SGLD
    elif sampling_method == "sgnht":
        optimizer_class = SGNHT
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    sample(
        run.model,
        loader,
        F.mse_loss,
        optimizer_class,
        optimizer_kwargs=dict(
            **sampler_config,
            num_samples=len(dataset),
        ),
        num_draws=num_draws,
        num_chains=num_chains,
        cores=cores,
        device=device,
        callbacks=callbacks,
    )

    llcs = llc_estimator.sample()
    trace = llcs["loss/trace"]
    llcs_over_time_mean = llcs["llc/means"]
    llcs_over_time_std = llcs["llc/stds"]

    fig = plt.figure()
    cmap = plt.cm.viridis

    for chain in range(num_chains):
        data = trace[chain, :]
        color = cmap(chain / num_chains)
        sns.lineplot(x=np.arange(num_draws), y=data, color=color, alpha=0.5, label=f"_Chain {chain}")
    
    # Horizontal line at the initial loss
    init_loss = trace[0, 0]
    plt.axhline(y=init_loss, color="k", linestyle="--")

    plt.xlabel("num_steps")
    plt.ylabel("$L_n(w_t)$")

    title_vars = pyvar_dict_to_latex({
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_tasks": num_tasks,
        "num_draws": num_draws,
        **sampler_config,
    })
    slug = pyvar_dict_to_slug({
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_tasks": num_tasks,
        "num_draws": num_draws,
        **sampler_config,
    })

    plt.title(f"LLC trace ({title_vars})")
    
    # Add extra axis to plot for the llcs_over_time
    ax2 = plt.twinx()
    ax2.plot(np.arange(num_draws), llcs_over_time_mean, color="r", alpha=0.5, label=r"$\hat\lambda$")
    ax2.fill_between(np.arange(num_draws), llcs_over_time_mean - llcs_over_time_std, 
                        llcs_over_time_mean + llcs_over_time_std, color="r", alpha=0.15)

    ax2.set_ylabel(r"$\hat\lambda$")
    ax2.legend()
    plt.savefig(f"figures/llc-trace-{slug}.png")
    plt.close()


@app.command("wandb")
def llc_sweep_with_wandb():      
    wandb.init(project="icl-llc", entity="devinterp")
    # Rename run
    print("Initialized wandb")
    config = dict(wandb.config)
    sampler_config = config.pop("analysis_config")
    title_config = sampler_config.copy()
    del title_config["num_draws"]
    del title_config["num_chains"]
    wandb.run.name = f"L{config['task_config']['num_layers']}H{config['task_config']['num_heads']}M{config['task_config']['num_tasks']}:{pyvar_dict_to_slug(title_config)}"
    wandb.run.save()
    estimate_llc_at_end(config, sampler_config)
    wandb.finish()


@app.command("plot")
def plot_grid_search_results(csv_path: str, num_chains: int=25):
    """TODO: Fill in docstring."""
    # Read the DataFrame from the CSV file
    df = pd.read_csv(csv_path)

    # Get unique values for lrs, gammas, and num_tasks
    unique_lrs = df['lr'].unique()
    unique_gammas = df['gamma'].unique()
    unique_num_tasks = df['num_tasks'].unique()

    # Sort for visual consistency
    unique_lrs.sort()
    unique_gammas.sort()
    unique_num_tasks.sort()

    # Initialize colormap
    cmap = plt.cm.viridis

    # Create subplots
    fig, axes = plt.subplots(len(unique_lrs), len(unique_gammas), figsize=(15, 15))

    fig.suptitle(f"$\hat\lambda$ grid search ($n_\mathrm{{chains}}={num_chains}$)")

    # Loop through the grid
    for i, lr in enumerate(unique_lrs):
        for j, gamma in enumerate(unique_gammas):
            ax = axes[i, j]

            # Filter DataFrame for specific lr and gamma
            filtered_df = df[(df['lr'] == lr) & (df['gamma'] == gamma)]

            for num_tasks in unique_num_tasks:
                task_specific_df = filtered_df[filtered_df['num_tasks'] == num_tasks]

                # Sort by 'num_draws' for plotting
                task_specific_df = task_specific_df.sort_values('num_draws')

                # Calculate color based on log2(num_tasks)
                color = cmap(np.log2(num_tasks) / np.log2(max(unique_num_tasks)))

                # Plot using Seaborn for better aesthetics
                sns.lineplot(x='num_draws', y='lc/mean', data=task_specific_df, ax=ax, label=f'M={num_tasks}', color=color)
                ax.fill_between(task_specific_df['num_draws'], task_specific_df['lc/mean'] - task_specific_df['lc/std'], 
                                task_specific_df['lc/mean'] + task_specific_df['lc/std'], color=color, alpha=0.15)

            ax.set_title(f"$\epsilon={lr}, \gamma={gamma}$")
            ax.set_xlabel(r"$t_\mathrm{SGLD}$")
            ax.set_ylabel(r"$\hat\lambda$")

    plt.legend()
    plt.savefig("figures/llc-grid-search.png")
    plt.close()



if __name__ == "__main__":
    prepare_experiments()
    app()
    


