import os
import pickle
import warnings
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
import tqdm
import typer
from devinfra.io.storage import (  # Import the base class and IDType
    BaseStorageProvider, IDType, S3StorageProvider, create_storage_provider,
    int_id_to_key, key_to_int_id)
from dotenv import load_dotenv
from torch import nn

from icl.analysis.evals import ICLEvaluator
from icl.analysis.utils import get_unique_run
from icl.constants import ANALYSIS, DATA, FIGURES, SWEEPS
from icl.figures.plotting import *
from icl.monitoring import stdlogger
from icl.train import Run

load_dotenv()
app = typer.Typer()

LR_SWEEP_ID = "hmy71gjb"

@dataclass
class Analysis:
    run: Run
    steps: List[int]
    models: List[nn.Module]
    opt_states: List[dict]
    

def load_state_dicts(id_, runs):
    path = DATA / id_ / "models.pt"

    if os.path.exists(path):
        stdlogger.info("Retrieving state dicts from disk")
        with open(DATA / id_ / "models.pt", "rb") as f:
            all_models = torch.load(f)

        with open(DATA / id_ / "optimizer_state_dicts.pt", "rb") as f:
            all_optimizer_state_dicts = torch.load(f)

        return all_models, all_optimizer_state_dicts

    all_models = []
    all_optimizer_state_dicts = []

    for run in tqdm.tqdm(runs, desc="Retrieving state dicts from bucket"):
        models = []
        optimizer_state_dicts = []

        for checkpoint in tqdm.tqdm(run.checkpointer):
            m = deepcopy(run.model)
            m.load_state_dict(checkpoint["model"])
            models.append(m)
            optimizer_state_dicts.append(checkpoint["optimizer"])
            
        all_models.append(models)
        all_optimizer_state_dicts.append(optimizer_state_dicts)

    stdlogger.info("Writing state dicts to disk")
    with open(DATA / id_ / "models.pt", "wb") as f:
        torch.save(all_models, f)

    with open(DATA / id_ / "optimizer_state_dicts.pt", "wb") as f:
        torch.save(all_optimizer_state_dicts, f)

    return all_models, all_optimizer_state_dicts


def get_runs(name):
    return [get_unique_run(
        str(SWEEPS / f"training-runs/{name}.yaml"), 
        task_config={"model_seed": model_seed, "layer_norm": True},
    ) for model_seed in range(5)]    


def get_evals(id_, analyses, batch_size=8192, reeval=False):   
    run = analyses[0].run
    evaluator = ICLEvaluator(
        pretrain_dist=run.pretrain_dist,
        true_dist=run.true_dist,
        max_examples=run.config.task_config.max_examples,
        eval_batch_size=batch_size,
        seed=run.config.task_config.true_seed,   
    )

    if os.path.exists(ANALYSIS / id_ / "evals_over_time.pkl") and not reeval:
        stdlogger.info("Retrieving evals from disk")
        with open(ANALYSIS / id_ / "evals_over_time.pkl", "rb") as f:
            return pickle.load(f)
        
    stdlogger.info("Evaluating models")
    evals_over_time = [{**evaluator(model), "step": step, "model_seed": i} for i, analysis in enumerate(analyses) for step, model in zip(analysis.steps, tqdm.tqdm(analysis.models))]
    df = pd.DataFrame(evals_over_time)

    save_evals(df, id_)
    return df


def save_evals(df, id_):
    stdlogger.info("Writing to disk")
    with open(ANALYSIS / f"{id_}/evals_over_time.pkl", "rb") as f:
        pickle.dump(df, f)


def get_llcs(sweep_id):
    import wandb

    api = wandb.Api()
    sweep = api.sweep(f"devinterp/icl/{sweep_id}")
    wandb_runs = sweep.runs

    df = None

    for llc_run in tqdm.tqdm(wandb_runs):
        history_df = llc_run.history()

        llc_mean_columns = [f'llc/mean/{i}' for i in range(8)]
        history_df[llc_mean_columns] = history_df[llc_mean_columns].replace("NaN", np.nan)

        llc_std_columns = [f'llc/std/{i}' for i in range(8)]
        history_df[llc_std_columns] = history_df[llc_std_columns].replace("NaN", np.nan)

        if df is None:
            df = history_df
        else:
            df = df.concatenate(history_df)

    return df


def merge_dfs(df1, df2, inplace=True):
    if not inplace:
        raise NotImplementedError()
    
    seeds = df1["model_seed"].unique()
    steps = df1["step"].unique()

    stdlogger.info(f"Merging {len(seeds)} seeds and {len(steps)} steps")
    for seed in seeds:
        for step in steps:
            for k in df2.columns:
                if k not in df1.columns:
                    df1.loc[(df1["model_seed"] == seed) & (df1["step"] == step), k] = df2.loc[(df2["model_seed"] == seed) & (df2["step"] == step), k]

    return df1

@app.command()
def make_figures(name: str, seed: int=1, llc_sweep_id: str=LR_SWEEP_ID, val_size=1024, reeval=False):
    for d in [ANALYSIS / name, DATA / name, FIGURES / name]:
        if not os.path.exists(d):
            os.makedirs(d)

    stdlogger.info("Retrieving runs")
    runs = get_runs(name)

    stdlogger.info("Retrieving models and optimizer states")
    all_models, all_optimizer_state_dicts = load_state_dicts(name, runs)

    steps = runs[0].checkpointer.file_ids
    for run in runs:
        # Check if different runs have different steps
        _steps = run.checkpointer.file_ids
        
        if set(_steps) != set(steps):
            warnings.warn("Found different checkpoints for different runs")
            steps = [s for s in steps if s in _steps]
    
    analyses = [Analysis(run, steps, models, opt_states) for run, models, opt_states in zip(runs, all_models, all_optimizer_state_dicts)]

    df = get_evals(name, analyses, batch_size=val_size, reeval=reeval)
    return 
    if 'llc/mean/mean' not in df.columns:
        stdlogger.info("Retrieving LLCs")
        llcs_df = get_llcs(llc_sweep_id)
        df = merge_dfs(df, llcs_df)
        save_evals(df, name)

    stdlogger.info("Analysis for seed %s", seed)


if __name__ == "__main__":
    app()

