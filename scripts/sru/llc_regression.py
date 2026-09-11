#!/usr/bin/env python3
"""LLC over the checkpoints of a regression run trained by train_regression.py, with the paper's sampler.

    python scripts/sru/llc_regression.py --sweep sweeps/regression/training-runs/L2H4Minf.yaml --seed 0 \
        --root /work/results --out /work/results [--steps 0,984,...] [--chains 10 --draws 4000 --burnin 1000]

Rebuilds the run config exactly as training did (same sweep, seed and overrides, so the checkpoint directory hash
matches), restores each checkpoint through the LocalStorageProvider, and runs `icl.analysis.sample.Sampler` on it
with the Table A.9 settings: 10 chains x 5000 SGLD steps (1000 burn-in, one draw per step), eps 3e-4, n*beta 66.7,
batch 1024, a fixed SGLD dataset of 2^20 sequences, seed reset before every checkpoint. gamma is passed to the
sampler as-is (see the unit spec for the gamma/2 convention question). Results are the sampler's flattened output
(the same keys as analysis/L2H4Minf/llcs.pt in the paper repo), one row per (seed, step, key), plus a wide table
of the per-chain LLCs. Tables are rewritten after every checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from train_regression import steps_spec, sweep_to_kwargs


def run_chain_workers(a) -> None:
    """Launch --chain-workers copies of this script, one chain each (chain ids 0..N-1, init seeds init_seed + id), then
    merge their tables: per-chain rows are concatenated and the pooled LLC per step is the mean over chains, its std the
    population std over chains, matching the sampler's own pooling."""
    import subprocess
    import sys
    out = Path(a.out)
    cmd_base = [sys.executable, __file__, "--sweep", a.sweep, "--seed", str(a.seed), "--root", a.root, "--max-examples", str(a.max_examples or ""),
                "--checkpoint-steps", a.checkpoint_steps, "--chains", "1", "--draws", str(a.draws), "--burnin", str(a.burnin),
                "--epsilon", str(a.epsilon), "--gamma", str(a.gamma), "--nbeta", str(a.nbeta), "--batch", str(a.batch),
                "--dataset-size", str(a.dataset_size), "--init-seed", str(a.init_seed)]
    if a.steps:
        cmd_base += ["--steps", a.steps]
    if a.train_steps:
        cmd_base += ["--train-steps", str(a.train_steps)]
    if a.init_loss_batches:
        cmd_base += ["--init-loss-batches", str(a.init_loss_batches)]
    if not a.max_examples:
        i = cmd_base.index("--max-examples"); del cmd_base[i:i + 2]
    procs = []
    for c in range(a.chain_workers):
        wdir = out / "chains" / f"seed{a.seed}-chain{c}"; wdir.mkdir(parents=True, exist_ok=True)
        log = open(wdir / "stdout.log", "w")
        procs.append((c, subprocess.Popen(cmd_base + ["--out", str(wdir), "--chain-id", str(c)], stdout=log, stderr=subprocess.STDOUT)))
    rcs = {c: p.wait() for c, p in procs}
    bad = {c: rc for c, rc in rcs.items() if rc != 0}
    wide = pd.concat(pd.read_parquet(out / "chains" / f"seed{a.seed}-chain{c}" / "data" / f"llc-seed{a.seed}.parquet") for c in range(a.chain_workers) if (out / "chains" / f"seed{a.seed}-chain{c}" / "data" / f"llc-seed{a.seed}.parquet").exists())
    pooled = wide.groupby("step").llc.agg(["mean", lambda x: x.std(ddof=0)]); pooled.columns = ["llc_pooled", "llc_pooled_std"]
    wide = wide.drop(columns=["llc_pooled", "llc_pooled_std"]).merge(pooled, left_on="step", right_index=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    wide.sort_values(["step", "chain"]).to_parquet(out / "data" / f"llc-seed{a.seed}.parquet", index=False)
    long = pd.concat(pd.read_parquet(out / "chains" / f"seed{a.seed}-chain{c}" / "data" / f"llc_long-seed{a.seed}.parquet") for c in range(a.chain_workers) if (out / "chains" / f"seed{a.seed}-chain{c}" / "data" / f"llc_long-seed{a.seed}.parquet").exists())
    long.to_parquet(out / "data" / f"llc_long-seed{a.seed}.parquet", index=False)
    for step, g in pooled.iterrows():
        print(f"seed {a.seed} step {int(step)}: llc {g.llc_pooled:.2f} (sd {g.llc_pooled_std:.2f}) over {int((wide.step == step).sum())} chains", flush=True)
    if bad:
        raise SystemExit(f"chain workers failed: {bad}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--root", required=True, help="the training run's --out (LOCAL_ROOT holding checkpoints/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", default="", help="comma-separated checkpoint steps; default all saved checkpoints")
    ap.add_argument("--train-steps", type=int, default=None, help="num_steps override used in training (pilots)")
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--checkpoint-steps", default="log_space=100,linear_space=100")
    ap.add_argument("--chains", type=int, default=10)
    ap.add_argument("--draws", type=int, default=4000)
    ap.add_argument("--burnin", type=int, default=1000)
    ap.add_argument("--epsilon", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=13.3)
    ap.add_argument("--nbeta", type=float, default=66.7)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--dataset-size", type=int, default=2 ** 20)
    ap.add_argument("--init-loss-batches", type=int, default=None, help="cap init-loss evaluation batches (pilots)")
    ap.add_argument("--init-seed", type=int, default=42)
    ap.add_argument("--chain-workers", type=int, default=1, help="run the chains as this many concurrent single-chain processes")
    ap.add_argument("--chain-id", type=int, default=None, help="(internal) this process samples one chain with this id")
    a = ap.parse_args()
    if a.chain_workers > 1:
        return run_chain_workers(a)

    os.environ.setdefault("AWS_REGRESSION_BUCKET_NAME", "")
    os.environ["LOCAL_ROOT"] = a.root
    import torch
    from icl.analysis.sample import SamplerConfig
    from icl.constants import DEVICE
    from icl.regression.config import get_config
    from icl.regression.experiments.utils import flatten_and_process
    from icl.regression.train import RegressionRun

    sweep = yaml.safe_load(open(a.sweep))
    kw = sweep_to_kwargs(sweep["parameters"], a.seed)
    if a.train_steps:
        kw["num_steps"] = a.train_steps
    if a.max_examples:
        kw["task_config"]["max_examples"] = a.max_examples
    kw["checkpointer_config"] = {"checkpoint_steps": steps_spec(a.checkpoint_steps), "bucket_name": None, "local_root": a.root}
    kw["logger_config"] = {"logging_steps": [], "project": None, "entity": None}
    cfg = get_config(project=None, entity=None, **kw)
    run = RegressionRun(cfg)
    saved = sorted(run.checkpointer.file_ids)
    steps = [int(s) for s in a.steps.split(",")] if a.steps else saved
    missing = [s for s in steps if s not in saved]
    if missing:
        raise SystemExit(f"checkpoints not found for steps {missing[:10]} (have {len(saved)})")

    temperature = a.dataset_size / a.nbeta  # the sampler's temperature T satisfies n*beta = dataset_size / T
    sampler_cfg = dict(num_chains=a.chains, num_draws=a.draws, num_burnin_steps=a.burnin, sampling_method="sgld",
                       grad_batch_origin="eval-dataset", grad_batch_size=a.batch, epsilon=a.epsilon, gamma=a.gamma,
                       temperature=temperature, eval_method="grad-minibatch", eval_batch_size=a.batch,
                       eval_dataset_size=a.dataset_size, eval_metrics=["likelihood-derived", "batch-loss"],
                       eval_online=False, eval_loss_fn="mse", init_seed=a.init_seed + (a.chain_id or 0),
                       num_init_loss_batches=a.init_loss_batches, device=str(DEVICE), cores=1)
    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    meta = {"seed": a.seed, "run_name": cfg.run_name, "steps": steps, "sampler": {k: v for k, v in sampler_cfg.items()},
            "nbeta": a.nbeta, "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__, "started": time.strftime("%FT%TZ", time.gmtime())}
    (out / f"llc-meta-seed{a.seed}.json").write_text(json.dumps(meta, indent=1, default=str))

    long_rows, wide_rows = [], []
    for i, step in enumerate(steps):
        t0 = time.time()
        run.model.load_state_dict(run.checkpointer.load_file(step)["model"])
        run.model.to(DEVICE).train()
        sampler = SamplerConfig(**sampler_cfg).to_sampler(run)  # re-seeds with init_seed: same batches and noise per checkpoint
        results = sampler.eval(run.model)
        results["loss/init"] = sampler.init_loss.item()
        flat = flatten_and_process(results)
        for k, v in flat.items():
            if isinstance(v, (int, float, np.floating, np.integer)):
                long_rows.append({"seed": a.seed, "step": step, "chain_id": a.chain_id, "key": k, "value": float(v)})
        # per-chain LLC from the chain's mean mini-batch loss: nbeta (mean_loss_c - init_loss); the pooled llc/mean is the
        # same expression on the chain average, so mean over chains of the per-chain values equals llc/mean
        init = float(flat["loss/init"])
        for k, v in flat.items():
            if k.startswith("batch-loss/mean/") and k.split("/")[-1].isdigit():
                c = int(k.split("/")[-1]) + (a.chain_id or 0)
                wide_rows.append({"seed": a.seed, "step": step, "chain": c, "chain_mean_loss": float(v), "llc": a.nbeta * (float(v) - init),
                                  "llc_pooled": float(flat["llc/mean"]), "llc_pooled_std": float(flat["llc/std"]), "init_loss": init,
                                  "nbeta": a.nbeta, "epsilon": a.epsilon, "gamma": a.gamma, "chains": a.chains, "draws": a.draws,
                                  "burnin": a.burnin, "batch": a.batch, "dataset_size": a.dataset_size, "seconds": time.time() - t0})
        pd.DataFrame(long_rows).to_parquet(out / "data" / f"llc_long-seed{a.seed}.parquet", index=False)
        pd.DataFrame(wide_rows).to_parquet(out / "data" / f"llc-seed{a.seed}.parquet", index=False)
        print(f"[{i + 1}/{len(steps)}] seed {a.seed} step {step}: llc {flat['llc/mean']:.2f} "
              f"(sd {flat['llc/std']:.2f}, init loss {init:.4f}) {time.time() - t0:.0f}s", flush=True)
    meta["finished"] = time.strftime("%FT%TZ", time.gmtime())
    (out / f"llc-meta-seed{a.seed}.json").write_text(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    main()
