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
    a = ap.parse_args()

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
                       eval_online=False, eval_loss_fn="mse", init_seed=a.init_seed,
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
                long_rows.append({"seed": a.seed, "step": step, "key": k, "value": float(v)})
        # per-chain LLC from the chain's mean mini-batch loss: nbeta (mean_loss_c - init_loss); the pooled llc/mean is the
        # same expression on the chain average, so mean over chains of the per-chain values equals llc/mean
        init = float(flat["loss/init"])
        for k, v in flat.items():
            if k.startswith("batch-loss/mean/") and k.split("/")[-1].isdigit():
                c = int(k.split("/")[-1])
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
