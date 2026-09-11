#!/usr/bin/env python3
"""Train the in-context linear regression transformer from a W&B sweep YAML without W&B, S3 or Sentry.

    python scripts/sru/train_regression.py --sweep sweeps/regression/training-runs/L2H4Minf.yaml \
        --seeds 0,1,2,3,4 --out /work/results [--steps 500000] [--max-examples 8]

Reads the sweep's `parameters` (taking `value`, or the requested seed for `model_seed`), builds the run config with
`icl.regression.config.get_config` exactly as `train.py` does, stores checkpoints under <out>/checkpoints/ through the
LocalStorageProvider, and writes <out>/data/training.parquet (seed, step, batch_loss every 100 steps) and
<out>/data/evals.parquet (seed, step, one column per evaluator metric at the logging steps) plus <out>/runs.json
with each seed's run name, checkpoint directory and config.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import yaml


def sweep_to_kwargs(params: dict, seed: int) -> dict:
    out = {}
    for k, v in params.items():
        if "parameters" in v:
            out[k] = sweep_to_kwargs(v["parameters"], seed)
        elif "value" in v:
            out[k] = v["value"]
        elif "values" in v:
            out[k] = seed if k == "model_seed" else v["values"][0]
    return out


def steps_spec(text: str) -> dict:
    return {k: int(v) for k, v in (kv.split("=") for kv in text.split(","))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override num_steps (pilots)")
    ap.add_argument("--max-examples", type=int, default=None, help="override task_config.max_examples")
    ap.add_argument("--checkpoint-steps", default="log_space=100,linear_space=100", help="the paper's 190-checkpoint grid")
    ap.add_argument("--logging-steps", default="log_space=100,linear_space=100")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AWS_REGRESSION_BUCKET_NAME", "")
    os.environ["LOCAL_ROOT"] = str(out)
    import torch
    from icl.regression.config import get_config
    from icl.regression.train import train

    sweep = yaml.safe_load(open(a.sweep))
    runs = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else {}
    for seed in [int(s) for s in a.seeds.split(",")]:
        kw = sweep_to_kwargs(sweep["parameters"], seed)
        if a.steps:
            kw["num_steps"] = a.steps
        if a.max_examples:
            kw["task_config"]["max_examples"] = a.max_examples
        kw["checkpointer_config"] = {"checkpoint_steps": steps_spec(a.checkpoint_steps), "bucket_name": None, "local_root": str(out)}
        csv = out / f"metrics-seed{seed}.csv"
        if csv.exists():
            csv.unlink()
        kw["logger_config"] = {"logging_steps": steps_spec(a.logging_steps), "project": None, "entity": None, "out_file": str(csv)}
        cfg = get_config(project=None, entity=None, **kw)
        n_params = sum(p.numel() for p in cfg.task_config.model_factory().parameters())
        print(f"seed {seed}: run {cfg.run_name}, {cfg.num_steps} steps, {len(cfg.checkpointer_config.checkpoint_steps)} checkpoints, {n_params} params", flush=True)
        t0 = time.time()
        train(cfg)
        wall = time.time() - t0
        runs[str(seed)] = {"run_name": cfg.run_name, "checkpoint_dir": str(out / "checkpoints" / cfg.checkpointer_config.project_dir),
                           "num_steps": cfg.num_steps, "n_params": n_params, "wall_seconds": wall,
                           "steps_per_second": cfg.num_steps / wall, "config": json.loads(json.dumps(cfg.model_dump(), default=str)),
                           "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__}
        (out / "runs.json").write_text(json.dumps(runs, indent=1))
        print(f"seed {seed}: {wall:.0f}s, {cfg.num_steps / wall:.1f} steps/s", flush=True)

    training, evals = [], []
    for seed, r in runs.items():
        df = pd.read_csv(out / f"metrics-seed{seed}.csv")
        df["seed"] = int(seed)
        if "batch/loss" in df:
            t = df[["seed", "step", "batch/loss"]].dropna().rename(columns={"batch/loss": "batch_loss"})
            training.append(t)
        ev_cols = [c for c in df.columns if c not in ("batch/loss",)]
        e = df[ev_cols].dropna(subset=[c for c in ev_cols if c not in ("seed", "step")], how="all")
        evals.append(e)
    pd.concat(training).sort_values(["seed", "step"]).to_parquet(out / "data" / "training.parquet", index=False)
    pd.concat(evals).sort_values(["seed", "step"]).to_parquet(out / "data" / "evals.parquet", index=False)
    print("wrote", out / "data" / "training.parquet", out / "data" / "evals.parquet")


if __name__ == "__main__":
    main()
