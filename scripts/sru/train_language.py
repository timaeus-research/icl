#!/usr/bin/env python3
"""Train the two-layer attention-only language model from the paper's sweep YAML without W&B or S3.

    ICL_STREAMING=1 python scripts/sru/train_language.py --sweep sweeps/language/training-runs/tetrahedron-3m.yaml \
        --seed 0 --out /work/results [--steps 50000] [--checkpoint-steps linear_space=501]

Builds the run config with `icl.language.config.get_config` as `train.py` does (AdamW 1e-3, wd 0.05, batch 100,
50k steps, TransformerLens attention-only config), streams timaeus/dsir-pile-10m-tokens in file order (one epoch,
no shuffle), stores checkpoints under <out>/checkpoints/ through the LocalStorageProvider, and writes
<out>/data/training.parquet (seed, step, batch_loss every 100 steps) and <out>/data/evals.parquet (seed, step,
test loss on the paper's test set at the logging steps) plus <out>/runs.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import yaml

from train_regression import steps_spec, sweep_to_kwargs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--checkpoint-steps", default="linear_space=501", help="linear_space=501 over 50k = every 100 steps")
    ap.add_argument("--logging-steps", default="log_space=50,linear_space=101")
    a = ap.parse_args()

    out = Path(a.out); (out / "data").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("AWS_LANGUAGE_BUCKET_NAME", ""); os.environ.setdefault("AWS_BUCKET_NAME", "")
    os.environ["LOCAL_ROOT"] = str(out)
    import torch
    from icl.language.config import get_config
    from icl.language.train import train

    sweep = yaml.safe_load(open(a.sweep))
    kw = sweep_to_kwargs(sweep["parameters"], a.seed)
    kw.setdefault("transformer_config", {})["seed"] = a.seed
    if "dataset" in kw:
        kw["trainset"] = kw.pop("dataset")
    if a.steps:
        kw["num_steps"] = a.steps
    if a.batch_size:
        kw["batch_size"] = a.batch_size
    kw["checkpointer_config"] = {"checkpoint_steps": steps_spec(a.checkpoint_steps), "bucket_name": None, "local_root": str(out)}
    csv = out / f"metrics-seed{a.seed}.csv"
    if csv.exists():
        csv.unlink()
    kw["logger_config"] = {"logging_steps": steps_spec(a.logging_steps), "project": None, "entity": None, "out_file": str(csv)}
    cfg = get_config(project=None, entity=None, **kw)
    n_params = sum(p.numel() for p in cfg.transformer_factory().parameters())
    print(f"seed {a.seed}: run {cfg.run_name}, {cfg.num_steps} steps, batch {cfg.batch_size}, {len(cfg.checkpointer_config.checkpoint_steps)} checkpoints, {n_params} params", flush=True)
    t0 = time.time()
    train(cfg)
    wall = time.time() - t0
    runs = json.loads((out / "runs.json").read_text()) if (out / "runs.json").exists() else {}
    runs[str(a.seed)] = {"run_name": cfg.run_name, "checkpoint_dir": str(out / "checkpoints" / cfg.checkpointer_config.project_dir),
                         "num_steps": cfg.num_steps, "batch_size": cfg.batch_size, "n_params": n_params, "wall_seconds": wall,
                         "steps_per_second": cfg.num_steps / wall, "config": json.loads(json.dumps(cfg.model_dump(), default=str)),
                         "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__}
    (out / "runs.json").write_text(json.dumps(runs, indent=1))
    print(f"seed {a.seed}: {wall:.0f}s, {cfg.num_steps / wall:.1f} steps/s", flush=True)

    df = pd.read_csv(csv); df["seed"] = a.seed
    training = df[["seed", "step", "batch/loss"]].dropna().rename(columns={"batch/loss": "batch_loss"}) if "batch/loss" in df else pd.DataFrame(columns=["seed", "step", "batch_loss"])
    ev_cols = [c for c in df.columns if c != "batch/loss"]
    evals = df[ev_cols].dropna(subset=[c for c in ev_cols if c not in ("seed", "step")], how="all")
    pd.DataFrame([{"seed": int(s), **{k: v for k, v in r.items() if k != "config"}} for s, r in runs.items()]).to_parquet(out / "data" / "runs.parquet", index=False)
    training.sort_values("step").to_parquet(out / "data" / "training.parquet", index=False)
    evals.sort_values("step").to_parquet(out / "data" / "evals.parquet", index=False)
    print("wrote", out / "data")


if __name__ == "__main__":
    main()
