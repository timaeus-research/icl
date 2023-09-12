"""
training the transformer on synthetic in-context regression task
"""
# manage environment

import logging
# from typing import Annotated, Dict, List, Literal, Optional, Tuple, TypedDict
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import dotenv
import numpy as np
import sentry_sdk
import torch
import torch.nn.functional as F
import tqdm
import typer
import wandb

print("[train] importing xla...")
import torch_xla.core.xla_model as xm
print("[train] import complete.")
print("[train] importing xla metrics...")
import torch_xla.debug.metrics as met
print("[train] import complete.")

from icl.config import ICLConfig, get_config
from icl.evals import ICLEvaluator
from icl.model import InContextRegressionTransformer
from icl.utils import set_seed


dotenv.load_dotenv()


stdlogger = logging.getLogger("ICL")


class StateDict(TypedDict):
    model: Dict
    optimizer: Dict
    scheduler: Dict


def state_dict(model, optimizer, scheduler) -> StateDict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }



def train(config: ICLConfig, is_debug: bool = False) -> InContextRegressionTransformer:
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
    logging.basicConfig(level=logging.INFO if not is_debug else logging.DEBUG)

    print("[train] initialising model.")
    # initialise model
    model = config.task_config.model_factory().to(config.device)
    model.train()

    print("[train] initialising pretraining data.")
    # initialise 'pretraining' data source (for training on fixed task set)
    pretrain_dist = config.task_config.pretrain_dist_factory().to(config.device)

    print("[train] initialising true data.")
    # initialise 'true' data source (for evaluation, including unseen tasks)
    true_dist = config.task_config.true_dist_factory().to(config.device)

    print("[train] initialising evaluator.")
    # initialise evaluations
    evaluator = ICLEvaluator(
        pretrain_dist=pretrain_dist,
        true_dist=true_dist,
        max_examples=config.task_config.max_examples,
        eval_batch_size=config.eval_batch_size,
        seed=config.task_config.true_seed
    )

    print("[train] initialising checkpointer.")
    # initialise monitoring code
    checkpointer = config.checkpointer_config.factory() if config.checkpointer_config is not None else None
    logger = config.logger_config.factory() if config.logger_config is not None else None

    # initialise torch optimiser
    optimizer = config.optimizer_config.factory(model.parameters())
    scheduler = config.scheduler_config.factory(optimizer)  # type: ignore

    num_steps = config.num_steps

    recent_losses = torch.zeros(100, device=config.device)

    # training loop
    for step in tqdm.trange(num_steps, desc="Training..."):
        tqdm.tqdm.write(f"[train] training loop step {step}")
        set_seed(config.task_config.sampling_seed + step)  # For reproducibility if we resume training

        # data generation and forward pass
        xs, ys = pretrain_dist.get_batch(
            num_examples=config.task_config.max_examples,
            batch_size=config.batch_size,
        )
        ys_pred = model(xs, ys)
        loss = F.mse_loss(ys, ys_pred)
        # backward pass and gradient step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        xm.mark_step()

        if step % 10 == 0:
            tqdm.tqdm.write("writing metrics...")
            tqdm.tqdm.write(met.metrics_report())

        recent_losses[step % 100] = loss

        if step % 100 == 0 and step > 0 and config.is_wandb_enabled:
            # TODO: Figure out how to make this work with Logger
            wandb.log({"batch/loss": loss.mean().item()}, step=step)

        # Log to wandb & save checkpoints according to log_steps
        if step in config.checkpointer_config.checkpoint_steps:
            stdlogger.info("Saving checkpoint at step %s", step)
            checkpointer.save_file(step, state_dict(model, optimizer, scheduler))

        if step in config.logger_config.logging_steps:
            stdlogger.info("Logging at step %s", step)
            model.eval()
            metrics = evaluator(model)
            model.train()
            logger.log(metrics, step=step)

    if config.is_wandb_enabled:
        wandb.finish()

    return model

