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

    # special code if device is 'xla'
    XLA = (config.device == 'xla')
    if XLA:
        stdlogger.info("device is 'xla'! some special code will run...")
        stdlogger.info("importing torch_xla...")
        import torch_xla.core.xla_model as xm
        # import torch_xla.debug.metrics as met
        stdlogger.info("configuring default XLA device...")
        config.device = xm.xla_device()
        stdlogger.info("xla ready!")

    # model initialisation
    stdlogger.info("initialising model")
    model = config.task_config.model_factory().to(config.device)
    model.train()

    # initialise 'pretraining' data source (for training on fixed task set)
    stdlogger.info("initialising data (pretrain)")
    pretrain_dist = config.task_config.pretrain_dist_factory().to(config.device)

    # initialise 'true' data source (for evaluation, including unseen tasks)
    stdlogger.info("initialising data (true)")
    true_dist = config.task_config.true_dist_factory().to(config.device)

    # initialise evaluations
    stdlogger.info("initialising evaluator")
    if XLA: xm.mark_step()
    evaluator = ICLEvaluator(
        pretrain_dist=pretrain_dist,
        true_dist=true_dist,
        max_examples=config.task_config.max_examples,
        eval_batch_size=config.eval_batch_size,
        seed=config.task_config.true_seed
    )
    if XLA: xm.mark_step()

    # initialise monitoring code
    stdlogger.info("initialising checkpointer and logger")
    checkpointer = config.checkpointer_config.factory() if config.checkpointer_config is not None else None
    logger = config.logger_config.factory() if config.logger_config is not None else None

    # initialise torch optimiser
    stdlogger.info("initialising optimiser and scheduler")
    optimizer = config.optimizer_config.factory(model.parameters())
    scheduler = config.scheduler_config.factory(optimizer)  # type: ignore

    # TODO: this is unused and may be slowing down XLA... use it or lose it
    # recent_losses = torch.zeros(100, device=config.device)

    # training loop
    stdlogger.info("starting training loop")
    stdlogger.info("note: first two iterations slow while XLA compiles")
    stdlogger.info("note: early iterations slow due to logspace checkpoints")
    for step in tqdm.trange(config.num_steps, desc="training..."):
        # per-step seeds for reproducibility if we resume training
        set_seed(config.task_config.sampling_seed + step)
        
        # training step
        if XLA: xm.mark_step()
        xs, ys = pretrain_dist.get_batch(
            num_examples=config.task_config.max_examples,
            batch_size=config.batch_size,
        )
        ys_pred = model(xs, ys)
        loss = F.mse_loss(ys, ys_pred)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if XLA: xm.mark_step()

        # see above
        # recent_losses[step % 100] = loss

        # wand logging: log batch loss every 100 steps
        if step % 100 == 0 and step > 0 and config.is_wandb_enabled:
            stdlogger.info("logging batch loss at step %s", step)
            # TODO: Figure out how to make this work with `logger`
            wandb.log({"batch/loss": loss.mean().item()}, step=step)
        
        # evaluate and log metrics to wandb according to log_steps
        if step in config.logger_config.logging_steps:
            stdlogger.info("evaluating metrics at step %s", step)
            if XLA: xm.mark_step()
            model.eval()
            metrics = evaluator(model)
            model.train()
            if XLA: xm.mark_step()
            stdlogger.info("logging metrics at step %s", step)
            logger.log(metrics, step=step)


        # save checkpoints according to checkpoint_steps
        if step in config.checkpointer_config.checkpoint_steps:
            # TODO: if xla: move model to CPU before saving
            stdlogger.info("saving checkpoint at step %s", step)
            if XLA: xm.mark_step()
            checkpointer.save_file(step, state_dict(model, optimizer, scheduler))
            if XLA: xm.mark_step()

    if config.is_wandb_enabled:
        wandb.finish()

    # TODO: if XLA, move model off TPU?
    return model

