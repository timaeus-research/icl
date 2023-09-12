import sys

import tqdm
import torch

from icl.model import InContextRegressionTransformer
from icl.tasks import RegressionSequenceDistribution
from icl.tasks import DiscreteTaskDistribution


# DEVICE CONFIG...

XLA = sys.argv[1] == 'xla'
if XLA:
    print("using XLA")
    print("importing...")
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    print("importing complete.")
    print("defining device...")
    DEVICE          = xm.xla_device()
    print("device defined.")
else:
    print("using CPU")
    DEVICE          = 'cpu'


# OTHER CONFIG

INPUT_DIMENSION     = 8
MAX_NUM_EXAMPLES    = 16
NUM_DISTINCT_TASKS  = 64
NOISE_VARIANCE      = 0.25

MODEL_EMBED_SIZE    = 128
MODEL_MLP_SIZE      = 128
MODEL_NUM_HEADS     = 2
MODEL_NUM_LAYERS    = 8

BATCH_SIZE          = 256
NUM_TRAINING_STEPS  = 500000
MAX_LEARNING_RATE   = 1e-3
MIN_LEARNING_RATE   = MAX_LEARNING_RATE / (NUM_TRAINING_STEPS / 2 - 1)
ANNEALING_STRATEGY  = 'linear'
PERCENT_WARMUP      = .5


def train():
    data = RegressionSequenceDistribution(
        task_distribution=DiscreteTaskDistribution(
            num_tasks=NUM_DISTINCT_TASKS,
            task_size=INPUT_DIMENSION,
        ),
        noise_variance=NOISE_VARIANCE,
    ).to(DEVICE)

    model = InContextRegressionTransformer(
        task_size=INPUT_DIMENSION,
        max_examples=MAX_NUM_EXAMPLES,
        embed_size=MODEL_EMBED_SIZE,
        mlp_size=MODEL_MLP_SIZE,
        num_heads=MODEL_NUM_HEADS,
        num_layers=MODEL_NUM_LAYERS,
    ).to(DEVICE)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=MAX_LEARNING_RATE, # unused, overwritten by scheduler
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=MAX_LEARNING_RATE,
        total_steps=NUM_TRAINING_STEPS,
        anneal_strategy=ANNEALING_STRATEGY,
        div_factor=MAX_LEARNING_RATE / MIN_LEARNING_RATE,
        final_div_factor=MAX_LEARNING_RATE / MIN_LEARNING_RATE,
        pct_start=PERCENT_WARMUP,
        cycle_momentum=False, # N/A, but required to avoid error
    )


    training_loop = tqdm.trange(NUM_TRAINING_STEPS)
    if XLA: xm.mark_step()
    for step in training_loop:
        xs, ys = data.get_batch(
            num_examples=MAX_NUM_EXAMPLES,
            batch_size=BATCH_SIZE,
        )
        ys_pred = model(xs, ys)
        loss = (ys - ys_pred).square().mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if XLA: xm.mark_step()
        
        if step % 100 == 0:
            training_loop.set_description(f"loss: {loss.item():.1f}")
    
    if XLA:
        print(met.metrics_report())

"""
Findings:

* CPU:                      ~12 steps / sec (~12hrs)
* home GPU:                 ~12 steps / sec (~12hrs)
* TPU (no mp, batch256):    ~33 steps / sec  (4+hrs)
    * vary batch size?
* TPU (mp)?
"""


if __name__ == "__main__":
    train()
