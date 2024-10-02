# The Developmental Landscape of In-Context Leanring

> We show that in-context learning emerges in transformers in discrete developmental stages. These stages are associated with changes in the local geometry of the loss landscape, the nature and magnitude of which are revealed by essential dynamics and the local learning coefficient of singular learning theory.

## Basic setup

Clone this repo and install standard dependencies, 

```
pip install -e . 
```

## Reproducing analysis

All the data needed to reproduce the analysis and regenerate the figures depicted in the paper is available in [`data/`](data). Figures are avaiable in [`figures/`](figures), including figures for other experiments and seeds not included in the paper.

To generate figures with the provided data, open the notebooks in [`notebooks/`](notebooks):

- [`notebooks/main-figures.ipynb`](notebooks/main-figures.ipynb) generates the main figures in the paper. 
- [`notebooks/essential-dynamics.ipynb`](notebooks/essential-dynamics.ipynb) generates the essential dynamics analysis (over a larger set of checkpoints) 
- [`notebooks/task-prior.ipynb`](notebooks/task-prior.ipynb) generates figures for a set of models trained on different task distributions that consist of a finite number of tasks. This is for figure 20, which contrasts the task prior to the "0 prediction."
- [`notebooks/llcs.ipynb`](notebooks/llcs.ipynb) generates figures related to LLC calibration (as discussed in Appendix E.3). 

## Reproducing experiments

### Checkpointing

Main language model checkpoints are available on [HuggingFace](https://huggingface.co/georgeyw/L2W256-3m-rr). 

#### Local storage

To save checkpoints locally, set the `LOCAL_ROOT` environment variable to a local directory (see `.env-template`). 

#### Using AWS 

If you choose to use AWS, set the `AWS_BUCKET_NAME` variable (see `.env-template`). This assumes that you have an AWS account and that you have set up a bucket. Make sure to also set the `AWS_SECRET_ACCESS_KEY` and `AWS_ACCESS_KEY_ID` variables. 

### Monitoring with W&B

To run experiments with W&B logging you will need your associated API key stored in your `.netrc`. Follow these steps:

1. Log in to your wandb account through the browser and copy your API
   key from [https://wandb.ai/authorize](https://wandb.ai/authorize).
2. Install wandb (`pip install -r requirements.txt` as above, or just
   `pip install wandb`)
3. Run the command `wandb login` (or `python -m wandb login`) and then
   paste your API key from step (2) when prompted.
4. This will create a file in your home directory called `.netrc`.
   Keep that file safe.

Set the `WANDB_ENTITY` env variable in `.env`.

### Configuring and deploying training runs

To run a training run, first configure a run in the [`/sweeps/training-runs`](sweeps/training-runs/) directory. The configuration for the runs featured in the paper are available in [`/sweeps/training-runs/L2H4Minf.yaml`](sweeps/training-runs/L2H4Minf.yaml).

To run a training run, prepare a wandb sweep with the following command: 

```
wandb sweep sweeps/training-runs/<config_name>.yaml
```

This will generate a sweep and print the associated ID. Run the sweep with the following command:

```
wandb agent <sweep_id>
```

### Running LLC estimation

To run LLC estimation, first configure a run in the [`/sweeps/analysis`](sweeps/analysis/) or [ `sweeps/calibration`](sweeps/calibration/) directory. The configuration for the runs featured in the paper are available in [`/sweeps/llc/0-L2H4-llcs-over-time.yaml`](sweeps/llc/0-L2H4-llcs-over-time.yaml).


To run a training run, prepare a wandb sweep with the following command: 

```
wandb sweep sweeps/analysis/<config_name>.yaml
```

This will generate a sweep and print the associated ID. Run the sweep with the following command:

```
wandb agent <sweep_id>
```

### Support for TPU training

Training on TPUs is supported out of the box. To disable TPU training, run agent with the DEVICE environment variable set to `cpu`:

```
DEVICE=cpu wandb agent <sweep_id>
```

LLC estimation is not yet supported on TPUs. Make sure to disable TPU device using the above command when running LLC estimation.


### Testing

Some of the model and baselines implementations have unit tests:

- Install additional testing dependencies `pip install pytest torch_testing`.
- Add tests in the `tests` directory (name the script `test_*.py`).
- Run the tests with the command `pytest`.


