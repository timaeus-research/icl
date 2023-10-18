import wandb
import pandas as pd

def get_dataframe_from_run(run):
    """Get the data from a specific run as a pandas DataFrame."""
    history = run.history()
    return pd.DataFrame(history)


def all_plots_of_sweep(sweep_name):
    api = wandb.Api()

    # Replace with your project and sweep name
    project = "icl"
    entity = "devinterp"


    # Iterate over runs in the sweep
    sweep = api.sweep(f"{entity}/{project}/{sweep_name}")
    for run in runs:
        print(f"Processing run {run.name}...")
        df = get_dataframe_from_run(run)
        plot_dataframe(df, run.name)