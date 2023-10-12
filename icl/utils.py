import os

import pandas as pd


def directory_creator(directory, new_subdir):
    # Creates a new directory if it doesn't already exist
    
    new_directory = directory + "/" + new_subdir
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        
    return new_directory

def open_or_create_csv(filename, headers=None):
    '''
    Open a CSV file if it exists. If it doesn't exist, create it.

    :param filename: The name/path of the CSV file
    :param headers: A list of headers to write to the new CSV, if it's being created
    :return: A DataFrame with the CSV content or an empty DataFrame with specified headers
    '''

    # Check if the file exists
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=headers)
        df.to_csv(filename, index=False)

    return df


PYVAR_TO_MATHVAR = {
    "num_tasks": "M",
    "num_layers": "L",
    "num_heads": "H",
    "max_examples": "K",
    "lr": r"\eta",
    "elasticity": r"\gamma",
    "num_draws": r"n_\mathrm{draws}",
    "num_chains": r"n_\mathrm{chains}",
    "num_samples": r"n",
}

PYVAR_TO_SLUGVAR = {
    "num_tasks": "M",
    "num_layers": "L",
    "num_heads": "H",
    "max_examples": "K",
    "lr": "lr",
    "elasticity": "g",
    "num_draws": "ndraws",
    "num_chains": "nchains",
    "num_samples": "n",
}


def pyvar_to_mathvar(name: str):
    return PYVAR_TO_MATHVAR[name.split(".")[-1].split("/")[-1]]


def pyvar_to_slugvar(name: str):
    return PYVAR_TO_SLUGVAR[name.split(".")[-1].split("/")[-1]]


def pyvar_dict_to_latex(d: dict):
    return "$" + ", ".join([f"{pyvar_to_mathvar(k)}={v}" for k, v in d.items()]) + "$"


def pyvar_dict_to_slug(d: dict):
    return "_".join([f"{pyvar_to_slugvar(k)}{v}" for k, v in d.items()])