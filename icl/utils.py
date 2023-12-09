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
    "epsilon": "eps",
    "temperature": "temp",
    "eval_method": "eval",
    "eval_loss_fn": "loss",
    "gamma": "gamma",
    "num_training_samples": "n",
    "batch_size": "m", 
}

def pyvar_to_mathvar(name: str):
    return PYVAR_TO_MATHVAR.get(name.split(".")[-1].split("/")[-1], name)


def pyvar_to_slugvar(name: str):
    return PYVAR_TO_SLUGVAR.get(name.split(".")[-1].split("/")[-1])


def pyvar_dict_to_latex(d: dict):
    return "$" + ", ".join([f"{pyvar_to_mathvar(k)}={v}" for k, v in d.items() if v is not None and k in PYVAR_TO_MATHVAR]) + "$"


def pyvar_dict_to_slug(d: dict):
    return "_".join([f"{pyvar_to_slugvar(k)}{v}" for k, v in d.items() if v is not None and k in PYVAR_TO_SLUGVAR])


def get_locations(L: int):
    locations = [
        'token_sequence_transformer.token_embedding',
    ]

    for l in range(L):
        locations.extend([
            f'token_sequence_transformer.blocks.{l}',
            f'token_sequence_transformer.blocks.{l}.attention.attention',
            f'token_sequence_transformer.blocks.{l}.compute',
        ])

    locations.extend([
        'token_sequence_transformer.unembedding',
    ])

    return locations


def get_model_locations_to_display(L: int):
    locations = {
        'token_sequence_transformer.token_embedding': "Embedding",
    }

    for l in range(L):
        locations.update({
            f'token_sequence_transformer.blocks.{l}': f"Block {l}",
            f'token_sequence_transformer.blocks.{l}.attention.attention': f"Block {l} Attention Logits",
            f'token_sequence_transformer.blocks.{l}.compute': f"Block {l} MLP",
        })

    locations.update({
        'token_sequence_transformer.unembedding': "Unembedding",
    })

    return locations


def get_model_locations_to_slug(L: int):
    locations = {
        'token_sequence_transformer.token_embedding': "0.0-embed",
    }

    for l in range(L):
        locations.update({
            f'token_sequence_transformer.blocks.{l}': f"{l+1}-block",
            f'token_sequence_transformer.blocks.{l}.attention.attention': f"{l+1}.1-attn",
            f'token_sequence_transformer.blocks.{l}.compute': f"{l+1}.2-mlp",
        })

    locations.update({
        'token_sequence_transformer.unembedding': f"{L+1}-unembed-ln",
    })

    return locations
