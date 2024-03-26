import argparse
import os
import sys
from dataclasses import asdict, dataclass, field

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utils import tokenize_and_concatenate
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainingArguments)

import wandb
from icl.constants import XLA

if XLA:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp


@dataclass
class ModelArguments:
    n_layers: int = field(default=2)
    d_model: int = field(default=256)
    d_head: int = field(default=32)
    n_heads: int = field(default=8)
    n_ctx: int = field(default=1024)
    d_vocab: int = field(default=5000)
    tokenizer_name: str = field(default="georgeyw/TinyStories-tokenizer-5k")
    normalization_type: str = field(default="LN")
    attn_only: bool = field(default=False)
    act_fn: str = field(default="gelu")
    # seed: int = field(default=0)
    positional_embedding_type: str = field(default="shortformer")

@dataclass
class DataTrainingArguments:
    dataset: str = field(default="wikitext")
    dataset_config: str = field(default="wikitext-103-raw-v1")
    dataset_col_name: str = field(default="text")
    train_dataset_split: str = field(default="train")
    eval_dataset_split: str = field(default="validation")


def main():
    load_dotenv()
    os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "tetrahedron-3m")

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses() 

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments

    print("\nModel arguments:")
    print(yaml.dump(asdict(model_args)))

    print("\nData arguments:")
    print(yaml.dump(asdict(data_args)))

    print("\nTraining arguments:")
    print(yaml.dump(asdict(training_args)))
    print("")

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    model = HookedTransformer(HookedTransformerConfig(**asdict(model_args)))

    # Datasets
    train_dataset = load_dataset(data_args.dataset, data_args.dataset_config, split=data_args.train_dataset_split)
    eval_dataset = load_dataset(data_args.dataset, data_args.dataset_config, split=data_args.eval_dataset_split)

    train_dataset = tokenize_and_concatenate(train_dataset, tokenizer, column_name=data_args.dataset_col_name, streaming=False, add_bos_token=False) # BOS is already added
    eval_dataset = tokenize_and_concatenate(eval_dataset, tokenizer, column_name=data_args.dataset_col_name, streaming=False, add_bos_token=False)

    train_dataset = train_dataset.rename_column("tokens", "input_ids")
    eval_dataset = eval_dataset.rename_column("tokens", "input_ids")

    # Training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # callbacks=[] # TODO: for custom checkpointing
    )

    trainer.train()


def _mp_fn(index):
    main()


if __name__ == "__main__":
    if XLA:
        xmp.spawn(_mp_fn, args=(), nprocs=8, start_method='fork')
    else:
        main()

