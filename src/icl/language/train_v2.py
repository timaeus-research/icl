import argparse
import os
from dataclasses import dataclass

import torch
import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformer_lens.utils import tokenize_and_concatenate
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

import wandb
from icl.constants import XLA

if XLA:
    import torch_xla.core.xla_model as xm


@dataclass
class TrainingConfig:
    name: str

    # Model-related
    n_layers: int
    d_model: int
    d_head: int
    n_heads: int
    n_ctx: int
    d_vocab: int
    tokenizer_name: str
    normalization_type: str
    attn_only: bool
    act_fn: str
    seed: int
    positional_embedding_type: str

    # Training-related
    dataset: str
    dataset_config: str
    dataset_col_name: str
    train_dataset_split: str
    eval_dataset_split: str
    num_train_epochs: int
    train_batch_size: int
    eval_batch_size: int
    eval_steps: int
    save_steps: int
    warmup_steps: int
    logging_steps: int
    tpu_num_cores: int
    tpu_metrics_debug: bool
    wandb: bool = False

    @property
    def cores(self):
        if XLA:
            return max(1, self.tpu_num_cores)
        return 1

    @property
    def per_device_train_batch_size(self):
        return self.train_batch_size // self.cores
    
    @property
    def per_device_eval_batch_size(self):
        return self.eval_batch_size // self.cores

    def __setattr__(self, name, value):
        if self.wandb:
            self.__dict__[name] = value
            wandb.config[name] = value
        else:
            self.__dict__[name] = value

    def to_model_config(self): 
        return HookedTransformerConfig(
            n_layers=self.n_layers,
            d_model=self.d_model,
            d_head=self.d_head,
            n_heads=self.n_heads,
            n_ctx=self.n_ctx,
            d_vocab=self.d_vocab,
            tokenizer_name=self.tokenizer_name,
            normalization_type=self.normalization_type,
            attn_only=self.attn_only,
            seed=self.seed,
            positional_embedding_type=self.positional_embedding_type,
            act_fn=self.act_fn
        )

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, tokens):
        return self.model(tokens).logits


def train(config):
    # Load the tokenizer and create the model
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    model = HookedTransformer(config.to_model_config())

    print(yaml.dump(config))
    # model = ModelWrapper(model)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="output",
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        warmup_steps=config.warmup_steps,
        logging_dir="logs",
        logging_steps=config.logging_steps,
        tpu_num_cores=config.tpu_num_cores,
        tpu_metrics_debug=config.tpu_metrics_debug,
        label_names=["tokens"],
        report_to=("wandb" if config.wandb else "none")
        # torch_compile=True
    )

    train_dataset = load_dataset(config.dataset, config.dataset_config, split=config.train_dataset_split)
    eval_dataset = load_dataset(config.dataset, config.dataset_config, split=config.eval_dataset_split)

    # Tokenize the datasets
    train_dataset = tokenize_and_concatenate(train_dataset, tokenizer, column_name=config.dataset_col_name, streaming=False, add_bos_token=False) # BOS is already added
    eval_dataset = tokenize_and_concatenate(eval_dataset, tokenizer, column_name=config.dataset_col_name, streaming=False, add_bos_token=False)

    # Change column name to `input_ids`
    train_dataset = train_dataset.rename_column("tokens", "input_ids")
    eval_dataset = eval_dataset.rename_column("tokens", "input_ids")

    print(train_dataset)
    print(eval_dataset)
    
    print("Train dataset size:", len(train_dataset))
    print("Eval dataset size:", len(eval_dataset))

    print("Train dataset example:", train_dataset[0])
    print("Eval dataset example:", eval_dataset[0])

    # Create the Trainer and train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # callbacks=[] # TODO: for custom checkpointing
    )

    trainer.train()

if __name__ == "__main__":
    load_dotenv()
    os.environ["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "tetrahedron-3m")

    parser = argparse.ArgumentParser(description="Train a language model")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for configuration")
    parser.add_argument("--name", type=str, default="tetrahedron-3m")

    # Model-related
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_head", type=int, default=32)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_ctx", type=int, default=1024)
    parser.add_argument("--d_vocab", type=int, default=5000)
    parser.add_argument("--tokenizer_name", type=str, default="georgeyw/TinyStories-tokenizer-5k")
    parser.add_argument("--normalization_type", type=str, default="LN")
    parser.add_argument("--attn_only", action="store_true")
    parser.add_argument('--act_fn', type=str, default='gelu')
    parser.add_argument("--seed", type=int, default=0)
    
    # Training-related
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-raw-v1")
    parser.add_argument("--dataset_col_name", type=str, default="text")
    parser.add_argument("--train_dataset_split", type=str, default="train")
    parser.add_argument("--eval_dataset_split", type=str, default="validation")
    parser.add_argument("--positional_embedding_type", type=str, default="shortformer")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--tpu_num_cores", type=int, default=8)
    parser.add_argument("--tpu_metrics_debug", action="store_true")

    args = parser.parse_args()

    if args.wandb:
        # Initialize wandb and get the configuration
        wandb.init()
        config = TrainingConfig(**wandb.config, wandb=True, name=wandb.run.name)
    else:

        config_dict = vars(parser.parse_args())
        config = TrainingConfig(**config_dict)

    train(config)
