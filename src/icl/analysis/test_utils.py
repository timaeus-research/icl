import torch
import pandas as pd
from icl.analysis.utils import (
    get_run,
    get_unique_config,
    get_unique_run,
    get_sweep_configs,
    wandb_run_to_df,
    wandb_runs_to_df,
    load_model_at_step,
    load_model_at_last_checkpoint,
    map_evals_over_checkpoints,
    split_attn_weights,
    get_weights,
    log_on_update,
    match_template,
)

def test_get_run():
    # Test case 1: Test get_run with valid sweep and filters
    run = get_run("sweep_path", filter1="value1", filter2="value2")
    assert isinstance(run, RegressionRun)
    
    # Test case 2: Test get_run with invalid sweep and filters
    run = get_run("invalid_sweep_path", filter1="value1", filter2="value2")
    assert run is None

def test_get_unique_config():
    # Test case 1: Test get_unique_config with valid sweep and filters
    config = get_unique_config("sweep_path", filter1="value1", filter2="value2")
    assert isinstance(config, RegressionConfig)
    
    # Test case 2: Test get_unique_config with invalid sweep and filters
    config = get_unique_config("invalid_sweep_path", filter1="value1", filter2="value2")
    assert config is None

def test_get_unique_run():
    # Test case 1: Test get_unique_run with valid sweep and filters
    run = get_unique_run("sweep_path", filter1="value1", filter2="value2")
    assert isinstance(run, RegressionRun)
    
    # Test case 2: Test get_unique_run with invalid sweep and filters
    run = get_unique_run("invalid_sweep_path", filter1="value1", filter2="value2")
    assert run is None

def test_get_sweep_configs():
    # Test get_sweep_configs with valid sweep and filters
    configs = list(get_sweep_configs("sweep_path", filter1="value1", filter2="value2"))
    assert isinstance(configs, list)
    assert all(isinstance(config, RegressionConfig) for config in configs)
    
    # Test get_sweep_configs with invalid sweep and filters
    configs = list(get_sweep_configs("invalid_sweep_path", filter1="value1", filter2="value2"))
    assert isinstance(configs, list)
    assert len(configs) == 0

def test_wandb_run_to_df():
    # Test wandb_run_to_df with a valid run
    run = RegressionRun()
    history_df = wandb_run_to_df(run)
    assert isinstance(history_df, pd.DataFrame)
    assert len(history_df) > 0

def test_wandb_runs_to_df():
    # Test wandb_runs_to_df with a list of valid runs
    runs = [RegressionRun(), RegressionRun(), RegressionRun()]
    df = wandb_runs_to_df(runs)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

def test_load_model_at_step():
    # Test load_model_at_step with a valid config and step
    config = RegressionConfig()
    model = load_model_at_step(config, step=10)
    assert isinstance(model, torch.nn.Module)

def test_load_model_at_last_checkpoint():
    # Test load_model_at_last_checkpoint with a valid config
    config = RegressionConfig()
    model = load_model_at_last_checkpoint(config)
    assert isinstance(model, torch.nn.Module)

def test_map_evals_over_checkpoints():
    # Test map_evals_over_checkpoints with a valid model, checkpointer, and evaluator
    model = torch.nn.Module()
    checkpointer = BaseStorageProvider()
    evaluator = ModelEvaluator()
    evals = list(map_evals_over_checkpoints(model, checkpointer, evaluator))
    assert isinstance(evals, list)
    assert len(evals) > 0

def test_split_attn_weights():
    # Test split_attn_weights with a valid tensor and parameters
    W = torch.randn(512, 8, 64)
    num_heads = 8
    embed_dim = 512
    head_size = 64
    attn_weights = list(split_attn_weights(W, num_heads, embed_dim, head_size))
    assert isinstance(attn_weights, list)
    assert len(attn_weights) == num_heads
    assert all(isinstance(weights, tuple) for weights in attn_weights)

def test_get_weights():
    # Test get_weights with a valid model and paths
    model = torch.nn.Module()
    paths = ["layer1", "layer2"]
    weights = list(get_weights(model, paths))
    assert isinstance(weights, list)
    assert len(weights) > 0
    assert all(isinstance(weight, torch.Tensor) for weight in weights)

def test_log_on_update():
    # Test log_on_update with a valid callback, monitor, and log_fn
    callback = Callback()
    monitor = Monitor()
    log_fn = LogFunction()
    updated_callback = log_on_update(callback, monitor, log_fn)
    assert callable(updated_callback.update)

def test_match_template():
    # Test match_template with valid template and string
    template = 'token_sequence_transformer.blocks.*.attention.**'
    string1 = 'token_sequence_transformer.blocks.0.attention.attention.weight'
    string2 = 'token_sequence_transformer.blocks.1.ffn.weight'
    string3 = 'token_sequence_transformer.blocks.2.attention.bias'
    assert match_template(template, string1) is True
    assert match_template(template, string2) is False
    assert match_template(template, string3) is Trueimport unittest
from icl.analysis.utils import match_template

class TestMatchTemplate(unittest.TestCase):
    def test_matching_string(self):
        template = 'token_sequence_transformer.blocks.*.attention.**'
        string1 = 'token_sequence_transformer.blocks.0.attention.attention.weight'
        string2 = 'token_sequence_transformer.blocks.2.attention.bias'

        self.assertTrue(match_template(template, string1))
        self.assertTrue(match_template(template, string2))

    def test_non_matching_string(self):
        template = 'token_sequence_transformer.blocks.*.attention.**'
        string1 = 'token_sequence_transformer.blocks.1.ffn.weight'

        self.assertFalse(match_template(template, string1))

    def test_wildcard_characters(self):
        template = 'token_sequence_transformer.blocks.*.attention.**'
        string1 = 'token_sequence_transformer.blocks.0.attention.attention.weight'
        string2 = 'token_sequence_transformer.blocks.0.attention.attention_bias.weight'

        self.assertTrue(match_template(template, string1))
        self.assertFalse(match_template(template, string2))

if __name__ == '__main__':
    unittest.main()