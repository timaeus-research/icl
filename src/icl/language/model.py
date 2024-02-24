from transformer_lens import HookedTransformer, HookedTransformerConfig


def get_model(num_layers=2):
    model_cfg = get_model_cfg(num_layers=num_layers)
    return HookedTransformer(model_cfg)
