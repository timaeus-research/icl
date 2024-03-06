from transformer_lens import HookedTransformer, HookedTransformerConfig

from icl.language.config import get_model_cfg


def get_model(num_layers=2):
    model_cfg = get_model_cfg(n_layers=num_layers)
    return HookedTransformer(model_cfg)
