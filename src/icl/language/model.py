from transformer_lens import HookedTransformer, HookedTransformerConfig


def get_model_cfg(num_layers=2):
    return HookedTransformerConfig(
        n_layers=num_layers,
        d_model=256,
        d_head=32,
        n_heads=8,
        n_ctx=1024,
        d_vocab=5000,
        tokenizer_name='georgeyw/TinyStories-tokenizer-5k',
        normalization_type='LN',
        attn_only=True,
        seed=1,
        positional_embedding_type='shortformer',
    )


def get_model(num_layers=2):
    model_cfg = get_model_cfg(num_layers=num_layers)
    return HookedTransformer(model_cfg)