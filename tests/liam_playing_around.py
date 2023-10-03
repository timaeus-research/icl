#%%

from icl.model import InContextRegressionTransformer, to_token_sequence
from icl.tasks import RegressionSequenceDistribution, DiscreteTaskDistribution


transformer = InContextRegressionTransformer(task_size=8,
                                                  max_examples=16,
                                                  embed_size=128,
                                                  mlp_size=128,
                                                  num_heads=2,
                                                  num_layers=8,
                                                  )

pretrain_dist = RegressionSequenceDistribution(
                    task_distribution=DiscreteTaskDistribution(
                        num_tasks=2,
                        task_size=8,
                    ),
                    noise_variance = 0.25**2,
                )
xs, ys = pretrain_dist.get_batch(num_examples=16, batch_size=4)
toks = to_token_sequence(xs,ys) # size (B, 2*K, D+1) 

# test forward pass with mechinterp
result = transformer.forward(xs, ys, mechinterp=True)  


# %%

transformer_state = init_transformer.state_dict()

# analysing the first attention layer 0 

attn_causal = transformer_state['token_sequence_transformer.blocks.0.attention.causal_mask']
attn_attn_weight = transformer_state['token_sequence_transformer.blocks.0.attention.attention.weight']
attn_compute_0_weight = transformer_state['token_sequence_transformer.blocks.0.compute.0.weight']
attn_compute_0_bias = transformer_state['token_sequence_transformer.blocks.0.compute.0.bias']
attn_compute_2_weight = transformer_state['token_sequence_transformer.blocks.0.compute.2.weight']
attn_compute_2_bias = transformer_state['token_sequence_transformer.blocks.0.compute.2.bias']
attn_layer_norms_0_weight = transformer_state['token_sequence_transformer.blocks.0.layer_norms.0.weight']
attn_layer_norms_0_bias = transformer_state['token_sequence_transformer.blocks.0.layer_norms.0.bias']
attn_layer_norms_1_weight = transformer_state['token_sequence_transformer.blocks.0.layer_norms.1.weight']
attn_layer_norms_1_bias = transformer_state['token_sequence_transformer.blocks.0.layer_norms.1.bias']

attention_layer_0 = {'attn_causal': attn_causal, 
                     'attn_attn_weight': attn_attn_weight,
                     'attn_compute_0_weight': attn_compute_0_weight,
                     'attn_compute_0_bias': attn_compute_0_bias,
                     'attn_compute_2_weight': attn_compute_2_weight,
                     'attn_compute_2_bias': attn_compute_2_bias,
                     'attn_layer_norms_0_weight': attn_layer_norms_0_weight,
                     'attn_layer_norms_0_bias':attn_layer_norms_0_bias,
                     'attn_layer_norms_1_weight': attn_layer_norms_1_weight,
                     'attn_layer_norms_1_bias':attn_layer_norms_1_bias
}

for key, value in attention_layer_0.items():
    print("Key:", key)
    # Assuming the value is a numpy array or tensor
    print("Size:", value.shape)
    # If the value is a list or some other Python collection, use:
    # print("Size:", len(value))
    print("Value:", value)
    print('---')  # to separate entries for better readability


embedding_weights = transformer_state['token_sequence_transformer.token_embedding.weight']
# %%
