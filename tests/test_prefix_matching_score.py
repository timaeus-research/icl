#%%

from icl.model import InContextRegressionTransformer

init_transformer = InContextRegressionTransformer(task_size=8,
                                                  max_examples=16,
                                                  embed_size=128,
                                                  mlp_size=128,
                                                  num_heads=2,
                                                  num_layers=8,
                                                  )


# %%

transformer_state = init_transformer.state_dict()
# %%
