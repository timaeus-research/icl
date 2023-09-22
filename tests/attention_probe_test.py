#%%
mechinterp_path = "/Users/liam/Documents/DevInterp/ICL2023/mechinterp"
from icl.mechinterp import AttentionProbe
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

attprobe = AttentionProbe(transformer, mechinterp_path)

df = attprobe.make_attention_df(xs, ys, "TEST0", "TEST_attn")


# %%
