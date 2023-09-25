MECHINTERP_PATH = "/Users/liam/Documents/DevInterp/ICL2023/mechinterp"

from icl.utils import directory_creator, open_or_create_csv
import pandas as pd
from itertools import product
import numpy as np


class AttentionProbe():
    """
    A class for analysing attention scores for a fixed transformer with variable inputs.

    Parameters:
    * `transformer : InContextRegressionTransformer`
        A pre-trained transformer model 
    * `MECHINTERP_PATH : str`
        mechinterp analysis folder on local device
    """
    def __init__(self, transformer, MECHINTERP_PATH):
        self.transformer = transformer
        self.path = MECHINTERP_PATH

    def make_attention_df(self, xs, ys, to_csv=False, name="TEST", subdirectory="TEST_attn"):
        """
        Create a dataframe of (flattened) attention scores of each head in 
        each layer of each example in batch. Dataframe is of size (B*L*H, T^2). 

        Parameters:

        * `xs : tensor(B, K, D)`
            batch of `B` sequences of `K` input vectors of `D` dims.
        * `ys : tensor(B, K, 1)`
            batch of `B` sequences of `K` output scalars.
        * `to_csv : boolean` 
            flags whether to save to csv 
        * `name : str`
            name of dataframe
        * `subdirectory : str'
            name of subdirectory in MECHINTERP_PATH, created if not already existing

        Returns:

        * `df : pandas DataFrame`
            A dataframe of shape (B*L*H, 3+T^2) containing attention scores. 

        """
        # run forward pass on transformer with given inputs
        _, attention = self.transformer.forward(xs, ys, mechinterp=True)
        B, L, H, T = attention.size()[:4]

        # attention is of form B L H T T 
        attention = attention.reshape(B,L,H,-1)             # B L H T T -> B L H T^2 
        attention_matrix = (attention
                            .contiguous()                   # Ensure memory layout matches shape
                            .view(-1, T**2)                 # Flatten last two dimensions
                            .detach()                       # Detach from computation graph
                            .numpy()                        # Convert to numpy array
                        )                                   # B L H T^2 -> B*L*H T^2
        
        # I have checked that the tensor is flattened correctly i.e. [0,0,0], [0,0,1], [0,1,0] etc. 
        combinations = np.array(list(product(range(B), range(L), range(H))))
        df_values = np.concatenate([combinations, attention_matrix], axis=1)

        headers = ['B', 'L', 'H'] + [f"A_[{i+1},{j+1}]" for i in range(T) for j in range(T)]
        df = pd.DataFrame(df_values, columns=headers)
        df[['B', 'L', 'H']] = df[['B', 'L', 'H']].astype(int) 

        if to_csv:
            attn_folder = directory_creator(self.path, new_subdir = subdirectory)
            df.to_csv(attn_folder + "/" + name + ".csv", index=False)
        
        return df
    
    def attention_avg_batch(self, df):
        """
        Get df of attention scores averaged over a batch 

        Parameters:
        * `df : pandas DataFrame`
            A df of the form returned by make_attention_df of size (BLH)

        Returns:

        * `df : pandas DataFrame`
            A dataframe of shape (L*H, 3+T^2) containing averaged attention scores. 
        """
        df_mean = (df
                   .drop(columns='B')       # B is irrelevant since groupby over [L, H]
                   .groupby(['L', 'H'])
                   .mean()
                   .reset_index())
        return df_mean
    
    def prefix_matching_score(self, induction_task):
        """
        Get average prefix matching score on an Induction batch defined in mechinterp_tasks.InductionHeadsTask.

        Parameters:
        * 
        """