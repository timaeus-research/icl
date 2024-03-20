"""
Decoder-only transformer architecture retrofit for in-context linear
regression problem.

"""


import torch

from icl.regression.dtransformer import DTransformer


class InContextRegressionTransformer(torch.nn.Module):
    """
    A transformer model specifically designed for in-context learning in a linear regression setting. 
    This model uses a transformer architecture to make predictions based on a sequence of input-target 
    pairs (x, y), simulating a linear regression problem with Gaussian noise on the targets.

    The transformer is trained to predict the target value for a given input within the context of 
    preceding input-target pairs. It leverages the structure of the transformer to encode and process 
    these sequences effectively.

    Parameters:
    task_size (int): The dimensionality (D) of the input vectors (x). This defines the size of each task vector.
    max_examples (int): The maximum number of input-target pairs (K) in the context. This sets the upper limit 
        on the length of the context sequence the model can handle.
    embed_size (int): The dimensionality of the residual stream in the transformer. This size is crucial for 
        the internal representations of the transformer.
    mlp_size (int): The width of the Multi-Layer Perceptron (MLP) layers used within the transformer. This 
        defines the capacity and complexity of the MLP components.
    num_heads (int): The number of attention heads in the transformer. Multiple heads allow the model to 
        focus on different parts of the input sequence simultaneously.
    num_layers (int): The number of layers in the transformer. Each layer consists of self-attention and 
        MLP components.
    device (str, optional): The device (e.g., 'cpu', 'cuda') on which the model computations are performed. 
        Defaults to 'cpu'.
    layer_norm (bool, optional): Flag to include or exclude layer normalization in the transformer layers. 
        Defaults to True.
    include_output (bool, optional): Flag to include or exclude the projection layer after attention.
        Defaults to False.

    The model is trained to minimize a loss function that incorporates the Mean Squared Error (MSE) between the 
    predicted and actual target values across all contexts of length up to K, as described in the 
    referenced literature.
    """
    def __init__(
        self,
        task_size: int,
        max_examples: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_layers: int,
        device='cpu',
        layer_norm=True,
        include_output=False,
    ):
        super().__init__()
        self.token_sequence_transformer = DTransformer(
            token_size=1 + task_size,    # task_size for x + 1 for y
            max_tokens=2 * max_examples, # one x + one y per example
            embed_size=embed_size,
            mlp_size=mlp_size,
            num_heads=num_heads,
            num_layers=num_layers,
            device=device,
            layer_norm=layer_norm,
            include_output=include_output,
        )
        self.task_size = task_size
        self.max_examples = max_examples
        self.device = device
        self.embed_size = embed_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.num_layers = num_layers
    
    def forward(self, xs, ys):
            """
            Forward pass of the regression model.

            Args:
                xs (torch.Tensor): Input tensor of shape (B, K, D), where B is the batch size,
                    K is the number of examples, and D is the input size.
                ys (torch.Tensor): Target tensor of shape (B, K, 1), where B is the batch size,
                    K is the number of examples, and 1 is the scalar output.

            Returns:
                torch.Tensor: Predicted output tensor of shape (B, K, 1), where B is the batch size,
                    K is the number of examples, and 1 is the scalar output.
            """
            # input validation
            B, K, D = xs.shape
            assert K <= self.max_examples, \
                f"too many examples for model {K} > {self.max_examples}. Shape: {xs.shape} and {ys.shape}"
            assert D == self.task_size, \
                f"incorrect input size for model {D} != {self.task_size}. Shape: {xs.shape} and {ys.shape}"
            B_, K_, _1 = ys.shape
            assert B == B_, f"batch size mismatch b/w xs:{B} and ys:{B_}. Shape: {xs.shape} and {ys.shape}"
            assert K == K_, f"num_examples mismatch b/w xs:{K} and ys:{K_}. Shape: {xs.shape} and {ys.shape}"
            assert _1 == 1, f"ys should be scalars. Shape: {ys.shape}"

            # encode examples as token sequence
            toks = to_token_sequence(xs, ys)

            # run dtransformer to predict next tokens
            toks_pred = self.token_sequence_transformer(toks)
            # decode y predictions from next-token-prediction
            ys_pred = from_predicted_token_sequence(toks_pred)
            return ys_pred
    
    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)



def to_token_sequence(xs, ys):
    """
    Convert a regression data set into a sequence of vector tokens using a
    concatenating / joint encoding.

    Parameters:

    * `xs : tensor(B, K, D)`
        batch of `B` sequences of `K` input vectors of `D` dims.
    * `ys : tensor(B, K, 1)`
        batch of `B` sequences of `K` output scalars.

    Returns:

    * `toks : tensor(B, 2K, D+1)`
        batch of `B` sequences of `2K` alternating encoded inputs and
        outputs.

    Examples:

    ```
    > xs = [ [1,2,3], [2,3,4], [6,5,4], ]
    > ys = [ 1,       2,       6,       ] # y = x * [ 1, 0, 0, ]
    > to_token_sequence(xs, ys)
      (error! wrong! inputs not batched! and ys not singletons!)

    > xs = [ [ [1,2,3], [2,3,4], [6,5,4], ] ]
    > ys = [ [ [1],     [2],     [6],     ] ]
    > to_token_sequence(xs, ys)
      [ [ [ 0, 1, 2, 3, ]       # 0, first x
        , [ 1, 0, 0, 0, ]       # first y, 0s
        , [ 0, 2, 3, 4, ]       # 0, second x
        , [ 2, 0, 0, 0, ]       # ...
        , [ 0, 6, 5, 4, ]
        , [ 6, 0, 0, 0, ]
        ] ]
    ```
    """
    B, K, D = xs.shape
    # convert to input-output pairs (of the form [ 0 x1 .. xD  y 0 .. 0 ])
    xys = torch.cat([
        torch.zeros_like(ys),   #   B K 1
        xs,                     # | B K D
        ys,                     # | B K 1
        torch.zeros_like(xs),   # | B K D
    ], dim=-1)                  # -> B K 2D+2
    # convert to token sequences (alternating [ 0 x1 .. xD ], [ y 0 .. 0 ])
    toks = xys.reshape(B, 2*K, D+1)
    return toks


def from_predicted_token_sequence(toks, return_xs=False):
    """
    Convert a sequence of vector next-token-predictions into a regression
    data set by decoding from a joint/concatenating encoding.
    
    Parameters:

    * `toks : tensor(B, 2K, D+1)`
        batch of `B` sequences of `2K` alternating encoded next token
        predictions from inputs and outputs.
    * `return_xs=False : bool`
        whether to decode and return xs, or just ys (default).

    Returns:

    * `ys : tensor(B, K, 1)`
        batch of `B` sequences of `K` output scalars.
    * `xs : tensor(B, K, D)`
        (only included if `return_xs=True`)
        batch of `B` sequences of `K` input vectors of `D` dims.

    Note: This should NOT be used as an inverse of `to_token_sequence`. For
    that, use `from_token_sequence`.
    """
    # ys: head of every even-indexed token for every sequence in batch
    ys = toks[:, 0::2, :1]
    if not return_xs:
        return ys
    else:
        # xs: tail of every odd-indexed token for every sequence in batch
        xs = toks[:, 1::2, 1:]
        return ys, xs


def from_token_sequence(toks):
    """
    Inverse of `to_token_sequence`. Convert a sequence of vector tokens into
    a regression data set by decoding from a concatenating / joint encoding.
    
    Parameters:

    * `toks : tensor(B, 2K, D+1)`
        batch of `B` sequences of `2K` alternating encoded inputs and
        outputs.

    Returns:

    * `xs : tensor(B, K, D)`
        batch of `B` sequences of `K` input vectors of `D` dims.
    * `ys : tensor(B, K, 1)`
        batch of `B` sequences of `K` output scalars.

    Note: This should NOT be used on the output of a transformer trained
    towards next-token prediction, since for such a transformer actually the
    ys are the result of predicting for the x tokens and vise versa. For that
    setting, use `from_predicted_token_sequence`.

    Example:
    
    ```
    > toks = [ [ [ 0, 1, 2, 3, ]       # 0, first x
               , [ 1, 0, 0, 0, ]       # first y, 0s
               , [ 0, 2, 3, 4, ]       # 0, second x
               , [ 2, 0, 0, 0, ]       # ...
               , [ 0, 6, 5, 4, ]
               , [ 6, 0, 0, 0, ]
               ] ]
    > xs, ys = from_token_sequence(toks)
    > xs
      [ [ [1,2,3], [2,3,4], [6,5,4], ] ]
    > ys
      [ [ [1],     [2],     [6],     ] ]
    """
    # xs: tail of every even-indexed token for every sequence in batch
    xs = toks[:, 0::2, 1:]
    # ys: head of every odd-indexed token for every sequence in batch
    ys = toks[:, 1::2, :1]
    return xs, ys

