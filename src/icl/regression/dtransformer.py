"""
Decode-only transformer module.

Resources:

* Code roughly following Karpathy's tutorial and nanoGPT implementation, but
  with some features removed (such as dropout) and with some micro-
  optimisations here and there.
* See also Phuong and Hutter's 'Formal algorithms for transformers', though
  this implementation differs in a few places e.g. using Pre-layer-norm
  rather than Post-layer-norm.

Notes:

* Takes vector tokens (rather than token indices) as input and output.
  So for language this would need to be one-hot encoded.
  * TODO: Embedding had different initalisation compared to Linear, namely
    N(0,1) rather than Uniform---should I care?
"""


import torch
import torch.nn as nn
import torch.nn.functional as fn


class DTransformer(nn.Module):
    def __init__(
        self,
        token_size,
        max_tokens,
        embed_size,
        mlp_size,
        num_heads,
        num_layers,
        device='cpu',
        layer_norm=True,
        include_output=False,
    ):
        super().__init__()
        self.token_embedding = nn.Linear(
            in_features=token_size,
            out_features=embed_size,
            bias=False,
            device=device,
        )
        self.postn_embedding = nn.Linear(
            in_features=max_tokens,
            out_features=embed_size,
            bias=False,
            device=device,
        )
        self.blocks = nn.ModuleList([
            MultiHeadedCausalSelfAttentionTransformerBlock(
                embed_size=embed_size,
                mlp_size=mlp_size,
                max_tokens=max_tokens,
                num_heads=num_heads,
                device=device,
                layer_norm=layer_norm,
                include_output=include_output,
            )
            for _ in range(num_layers)
        ])
        # unembedding

        if layer_norm:
            self.unembedding = nn.Sequential(
                nn.LayerNorm(
                    normalized_shape=embed_size,
                    device=device,
                ),
                nn.Linear(
                    in_features=embed_size,
                    out_features=token_size,
                    device=device,
                ),
            )
        else:
            self.unembedding = nn.Linear(
                in_features=embed_size,
                out_features=token_size,
                device=device,
            )

        self.max_tokens = max_tokens
        

    def forward(self, toks):
        # if mechinterp=True, write attention patterns on forward pass to CSV
        # all other forward passes inherit mechinterp from this initial one

        _B, T, _V = toks.shape
        assert T <= self.max_tokens, f"too many tokens! {T} > {self.max_tokens}"

        # semantic and positional token embeddings
        x_positions = self.postn_embedding.weight.T[:T, :] # Tmax C ->   T C
        x_semantics = self.token_embedding(toks)    # B T V @ . V C -> B T C
        x = x_semantics + x_positions               # B T C + . T C -> B T C

        # apply the num_layers layers / attention blocks in sequence
        for block in self.blocks:
            x = x + block(x) # B T C + B T C -> B T C

        # unembedding: transform back to predicted next tokens
        y = self.unembedding(x)                     # B T C @ . C V -> B T V
        
        return y

        # NOTE:
        # during training,  we only care about y[:, :-1, :]...
        # during inference, we only care about y[:, -1:, :]...
        # TODO: optimise!
        # (moreover in the in-context regression setting, we really only care
        # about every second token prediction to begin with...)


class MultiHeadedCausalSelfAttentionTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        mlp_size,
        max_tokens,
        num_heads,
        device='cpu',
        layer_norm=True,
        include_output=False,
    ):
        super().__init__()
        self.attention = MultiHeadedCausalSelfAttention(
            embed_size=embed_size,
            max_tokens=max_tokens,
            num_heads=num_heads,
            include_output=include_output,
            device=device,
        )
        self.compute = nn.Sequential(
            nn.Linear(embed_size, mlp_size, device=device),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_size, device=device),
        )

        if layer_norm:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(normalized_shape=embed_size, device=device)
                for _ in ('before-attention', 'before-compute')
            ])
        else:
            self.layer_norms = nn.ModuleList([nn.Identity() for _ in range(2)])

        self.resid_after_attn = nn.Identity()

    def forward(self, x):
        # B, T, C = x.shape
        x = x + self.attention(self.layer_norms[0](x))
        self.resid_after_attn(x)
        x = x + self.compute(self.layer_norms[1](x))
        return x
            

class MultiHeadedCausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_size,
        max_tokens,
        num_heads,
        device='cpu',
        include_output=False,
    ):
        super().__init__()
        # validate dimensions
        if embed_size % num_heads:
            raise ValueError("num_heads must divide embed_size")
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        # batched key/query/value projections
        self.attention = nn.Linear(
            in_features=embed_size,
            out_features=3*embed_size,
            bias=False,
            device=device,
        )
        self.attention_softmax = nn.Softmax(dim=-1)

        # precompute causal mask
        mask_shape = (max_tokens, max_tokens)
        causal_mask = torch.log(torch.tril(torch.ones(mask_shape, device=device)))
        self.register_buffer('causal_mask', causal_mask)
        # precompute attention normalisation factor
        self.attention_scale = self.head_size ** 0.5

        self.include_output = include_output

        if self.include_output:
            self.output = nn.Linear(
                in_features=embed_size,
                out_features=embed_size,
                bias=False,
                device=device,
            )

    def forward(self, x):
        # unpack dimensions
        B, T, C = x.size()  # batch size, num_tokens, embed_size
        H = self.num_heads  # num_heads
        c = self.head_size  # head size

        # perform Q, K, V transforms, all at once
        Q, K, V = (self.attention(x)    # B T C @ C 3C  -> B T 3C
                .view(B, T, H, 3*c)     #               -> B T H 3c
                .transpose(-2, -3)      #               -> B H T 3c
                .split(c, dim=-1)       #               -> (B H T c) * 3
            )
        # now Q, K, V are each of shape (B, H, T, c)

        # compute affinities, scaled and with causal mask
        A = Q @ K.transpose(-2, -1)     # B H T c @ B H c T -> B H T T
        A = A / self.attention_scale    # B H T T / . . . T -> B H T T
        A = A + self.causal_mask[:T,:T] # B H T T + . . T T -> B H T T

        # convert affinities to mixing weights and mix value vectors
        p = self.attention_softmax(A)
        y = p @ V                   # B H T T @ B H T c -> B H T c

        # recombine / concatenate heads into new embedding
        y = (y                      #    B H T c
            .transpose(-3, -2)  # -> B T H c
            .contiguous()       # -> (make underlying memory match view)
            .view(B, T, C)      # -> B T C
        )

        if self.include_output:
            y = self.output(y)
        
        return y

    @property
    def qkv(self):
        return (
            self.attention(torch.eye(self.embed_size, device=self.attention.weight.device))
            .view(self.num_heads, 3 * self.head_size)
            .split(self.head_size, dim=-1)
        )
    