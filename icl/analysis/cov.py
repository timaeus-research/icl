import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch import nn


def get_weights(model, paths):
    for path in paths:
        full_path = path.split(".")
        layer = model

        for p in full_path:
            layer = getattr(layer, p)

        yield layer.weight.view((-1,))

        if layer.bias is not None:
            yield layer.bias.view((-1,))
 
 

class CovarianceCallback:
    def __init__(self, num_weights: int, device = "cpu", paths: List[str] = [], num_evals=3):
        self.first_moment = torch.zeros(num_weights, device=device)
        self.second_moment = torch.zeros(num_weights, num_weights, device=device)
        self.num_draws = 0
        self.paths = paths
        self.num_evals = num_evals

    def share_memory_(self):
        self.first_moment.share_memory_()
        self.second_moment.share_memory_()
        return self

    def accumulate(self, model: nn.Module):
        weights = torch.cat([w.view((-1,)) for w in get_weights(model, self.paths)])

        self.first_moment += weights
        self.second_moment += torch.outer(weights, weights)

        self.num_draws += 1

    def finalize(self):
        self.first_moment /= self.num_draws
        self.second_moment /= self.num_draws

    def reset(self):
        self.first_moment.zero_()
        self.second_moment.zero_()
        self.num_draws = 0

    def __call__(self, model):
        self.accumulate(model)

    def to_matrix(self):
        return self.second_moment - torch.outer(self.first_moment, self.first_moment)

    def sample(self):
        cov = self.to_matrix().detach().cpu().numpy()
        evals, evecs = eigsh(cov, k=self.num_evals, which='LM')
        return {
            "cov/matrix": cov,
            **{f"cov/eval/{i}": evals[i] for i in range(self.num_evals)},
            **{f"cov/evec/{i}": evecs[:, i] for i in range(self.num_evals)},
        }


class WithinHeadCovarianceCallback:
    def __init__(self, head_size: int, embed_size: int, num_heads: int, device = "cpu", paths: List[str] = [], num_evals=3):
        self.head_size = head_size
        self.embed_dim = embed_size
        self.num_layers = num_layers = len(paths)
        self.num_heads = num_heads

        self.first_moment = torch.zeros(num_layers, num_heads, self.num_weights_per_head, device=device)
        self.second_moment = torch.zeros(num_layers, num_heads, self.num_weights_per_head, self.num_weights_per_head, device=device)
        self.num_draws = 0
        self.paths = paths
        self.num_evals = num_evals

    def share_memory_(self):
        self.first_moment.share_memory_()
        self.second_moment.share_memory_()
        return self

    def accumulate(self, model: nn.Module):
        for l, path in enumerate(self.paths):
            attn = next(get_weights(model, [path])).flatten() \
                .view((self.embed_dim, self.num_heads, self.head_size * 3))

            # print(attn.shape, self.first_moment.shape, self.second_moment.shape)
            for h in range(self.num_heads):
                head = attn[:, h, :].flatten()
                self.first_moment[l, h] += head
                self.second_moment[l, h] += torch.outer(head, head)

        self.num_draws += 1

    def finalize(self):
        self.first_moment /= self.num_draws
        self.second_moment /= self.num_draws

    def reset(self):
        self.first_moment.zero_()
        self.second_moment.zero_()
        self.num_draws = 0

    def __call__(self, model):
        self.accumulate(model)

    def to_matrix(self):
        covariance = self.second_moment

        for l in range(self.num_layers):
            for h in range(self.num_heads):
                first_moment_head = self.first_moment[l, h]
                covariance[l, h] -= torch.outer(first_moment_head, first_moment_head)

        return covariance

    def sample(self):
        cov = self.to_matrix().detach().cpu().numpy()
        results = {}

        # Restructure the eigenvectors to be in the shape of the attention layer
        evals = np.zeros((self.num_evals, self.num_layers, self.num_heads))
        evecs = np.zeros((self.num_evals, self.num_layers, self.num_heads, self.embed_dim, self.head_size * 3))

        for l in range(self.num_layers):
            for h in range(self.num_heads):
                head_cov = cov[l, h]
                head_evals, head_evecs = eigsh(head_cov, k=self.num_evals, which='LM')

                for i in  range(self.num_evals):
                    evecs[i,l,h,:,:] = head_evecs[:, i].reshape((self.embed_dim, self.head_size * 3))
                    evals[i,l,h] = head_evals[i]

        results.update({
            f"cov/evecs": evecs,
            f"cov/evals": evals
        })

        return results

    @property
    def num_weights_per_head(self):
        return 3 * self.head_size * self.embed_dim

    @property
    def num_weights_per_layer(self):
        return self.num_heads * self.num_weights_per_head

    @property
    def num_weights(self):
        return self.num_layers * self.num_weights_per_layer
    


LayerWeightsAccessor = Callable[[nn.Module], torch.Tensor]

class WithinLayerCovarianceAccumulator:
    """
    A CovarianceAccumulator to compute covariance between arbitrary layers.
    For use with `estimate`.

    Attributes:
        num_heads (int): The number of attention heads.
        num_weights_per_head (int): The number of weights per attention head.
        accessors (List[LayerWeightsAccessor]): Functions to access attention head weights.
        num_layers (int): The number of layers (= number of accessors).
        num_weights_per_layer (int): The number of weights per layer.
        num_weights (int): The total number of weights.
    """
    def __init__(self, model, device = "cpu", num_evals=3, **accessors: LayerWeightsAccessor):
        self.num_layers = len(accessors)
        self.accessors = accessors
        self.num_weights_per_layer = {name: len(accessor(model).flatten()) for name, accessor in accessors.items()}
        self.first_moments = {name: torch.zeros(num_weights, device=device) for name, num_weights in self.num_weights_per_layer.items()}
        self.second_moments = {name: torch.zeros(num_weights, num_weights, device=device) for name, num_weights in self.num_weights_per_layer.items()}
        self.num_draws = 0
        self.num_evals = num_evals
        self.is_finished = False

    def share_memory_(self):
        if str(self.device) == "mps":
            warnings.warn("Cannot share memory with MPS device. Ignoring.")
            return self

        for name in self.accessors:
            self.first_moments[name].share_memory_()
            self.second_moments[name].share_memory_()

        return self

    @property
    def num_weights(self):
        """The total number of weights."""
        return sum(self.num_weights_per_layer.values())

    def accumulate(self, model: nn.Module):
        """Accumulate moments from model weights."""
        assert not self.is_finished, "Cannot accumulate after finalizing."

        for name, accessor in self.accessors.items():
            weights = accessor(model).flatten()
            self.first_moments[name] += weights
            self.second_moments[name] += torch.outer(weights, weights)

        self.num_draws += 1

    def finalize(self):
        """Finalize the moments by dividing by the number of draws."""

        for name in self.accessors:
            self.first_moments[name] /= self.num_draws
            self.second_moments[name] /= self.num_draws
            self.is_finished = True

    def reset(self):
        """Reset the accumulator."""
        for name in self.accessors:
            self.first_moments[name].zero_()
            self.second_moments[name].zero_()

        self.num_draws = 0
        self.is_finished = False

    def to_matrices(self):
        """Convert the moments to a covariance matrix."""

        covariances = {}

        for name in self.accessors:
            first_moment = self.first_moments[name]
            covariances[name] = self.second_moments[name] - torch.outer(first_moment, first_moment)

        return covariances

    def to_eigens(self, include_matrix=False):
        """Convert the covariance matrix to pairs of eigenvalues and vectors."""
        covariances = {k: v.detach().cpu().numpy() for k, v in self.to_matrices().items()}
        results = {}

        for name, cov in covariances.items():
            evals, evecs = eigsh(cov, k=self.num_evals, which='LM')

            # Reverse the order of the eigenvalues and vectors
            evals = evals[::-1]
            evecs = evecs[:, ::-1]

            results.update({
                name: {
                    "evecs": evecs,
                    "evals": evals
                }
            })

            if include_matrix:
                results[name]["matrix"] = cov

        return results

    def __call__(self, model):
        self.accumulate(model)


class BetweenLayerCovarianceAccumulator:
    """
    A CovarianceAccumulator to compute covariance between arbitrary layers.
    For use with `estimate`.
    """
    def __init__(self, model, pairs: Dict[str, Tuple[str, str]], device = "cpu", num_evals=3, **accessors: Dict[str, LayerWeightsAccessor]):
        self.num_layers = len(accessors)
        self.accessors = accessors
        self.pairs = pairs
        self.num_weights_per_layer = {name: len(accessor(model).flatten()) for name, accessor in accessors.items()}
        self.first_moments = {name: torch.zeros(num_weights, device=device) for name, num_weights in self.num_weights_per_layer.items()}
        self.second_moments = {pair_name: torch.zeros(self.num_weights_per_layer[name1], self.num_weights_per_layer[name2], device=device) for pair_name, (name1, name2) in pairs.items()}
        self.num_draws = 0
        self.num_evals = num_evals
        self.is_finished = False
        self.device = device

    def share_memory_(self):
        if str(self.device) == "mps":
            warnings.warn("Cannot share memory with MPS device. Ignoring.")
            return self

        for name in self.accessors:
            self.first_moments[name].share_memory_()
            self.second_moments[name].share_memory_()

        return self

    @property
    def num_weights(self):
        """The total number of weights."""
        return sum(self.num_weights_per_layer.values())

    def accumulate(self, model: nn.Module):
        """Accumulate moments from model weights."""
        assert not self.is_finished, "Cannot accumulate after finalizing."
        weights = {name: accessor(model).flatten() for name, accessor in self.accessors.items()}

        for name, w in weights.items():
            self.first_moments[name] += w

        for pair_name, (name1, name2) in self.pairs.items():
            self.second_moments[pair_name] += torch.outer(weights[name1], weights[name2])

        self.num_draws += 1

    def finalize(self):
        """Finalize the moments by dividing by the number of draws."""

        for name in self.accessors:
            self.first_moments[name] /= self.num_draws
        
        for name in self.pairs:
            self.second_moments[name] /= self.num_draws
        
        self.is_finished = True

    def reset(self):
        """Reset the accumulator."""
        for name in self.accessors:
            self.first_moments[name].zero_()

        for name in self.pairs:
            self.second_moments[name].zero_()

        self.num_draws = 0
        self.is_finished = False
    
    def to_matrices(self):
        """Convert the moments to a covariance matrix."""
        covariances = {}

        for name, (layer1, layer2) in self.pairs.items():
            first_moment1 = self.first_moments[layer1]
            first_moment2 = self.first_moments[layer2]
            covariances[name] = self.second_moments[name] - torch.outer(first_moment1, first_moment2)

        return covariances

    def to_eigens(self, include_matrix=False):
        """Convert the covariance matrix to pairs of eigenvalues and vectors."""
        covariances = {k: v.detach().cpu().numpy() for k, v in self.to_matrices().items()}
        results = {}

        for name, cov in covariances.items():
            evals, evecs = eigsh(cov, k=self.num_evals, which='LM')

            # Reverse the order of the eigenvalues and vectors
            evals = evals[::-1]
            evecs = evecs[:, ::-1]

            results.update({
                name: {
                    "evecs": evecs,
                    "evals": evals
                }
            })

            if include_matrix:
                results[name]["matrix"] = cov

        return results

    def __call__(self, model):
        self.accumulate(model)


def make_layer_accessor(path: str):
    def accessor(model: nn.Module) -> torch.Tensor:
        layer = model
        for name in path.split("."):
            layer = getattr(layer, name)

        return layer.weight

    return accessor


# TODO: Check that this yields the correct weights
def make_head_accessor(attn_layer: str, head: int, num_heads=2, embed_dim=4, head_size: Optional[int] = None):
    """Make a function to access the weights of a particular attention head."""
    head_size = embed_dim // num_heads if head_size is None else head_size
    num_head_params = 3 * head_size * embed_dim

    def accessor(model: nn.Module) -> torch.Tensor:
        return make_layer_accessor(attn_layer)(model)[head * num_head_params: (head + 1) * num_head_params]

    return accessor


def _head_loc(l: int, h: int):
    return f"{1 + l}.1.{h}:block_{l}/head_{h}"


def _ln_loc(l: int, i: int):
    return f"{1 + l}.{2 * i}:block_{l}/ln_{i}"


def _mlp_loc(l: int, i: int):
    return f"{1 + l}.3.{i}:block_{l}/mlp_{i}"


def make_transformer_accessors(L=2, H=4):
    assert L < 10, "L must be less than 10"

    accessors = {
        _head_loc(l, h): make_head_accessor(f"token_sequence_transformer.blocks.{l}.attention.attention", h)
        for l in range(L) for h in range(H)
    }

    accessors.update({
        _mlp_loc(l, i): make_layer_accessor(f"token_sequence_transformer.blocks.{l}.compute.{2 * i}")
        for l in range(L) for i in range(2)
    })

    accessors.update({
        _ln_loc(l, i): make_layer_accessor(f"token_sequence_transformer.blocks.{l}.layer_norms.{i}")
        for i in range(2) for l in range(L)
    })

    accessors.update({
        "0.0:embed/token": make_layer_accessor("token_sequence_transformer.token_embedding"),
        "0.1:embed/pos": make_layer_accessor("token_sequence_transformer.postn_embedding"),
        f"{1+L}.0:unembed/ln": make_layer_accessor("token_sequence_transformer.unembedding.0"),
        f"{1+L}.1:unembed/linear": make_layer_accessor("token_sequence_transformer.unembedding.1"),
    })

    # Sort
    accessors = {k: v for k, v in sorted(accessors.items(), key=lambda x: x[0])}
    
    return accessors


def make_transformer_accessors_and_interactions(L=2, H=4):
    accessors = make_transformer_accessors(L, H)

    # We want all within-layer covariances
    pairs = {key: (key, key) for key in accessors.keys()}

    # We want between-head covariances for successive layers
    if L >= 2:
        for l in range(0, L-1):
            for h1 in range(H):
                for h2 in range(H):
                    pairs[f"{_head_loc(l, h1)}-{_head_loc(l+1, h2)}"] = (_head_loc(l, h1), _head_loc(l+1, h2))

    # Let's check between mlp covariances within a single block
    for l in range(L):
        pairs[f"{_mlp_loc(l, 0)}-{_mlp_loc(l, 1)}"] = (_mlp_loc(l, 0), _mlp_loc(l, 1))

    # And let's check between embeds and unembeds
    # pairs[f"0.0:embed/token-{1+L}.1:unembed/linear"] = ("0.0:embed/token", f"{1+L}.1:unembed/linear")

    return accessors, pairs


def make_transformer_cov_accumulator(model, device="cpu", num_evals=3):
    L = len(model.token_sequence_transformer.blocks)
    H = model.token_sequence_transformer.blocks[0].attention.num_heads

    accessors, pairs = make_transformer_accessors_and_interactions(L, H)

    return BetweenLayerCovarianceAccumulator(
        model,
        pairs,
        device,
        num_evals,
        **accessors
    )