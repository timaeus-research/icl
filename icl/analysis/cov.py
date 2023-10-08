from typing import List

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch import nn

from icl.analysis.sample import get_weights


class CovarianceCallback:
    def __init__(self, num_weights: int, device = "cpu", paths: List[str] = [], num_evals=3):
        self.first_moment = torch.zeros(num_weights, device=device).share_memory_()
        self.second_moment = torch.zeros(num_weights, num_weights, device=device).share_memory_()
        self.num_draws = 0
        self.paths = paths
        self.num_evals = num_evals

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

        self.first_moment = torch.zeros(num_layers, num_heads, self.num_weights_per_head, device=device).share_memory_()
        self.second_moment = torch.zeros(num_layers, num_heads, self.num_weights_per_head, self.num_weights_per_head, device=device).share_memory_()
        self.num_draws = 0
        self.paths = paths
        self.num_evals = num_evals

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