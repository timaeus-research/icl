import pytest
import torch
import torch.nn as nn
from scipy.linalg import eigh
from torch.testing import assert_allclose

from icl.analysis.cov import BetweenLayerCovarianceAccumulator
from icl.model import InContextRegressionTransformer


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 10)


def test_dummy_model():
    model = DummyModel()
    accessors = {
        "fc1": lambda m: m.fc1.weight,
        "fc2": lambda m: m.fc2.weight
    }
    accumulator = BetweenLayerCovarianceAccumulator(model, {"pair": ("fc1", "fc2")}, "cpu", **accessors)

    first_moment_1 = torch.zeros(5 * 10)
    first_moment_2 = torch.zeros(5 * 10)

    second_moment = torch.zeros(5 * 10, 5 * 10)
    
    for _ in range(100):
        # Generate two sets of weights with the above covariance structure
        model.fc1.weight.data.random_(0, 2)
        model.fc2.weight.data.random_(0, 2)

        # Update the first moments
        first_moment_1 += model.fc1.weight.flatten()
        first_moment_2 += model.fc2.weight.flatten()

        # Update the second moments
        second_moment += torch.outer(model.fc1.weight.flatten(), model.fc2.weight.flatten())

        accumulator(model)

    first_moment_1 /= 100
    first_moment_2 /= 100
    second_moment /= 100

    accumulator.finalize()
    covariance = accumulator.to_matrices()["pair"]
    assert_allclose(covariance, second_moment - torch.outer(first_moment_1, first_moment_2), atol=0.5, rtol=0.9)


def test_reset():
    model = DummyModel()
    accessors = {
        "fc1": lambda m: m.fc1.weight,
        "fc2": lambda m: m.fc2.weight
    }
    accumulator = BetweenLayerCovarianceAccumulator(model, {"pair": ("fc1", "fc2")}, "cpu", **accessors)
    accumulator(model)
    accumulator.reset()
    assert accumulator.num_draws == 0
    assert not accumulator.is_finished
    assert torch.all(accumulator.first_moments["fc1"] == 0)
    assert torch.all(accumulator.second_moments["pair"] == 0)



def test_full_transformer():
    model = InContextRegressionTransformer()