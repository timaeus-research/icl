from contextlib import contextmanager

import pytest
import torch
from torch import nn
from torch.optim import Adam

from icl.utils import get_device, temp_to, to


def test_get_device():
    tensor_on_cpu = torch.Tensor([1, 2, 3])
    tensor_on_cuda = torch.Tensor([1, 2, 3]).to("mps")

    assert str(get_device(tensor_on_cpu)) == "cpu"
    assert str(get_device(tensor_on_cuda)) == "mps:0"

def test_to():
    tensor_on_cpu = torch.Tensor([1, 2, 3])
    tensor_on_cuda = to(tensor_on_cpu, "mps")

    assert str(tensor_on_cuda.device) == "mps:0"

def test_temp_to():
    model = nn.Linear(10, 10).to("mps")
    optimizer = Adam(model.parameters())

    # Create some state in the optimizer
    input_tensor = torch.randn(32, 10, device="mps")
    target_tensor = torch.randn(32, 10, device="mps")
    output = model(input_tensor)
    loss = ((output - target_tensor) ** 2).mean()
    loss.backward()
    optimizer.step()

    # Capture the initial state_dict of the optimizer
    initial_state_dict = optimizer.state_dict()

    # Verify device changes
    original_device = get_device(initial_state_dict)
    
    with temp_to(initial_state_dict, 'cpu'):
        assert str(get_device(initial_state_dict)) == 'cpu'
        
    assert get_device(initial_state_dict) == original_device
