from contextlib import contextmanager

import pyhessian
import torch
from pyhessian import hessian  # Hessian computation
from torch.nn import functional as F
from torch import nn

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs[0], inputs[1])
    

def torch_eig_wrapper(*args, eigenvectors=True, **kwargs):
    # Call the new torch.linalg.eig function
    eigenvalues, eigenvecs = torch.linalg.eig(*args, **kwargs)
    
    # Format the output to mimic the old torch.eig
    # torch.eig used to return a tensor with [real, imaginary] parts for eigenvalues
    # torch.linalg.eig returns a tensor of complex numbers for eigenvalues
    eigenvalues_real = eigenvalues.real
    eigenvalues_imag = eigenvalues.imag

    eigenvalues_combined = torch.stack((eigenvalues_real, eigenvalues_imag), dim=-1)
    return eigenvalues_combined, eigenvecs


# Monkey patch the torch.eig function in the pyhessian module
torch.eig = torch_eig_wrapper 


@contextmanager
def batch_hessian(model, xs, ys):
    device = xs.device
    xs = xs.to('cpu')
    ys = ys.to('cpu')
    model = ModelWrapper(model.to('cpu'))
    yield hessian(model, F.mse_loss, data=((xs, ys), ys), cuda=False)

    xs = xs.to(device)
    ys = ys.to(device)
    model = model.to(device)


