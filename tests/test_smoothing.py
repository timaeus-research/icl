import numpy as np
from scipy.ndimage import gaussian_filter1d

from icl.analysis.smoothing import (_gaussian_kernel1d,
                                    gaussian_filter1d_variable_sigma)


def test_gaussian_kernel_sum():
    kernel = _gaussian_kernel1d(sigma=1.0, order=0, radius=5)
    assert np.isclose(kernel.sum(), 1), "Kernel sum should be close to 1 for normalization"

def test_gaussian_kernel_symmetry():
    kernel = _gaussian_kernel1d(sigma=1.0, order=0, radius=5)
    assert np.allclose(kernel, kernel[::-1]), "Kernel should be symmetric for order 0"

def test_gaussian_kernel_negative_order():
    try:
        _gaussian_kernel1d(sigma=1.0, order=-1, radius=5)
        assert False, "Should raise ValueError for negative order"
    except ValueError:
        assert True

def test_gaussian_filter1d_variable_sigma_length_mismatch():
    input_array = np.random.rand(10)
    sigma = np.random.rand(5)  # Length mismatch
    try:
        gaussian_filter1d_variable_sigma(input_array, sigma)
        assert False, "Should raise ValueError due to sigma length mismatch"
    except ValueError:
        assert True

def test_gaussian_filter1d_variable_sigma_constant_sigma():
    input_array = np.random.rand(20).reshape((2, 10))
    sigma_value = 0.5
    output_var_sigma = gaussian_filter1d_variable_sigma(input_array, sigma_value, mode='constant')
    output_const_sigma = gaussian_filter1d(input_array, sigma_value, mode='constant')
    assert np.allclose(output_var_sigma, output_const_sigma), "Output should be similar for constant sigma"
