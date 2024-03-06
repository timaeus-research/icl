
import numbers

import numpy as np
from scipy import special
from scipy.ndimage import correlate1d, gaussian_filter1d


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian kernel given the standard deviation, order, and radius.

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian kernel.
    order : int
        The order of the derivative of the Gaussian to compute.
    radius : int
        The radius of the kernel, which determines its size (2*radius + 1).

    Returns
    -------
    output : ndarray
        The coefficients of the 1-D Gaussian kernel.
    """
    if order < 0:
        raise ValueError("Order of derivative must be non-negative")
    p = np.polynomial.Polynomial([0, 0, -0.5 / (sigma ** 2)])
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(p(x), dtype=np.double)
    phi_x /= phi_x.sum()

    if order == 0:
        return phi_x
    elif order == 1:
        # First derivative of Gaussian
        q = np.polynomial.Polynomial([-1 / sigma ** 2, 0, 0])
        return q(x) * phi_x
    elif order == 2:
        # Second derivative of Gaussian
        q = np.polynomial.Polynomial([1 / sigma ** 2, -1 / sigma ** 2, 0, 0])
        return q(x) * phi_x
    else:
        # For higher order derivatives, use Hermitian polynomials
        q = np.polynomial.HermiteE([0] * (order + 1))
        q = q.deriv(order)
        q = q * (np.exp(-x**2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma ** (order + 1)))
        return q(x)
    

def gaussian_filter1d_variable_sigma(input, sigma, axis=-1, order=0, output=None,
                                     mode="reflect", cval=0.0, truncate=4.0):
    """1-D Gaussian filter with variable sigma.

    Parameters
    ----------
    input : array_like
        Input array to filter.
    sigma : scalar or sequence of scalar
        Standard deviation(s) for Gaussian kernel. If a sequence is provided,
        it must have the same length as the input array along the specified axis.
    axis : int, optional
        The axis of input along which to calculate. Default is -1.
    order : int, optional
        An order of 0 corresponds to convolution with a Gaussian kernel.
        A positive order corresponds to convolution with that derivative of a Gaussian.
    output : ndarray, optional
        Output array. Has the same shape as `input`.
    mode : {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}, optional
        The `mode` parameter determines how the input array is extended beyond its boundaries.
    cval : scalar, optional
        Value to fill past edges of input if `mode` is 'constant'.
    truncate : float, optional
        Truncate the filter at this many standard deviations.

    Returns
    -------
    output : ndarray
        The result of the 1D Gaussian filter with variable sigma.
    """

    if input.ndim == 0:
        raise ValueError("Input array should have at least one dimension")

    if np.isscalar(sigma):
        sigma = np.ones(input.shape[axis]) * sigma
    elif len(sigma) != input.shape[axis]:
        raise ValueError("Length of sigma must match the dimension of the input array along the specified axis.")

    # Move the specified axis to the front
    input = np.moveaxis(input, axis, 0)
    
    # Define the output array if not provided
    if output is None:
        output = np.zeros_like(input)

    # Iterate over each position along the specified axis
    for i in range(input.shape[0]):
        # Extract the local sigma value
        local_sigma = sigma[i]
        lw = int(truncate * local_sigma + 0.5)
        min_i = max(0, i - lw)
        max_i = min(input.shape[0], i + lw + 1)
        # Generate the local weights for the Gaussian kernel
        output[i] = gaussian_filter1d(input[min_i:max_i], local_sigma, axis=axis, order=order, mode=mode, cval=cval, truncate=truncate)[i - min_i]
        # Apply the local filter

    # Move the axis back to its original position
    output = np.moveaxis(output, 0, axis)

    return output




# left_pad = int(sigma[0] * truncate)
# right_pad = int(sigma[-1] * truncate + 0.5)

# kwargs = {}
# if mode == "constant":
#     kwargs["constant_values"] = cval
    
# padded_input = np.pad(input, [(int(left_pad), int(right_pad)), *([(0, 0)] * (input.ndim - 1))], mode=mode, **kwargs)

# # Iterate over each position along the specified axis
# for i in range(input.shape[0]):
#     # Extract the local sigma value
#     local_sigma = sigma[i]
#     lw = int(truncate * local_sigma + 0.5)
    
#     I = i + left_pad
#     min_i = I - lw
#     max_i = I + lw + 1
#     # Generate the local weights for the Gaussian kernel
#     weights = _gaussian_kernel1d(local_sigma, order, lw)
#     output[i] = weights.T @ padded_input[min_i:max_i]
#     # output[i] = gaussian_filter1d(input[min_i:max_i], local_sigma, axis=0, order=order, mode=mode, cval=cval, truncate=truncate)[i - min_i]
#     # Apply the local filter
#     # correlate1d(input.take(indices=i, axis=axis), weights, axis=axis, output=output.take(indices=i, axis=axis), mode=mode, cval=cval)

# # Move the axis back to its original position
# output = np.moveaxis(output, 0, axis)