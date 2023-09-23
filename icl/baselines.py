import torch

from icl.tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                       RegressionSequenceDistribution)

_DEFAULT_MAX_BYTES = 2**28 # 256 MiB
# chosen to fit comfortably in matt's GTX 1050 VRAM...
# fits, e.g., (B=2048 batch * K=16 context * M=2048 tasks)=2**26 float32s


def dmmse_predictor(
    xs,
    ys,
    prior: DiscreteTaskDistribution,
    noise_variance: float,
    return_ws_hat=False,
    _max_bytes=_DEFAULT_MAX_BYTES,
):
    """
    Return the Bayes-optimal predictions for each prefix of the data set
    `xs`, `ys` given discrete prior `prior` over tasks and known error model
    Gaussian with variance `noise_model`.

    Parameters:

    * `xs : tensor(B, K, D)`
        batch of `B` sequences of `K` input vectors of `D` dims.
    * `ys : tensor(B, K, 1)`
        batch of `B` sequences of `K` output scalars.
    * `prior : DiscreteTaskDistribution`
        contains the finite set of task vectors.
    * `noise_variance : float >= 0`
        known variance of the normally-distributed noise on the regression
        data.
    * `return_ws_hat=False : bool`
        whether to return Bayes-optimal model too (default: no).
    * `_max_bytes=2**28 : int`
        a cap on the size of the intermediate BxKxM matrix, above which the
        computation will be internally minibatched along the B dimension

    Returns:

    * `ys_pred : tensor(B, K, 1)`
        `ys_pred[b,k]` is the Bayes-optimal estimate of the output given
        input `xs[b,k]` having seen the examples `xs[b,:k]` and `ys[b,:k]`,
        assuming prior `prior` and known noise model `N(0, noise_variance)`.
    * `ws_hat : tensor(B, K, D)` (if `return_ws_hat` is `True`)
        `ws_hat[:, k]` is the Bayes-optimal solution vector given data
        `xs[:,:k]` and `ys[:,:k]` (note: not inclusive of index `k`).

    Notes:

    * implements paper formula (7).
    * `_max_bytes` assumes the inputs use float32 data type
    * please use `B`, `K`, `M`, and `_max_bytes` that are powers of 2,
      because I didn't generalise the code to work with weird batch sizes
    """
    # check all the dimensions of the inputs line up
    B, K, D = xs.shape
    M, D_   = prior.num_tasks, prior.task_size
    if D != D_:
        raise ValueError(f"dimension mismatch: data {D} != prior {D_}")
    # check the data device
    device = xs.device
    device_ = prior.tasks.device
    if device_ != device:
        raise ValueError(f"devices: task {device_} != data {device}")

    # B K M matrices may exceed (V)RAM: enforce a maximum batch size,
    b_max = _max_bytes // (K * M * 4) # 4 bytes per float32
    if b_max == 0:
        # K, M themselves too large, further minibatching would be needed
        raise ValueError("not enough bytes to fit even singleton batches")
    # above this maximum batch size, we will further minibatch the computation
    b = min(B, b_max)
    # below implementation assumes minibatch dim b divides batch dim B
    # (sufficient but not necessary: B, K, M, _max_bytes are all powers of 2)
    if B % b != 0:
        raise ValueError(f"minibatch size {b} doesn't divide batch size {B}")
    
    # compute the optimal solution vector per prefix (one minibatch at a time)
    ws_hat = torch.empty(B, K, D, device=device)
    minibatches_ids = torch.arange(B, device=device).view(-1, b)
    for idx in minibatches_ids:
        # compute w_hat for each k
        # TODO: micro-optimisation: k=0 -> mean task, skip softmax
        loss = torch.empty(b, K, M, device=device)
        for k in range(K): # unclear how to (or whether to) vectorise
            Xk = xs[idx, :k, :]                 # b K D, slice  -> b k D
            yk = ys[idx, :k, :]                 # b K 1, slice  -> b k 1
            # compute loss for each task in the prior
            yh = Xk @ prior.tasks.T             # b k D @ . D M -> b k M
            L  = (yk - yh).square()             # b k 1 - b k M -> b k M
            loss[:, k, :] = L.sum(axis=-2)      # b k M, reduce -> b M
        # average task with Boltzmann posterior at temperature 2sigma^2
        score = -loss / (2 * noise_variance)    # b K M / . . .     -> b K M
        probs = score.softmax(axis=-1)          # b K M, normalise  -> b K M
        ws_hat[idx] = probs @ prior.tasks       # b K M @ . M D     -> b K D

    # compute y_hat for each k
    ys_pred = (
            xs.view(B, K, 1, D)                       #    B K 1 D
          @ ws_hat.view(B, K, D, 1)                   #  @ B K D 1
        ).view(B, K, 1)                               # -> B K 1 1 -> B K 1
    
    # that's all!
    if return_ws_hat:
        return ys_pred, ws_hat
    else:
        return ys_pred


def ridge_predictor(xs, ys, noise_variance, return_ws_hat=False):
    """
    Return the Bayes-optimal predictions for each prefix of the data set
    `xs`, `ys` given standard normal prior over tasks and known error model
    Gaussian with variance `noise_model`.

    Parameters:

    * `xs : tensor(B, K, D)`
        batch of `B` sequences of `K` input vectors of `D` dims.
    * `ys : tensor(B, K, 1)`
        batch of `B` sequences of `K` output scalars.
    * `noise_variance : float >= 0`
        known variance of the normally-distributed noise on the regression
        data.
    * `return_ws_hat=False : bool`
        whether to return Bayes-optimal model too (default: no).

    Returns:

    * `ys_pred : tensor(B, K, 1)`
        `ys_pred[b,k]` is the Bayes-optimal estimate of the output given
        input `xs[b,k]` having seen the examples `xs[b,:k]` and `ys[b,:k]`,
        assuming prior `prior` and known noise model `N(0, noise_variance)`.
    * `ws_hat : tensor(B, K, D)` (if `return_ws_hat` is `True`)
        `ws_hat[:, k]` is the Bayes-optimal solution vector given data
        `xs[:,:k]` and `ys[:,:k]` (note: not inclusive of index `k`).

    Note: implements paper formula (8) which is the well-known ridge
    regression estimator.
    """
    B, K, D = xs.shape
    device = xs.device
    
    # compute w_hat for each k
    # TODO: micro-optimisation: the k=0 case is always wk=0, skip it?
    XTX = torch.empty(B, K, D, D, device=device)
    RHS = torch.empty(B, K, D, 1, device=device)
    for k in range(K): # unclear how to (or whether to) vectorise
        Xk  = xs[:, :k, :]                  # B K D         -> B k D
        yk  = ys[:, :k, :]                  # B K 1         -> B k 1
        # compute the normal equations for this k
        XkT = Xk.transpose(-2, -1)          # B k D         -> B D k
        XTX[:, k, :, :] = XkT @ Xk          # B D k @ B k D -> B D D
        RHS[:, k, :, :] = XkT @ yk          # B D k @ B k 1 -> B D 1
    # that's the last of the dependence on k, so, batch from here:
    # regularise the normal equations (making it ridge regression)
    reg = noise_variance * torch.eye(D, device=device)
    LHS = XTX + reg                         # B K D D + . . D D -> B K D D
    # solve the regularised normal equations
    # solve(LHS @ w == RHS) = inv(LHS) @ RHS
    # note: GPU syncs with CPU here, except on MPS (not implemented yet)
    ws_hat = torch.linalg.solve(LHS, RHS)   # BKDD^-1 @ BKD1 -> B K D 1
    ws_hat = ws_hat.view(B, K, D)           #                -> B K D
    
    # compute y_hat for each k
    ys_pred = (
            xs.view(B, K, 1, D)             #    B K 1 D
          @ ws_hat.view(B, K, D, 1)         #  @ B K D 1
        ).view(B, K, 1)                     # -> B K 1 1 -> B K 1
    
    # that's all!
    if return_ws_hat:
        return ys_pred, ws_hat
    else:
        return ys_pred

