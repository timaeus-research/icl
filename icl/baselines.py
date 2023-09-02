import torch

from icl.tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                       RegressionSequenceDistribution)


def dmmse_predictor(xs, ys, prior: DiscreteTaskDistribution, noise_variance: float, return_ws_hat=False):
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

    Returns:

    * `ys_pred : tensor(B, K, 1)`
        `ys_pred[b,k]` is the Bayes-optimal estimate of the output given
        input `xs[b,k]` having seen the examples `xs[b,:k]` and `ys[b,:k]`,
        assuming prior `prior` and known noise model `N(0, noise_variance)`.
    * `ws_hat : tensor(B, K, D)` (if `return_ws_hat` is `True`)
        `ws_hat[:, k]` is the Bayes-optimal solution vector given data
        `xs[:,:k]` and `ys[:,:k]` (note: not inclusive of index `k`).

    Note: implements paper formula (7).
    """
    B, K, D = xs.shape
    M, D_   = prior.num_tasks, prior.task_size
    if D != D_:
        raise ValueError(f"dimension mismatch: data {D} != prior {D_}")
    device = xs.device
    device_ = prior.device

    if device_ != device and not (str(device_).startswith(str(device)) or str(device).startswith(str(device_))):
        raise ValueError(f"devices: task {device_} != data {device}")

    ws_hat = torch.zeros(B, K, D, device=device)

    # loop over minibatches of tasks (we can't fit them all in memory for large M)
    batch_size = 2048
    for m1 in range(0, M, batch_size):
        m2 = min(m1 + batch_size, M)
        tasks = torch.stack([prior.sample_task(m) for m in range(m1, m2)])

        # compute w_hat for each k
        # TODO: micro-optimisation: k=0 -> mean task, skip softmax?
        loss = torch.empty(B, K, M, device=device)
        for k in range(K): # unclear how to (or whether to) vectorise
            Xk = xs[:, :k, :]                   # B K D, slice  -> B k D
            yk = ys[:, :k, :]                   # B K 1, slice  -> B k 1
            # compute loss for each task in the prior
            yh = Xk @ tasks.T             # B k D @ . Dm -> B k m
            L  = (yk - yh).square()             # B k 1 - B k m -> B k m
            loss[:, k, :] = L.sum(axis=-2)      # B k m, reduce -> B m
        # average task with Boltzmann posterior at temperature 2sigma^2
        score  = -loss / (2*noise_variance) # B K m / . . .     -> B K m
        probs  = score.softmax(axis=-1)     # B K m, normalise  -> B K m

        ws_hat += probs @ tasks        # B K m @ . m D     -> B K D
    
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

