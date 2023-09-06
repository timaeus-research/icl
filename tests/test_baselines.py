import torch
import numpy as np
import torch_testing as tt

from icl.tasks import DiscreteTaskDistribution, GaussianTaskDistribution
from icl.tasks import RegressionSequenceDistribution
from icl.baselines import dmmse_predictor, ridge_predictor


def test_dmmse_predictor_first_is_uniform():
    B, K, D, M, V = 256, 1, 4, 16, .25
    T = DiscreteTaskDistribution(task_size=D, num_tasks=M)
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    ws = T.tasks
    _, ws_hat = dmmse_predictor(
        xs,
        ys,
        prior=T,
        noise_variance=V,
        return_ws_hat=True,
    )
    # now for K=1 there is no context for the first predictor so it's just
    # going to be the uniform average over all the tasks
    tt.assert_allclose(ws_hat, ws.mean(axis=0).expand(B, K, D))


def test_dmmse_predictor_python_loops():
    B, K, D, M, V = 256, 2, 4, 8, .25
    T = DiscreteTaskDistribution(task_size=D, num_tasks=M)
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    ws = T.tasks
    _, ws_hat = dmmse_predictor(
        xs,
        ys,
        prior=T,
        noise_variance=V,
        return_ws_hat=True,
    )
    # now do a straight-forward non-vectorised implementation of eqn 7 and
    # compare
    for b in range(B):
        for k in range(K):
            # compute denominator
            denom = 0.
            for l in range(M):
                # compute inner sum
                # note: there is no need for k-1 in the loop guard because
                # we are counting k in range(K) from 0, this k is actually
                # the 'number of contextual examples to consider' (0, 1, ...,
                # k-1)
                loss_sofar = 0.
                for j in range(k):
                    loss_sofar += (ys[b,j,0] - ws[l] @ xs[b,j]).square()
                denom += np.exp(-1/(2*V) * loss_sofar)
            # compute numerator
            numer = torch.zeros(D)
            for i in range(M):
                # compute inner sum
                loss_sofar = 0.
                for j in range(k):
                    loss_sofar += (ys[b,j,0] - ws[i] @ xs[b,j]).square()
                numer += np.exp(-1/(2*V) * loss_sofar) * ws[i]
            # combine
            wk_hat_expected = numer / denom
            # check
            wk_hat_actual = ws_hat[b, k]
            tt.assert_allclose(wk_hat_expected, wk_hat_actual, atol=1e-5)


def test_dmmse_predictor_against_computation_without_minibatching():
    B, K, D, M, V = 2048, 16, 8, 128, .25
    T = DiscreteTaskDistribution(
        task_size=D,
        num_tasks=M,
    )
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    ws = T.tasks
    
    # do the unbatched method
    # compute w_hat for each k
    loss = torch.empty(B, K, M)
    for k in range(K): # unclear how to (or whether to) vectorise
        Xk = xs[:, :k, :]                   # B K D, slice  -> B k D
        yk = ys[:, :k, :]                   # B K 1, slice  -> B k 1
        # compute loss for each task in the prior
        yh = Xk @ ws.T                      # B k D @ . D M -> B k M
        L  = (yk - yh).square()             # B k 1 - B k M -> B k M
        loss[:, k, :] = L.sum(axis=-2)      # B k M, reduce -> B M
    # average task with Boltzmann posterior at temperature 2sigma^2
    score  = -loss / (2*V)              # B K M / . . .     -> B K M
    probs  = score.softmax(axis=-1)     # B K M, normalise  -> B K M
    ws_hat = probs @ ws                 # B K M @ . M D     -> B K D

    # do the implemented method, forcing a lot of batching
    _, ws_hat_actual = dmmse_predictor(
        xs,
        ys,
        prior=T,
        noise_variance=V,
        return_ws_hat=True,
        _max_bytes=2**20, # 1 MiB
        # fits 4 bytes * 16 ctx * 128 tasks * 128 batch size
        # so this should force minibatching our B=2048 into 16 batches
    )
    
    # compare
    tt.assert_allclose(ws_hat, ws_hat_actual)


def test_ridge_predictor_first_is_zero():
    B, K, D, V = 256, 1, 4, .25
    T = GaussianTaskDistribution(task_size=D)
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    _, ws_hat = ridge_predictor(xs, ys, noise_variance=V, return_ws_hat=True)
    # now for K=1 there is no context for the first predictor so it's just
    # going to be the prior average weight, i.e., 0
    tt.assert_allclose(ws_hat, torch.zeros(B, K, D))


def test_ridge_predictor_second_is_easy():
    B, K, D, V = 256, 2, 4, .25
    T = GaussianTaskDistribution(task_size=D)
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    _, ws_hat = ridge_predictor(xs, ys, noise_variance=V, return_ws_hat=True)
    # now for K=2 there is a single vector of context for the second
    # predictor so we can do a single low-dimensional update and check the
    # dimensions all check out (also here I use inv instead of solve)
    for b in range(B):
        k = 1
        x = xs[b, k-1, :]               # B K D, slice      -> D
        y = ys[b, k-1, 0]               # B K 1, slice      -> .
        xxT = x.outer(x)                # D outer D         -> D D
        LHS = xxT + V * torch.eye(D)    # D D + (. . * D D) -> D D
        inv = torch.linalg.inv(LHS)     # inv(D D)          -> D D
        RHS = x * y                     # D * .             -> D
        wk_hat_expected = inv @ RHS     # D D @ D           -> D
        # check
        wk_hat_actual = ws_hat[b, k, :] # B K D, slice      -> D
        tt.assert_allclose(wk_hat_expected, wk_hat_actual, atol=1e-5)


def test_ridge_predictor_python_loops():
    B, K, D, V = 256, 1, 4, .25
    T = GaussianTaskDistribution(task_size=D)
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    _, ws_hat = ridge_predictor(xs, ys, noise_variance=V, return_ws_hat=True)
    # now do a straight-forward non-vectorised implementation of eqn 8 and
    # compare
    for b in range(B):
        for k in range(K):
            # note: there is no need for k-1 in the indexing because we are
            # already counting k in range(K) from 0, thus this k is actually
            # the 'number of contextual examples to consider': 0, 1, ..., k-1
            Xk = xs[b, :k, :]
            yk = ys[b, :k, :]
            LHS = Xk.T @ Xk + V * torch.eye(D)
            RHS = Xk.T @ yk
            # wk = inv(LHS) @ RHS <--> wk = solve(LHS @ wk == RHS)
            wk_hat_expected = torch.linalg.solve(LHS, RHS).view(D)
            # check
            wk_hat_actual = ws_hat[b, k]
            tt.assert_allclose(wk_hat_expected, wk_hat_actual)


