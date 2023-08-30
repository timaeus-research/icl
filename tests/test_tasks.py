import torch
import torch_testing as tt

from icl.tasks import RegressionSequenceDistribution, SingletonTaskDistribution
from icl.tasks import DiscreteTaskDistribution, GaussianTaskDistribution


def test_singleton_task_distribution():
    task = torch.ones(4)
    distr = SingletonTaskDistribution(task)
    task_sample = torch.ones(10, 4)
    tt.assert_equal(distr.task, task)
    tt.assert_equal(distr.sample_tasks(1), torch.ones(1, 4))
    tt.assert_equal(distr.sample_tasks(10), torch.ones(10, 4))


def test_regression_data_generation_zero_function():
    data = RegressionSequenceDistribution(
        task_distribution=SingletonTaskDistribution(torch.zeros(4)),
        noise_variance=0,
    )
    _xs, ys = data.get_batch(num_examples=16, batch_size=64)
    tt.assert_equal(ys, torch.zeros(64, 16, 1))


def test_regression_data_generation_shape():
    data = RegressionSequenceDistribution(
        task_distribution=SingletonTaskDistribution(torch.ones(4)),
        noise_variance=0,
    )
    xs, ys = data.get_batch(num_examples=16, batch_size=64)
    assert tuple(xs.shape) == (64, 16, 4)
    assert tuple(ys.shape) == (64, 16, 1)


def test_regression_data_generation_sum_function():
    data = RegressionSequenceDistribution(
        task_distribution=SingletonTaskDistribution(torch.ones(4)),
        noise_variance=0,
    )
    xs, ys = data.get_batch(num_examples=16, batch_size=64)
    # check contents of batch: each y should be x0+x1+x2+x3
    tt.assert_equal(ys, xs.sum(axis=-1, keepdim=True))


def test_regression_data_generation_arange():
    B, K, D = 8, 8, 16
    task = torch.arange(D, dtype=torch.float32)
    data = RegressionSequenceDistribution(
        task_distribution=SingletonTaskDistribution(task),
        noise_variance=0,
    )
    xs, ys = data.get_batch(num_examples=K, batch_size=B)
    # one-by-one each y should be sum_i i*x_i
    for b in range(B):
        for k in range(K):
            y = ys[b, k, 0]
            x = xs[b, k, :]
            assert y - (task @ x) < 1e-4 # close enough?


def test_regression_data_generation_zero_plus_variance():
    B, K, D = 1024, 4096, 1
    task = torch.zeros(D)
    data = RegressionSequenceDistribution(
        task_distribution=SingletonTaskDistribution(task),
        noise_variance=4., # standard deviation 2
    )
    _xs, ys = data.get_batch(num_examples=K, batch_size=B)
    # the ys should be gaussian with variance 4 / stddev 2 
    sample_var, sample_mean = torch.var_mean(ys)
    assert abs(sample_mean - 0) < 1e-2
    assert abs(sample_var - 4.) < 1e-2
    # Note: the test uses Bessel's correction, but we actually know the mean,
    # so could just compute the sample variance without correction with known
    # mean zero and maybe this would fail with slightly lower probability.
    # Anyway, I expect this will be fine but we will see.


def test_discrete_task_distribution():
    D, M, N = 4, 16, 128
    distr = DiscreteTaskDistribution(task_size=D, num_tasks=M)
    tasks = distr.sample_tasks(n=N)
    assert tuple(tasks.shape) == (N, D)
    for n in range(N):
        assert tasks[n] in distr.tasks


def test_gaussian_task_distribution():
    D, N = 64, 128
    distr = GaussianTaskDistribution(task_size=D)
    tasks = distr.sample_tasks(n=N)
    assert tuple(tasks.shape) == (N, D)


