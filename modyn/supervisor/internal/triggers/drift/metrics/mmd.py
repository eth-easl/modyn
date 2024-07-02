from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
from modyn.config.schema.pipeline.trigger.drift.result import MetricResult
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import pairwise_kernels
from sklearn.metrics import pairwise_distances

# ------------------------------------------------- unbiased estimate ------------------------------------------------ #
# has quadratic time [O((m+n)^2)] complexity see https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf


def _mmd2_u(kernel_matrix: torch.Tensor, m: int, n: int) -> float:
    """The MMD^2_u unbiased statistic. c_i account for the imbalance in the number of samples in X and Y."""
    kernel_matrix_xx = kernel_matrix[:m, :m]
    kernel_matrix_yy = kernel_matrix[m:, m:]
    kernel_matrix_xy = kernel_matrix[:m, m:]

    c1 = 1 / (m * (m - 1))
    sum_x = torch.sum(kernel_matrix_xx - torch.diag(torch.diagonal(kernel_matrix_xx)))

    c2 = 1 / (n * (n - 1))
    sum_y = torch.sum(kernel_matrix_yy - torch.diag(torch.diagonal(kernel_matrix_yy)))

    sum_xy = torch.mean(kernel_matrix_xy)

    # unbiased estimate for MMD
    return float(c1 * sum_x + c2 * sum_y - 2 * sum_xy)


def _mmd2_u_bstrp(kernel_matrix: torch.Tensor, m: int, n: int, x_y_idxs: tuple[np.ndarray, np.ndarray]) -> float:
    """The MMD^2_u unbiased statistic for bootstrap subsample."""
    x_idx, y_idx = x_y_idxs

    kernel_matrix_xx = kernel_matrix[[[idx] for idx in x_idx], x_idx]
    kernel_matrix_yy = kernel_matrix[[[idx] for idx in y_idx], y_idx]
    kernel_matrix_xy = kernel_matrix[[[idx] for idx in x_idx], y_idx]

    c1 = 1 / (m * (m - 1))
    sum_x = torch.sum(kernel_matrix_xx - torch.diag(torch.diagonal(kernel_matrix_xx)))

    c2 = 1 / (n * (n - 1))
    sum_y = torch.sum(kernel_matrix_yy - torch.diag(torch.diagonal(kernel_matrix_yy)))

    sum_xy = torch.mean(kernel_matrix_xy)

    # unbiased estimate for MMD
    return c1 * sum_x + c2 * sum_y - 2 * sum_xy


def _mm2_u_null_distrib_bootstrap(
    kernel_matrix: torch.Tensor, m: int, n: int, num_bootstraps: int, num_workers: int
) -> np.ndarray:
    """Compute the bootstrapped null-distribution of MMD2u and run a two-sample test via bootstrapping.

    Assume that the null hypothesis is true, i.e., the samples are drawn from the same distribution.
    We compute the null-distribution by bootstrapping samples from the sample pool of X and Y, and compute MMD2u
    on these samples. We later compute a p-value by comparing the observed MMD2u to the null-distribution.

    p-values are defined as the proportion of bootstrapped MMD2u that are at least as extreme as the observed MMD2u.
    If our H0 does not hold and X and Y are drawn from different distributions, the the bootstrapped MMD2u values
    will be smaller as the sets contain samples from both distributions averaging the distances out.
    Hence the proportion of bootstrapped MMD2u values that are at least as extreme as the observed MMD2u will be
    smaller making the p-value smaller.

    We reject the null hypothesis if the p-value is smaller than the significance level.
    """

    # our quadratic unbiased estimator can deal with n != m, therefore we sample m values from X and n values from Y.
    # Bootstrapping uses sampling with replacement.

    x_size = max(int(m * m / (m + n)), 1)
    y_size = max(int(m * n / (m + n)), 1)

    sample_ids = [
        (np.random.choice(m, x_size, replace=True), np.random.choice(n, y_size, replace=True))
        for _ in range(num_bootstraps)
    ]

    _bootstrap_job = partial(_mmd2_u_bstrp, kernel_matrix, x_size, y_size)
    if num_workers > 1:
        with Pool(num_workers) as pool:
            null_distrib = np.array(pool.map(_bootstrap_job, sample_ids))
    else:
        null_distrib = np.zeros(num_bootstraps)
        for i, idxs in enumerate(sample_ids):
            null_distrib[i] = _bootstrap_job(idxs)

    return null_distrib


# -------------------------------------------------- hypothesis test ------------------------------------------------- #


def mmd2_two_sample_test(
    reference_emb: torch.Tensor | np.ndarray,
    current_emb: torch.Tensor | np.ndarray,
    num_bootstraps: int = 1000,
    quantile_probability: float = 0.05,
    pca_components: int | None = None,
    device: str = "cpu",
    num_workers: int = 1,
) -> MetricResult:
    """Compute the MMD^2 two-sample test statistic and the p-value.

    The MMD^2 two-sample test statistic is computed using the unbiased estimator for small sample sizes
    (m, n < 1000) and the linear approximation estimator for large sample sizes (m, n >= 1000).

    The p-value is computed by bootstrapping the null-distribution of MMD^2 and comparing the observed MMD^2
    to the null-distribution.

    Args:
        x: The first sample.
        y: The second sample.
        num_bootstraps: The number of bootstraps to compute the null-distribution.

    Returns:
        The MMD^2 two-sample test statistic and a boolean indicating whether the null hypothesis is rejected.
    """
    # pylint: disable=too-many-locals
    if pca_components:
        pca = PCA(n_components=pca_components, random_state=0)
        reference_emb = pca.fit_transform(reference_emb)
        current_emb = pca.fit_transform(current_emb)

    x = torch.from_numpy(reference_emb).to(device)
    y = torch.from_numpy(current_emb).to(device)
    m = len(x)
    n = len(y)

    xy = torch.vstack([x, y])
    kernel_matrix = torch.from_numpy(
        pairwise_kernels(xy, metric="rbf", gamma=1.0 / np.median(pairwise_distances(xy, metric="euclidean")) ** 2)
    )
    statistic = _mmd2_u(kernel_matrix, m, n)
    null_distrib = _mm2_u_null_distrib_bootstrap(kernel_matrix, m, n, num_bootstraps, num_workers)

    p_value = max(float(np.mean(null_distrib > statistic)), 1 / num_bootstraps)
    return MetricResult(
        metric_id="mmd",
        distance=statistic,
        p_val=p_value,
        is_drift=p_value <= quantile_probability,
    )
