from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray


def get_data(
    n: int = 10_000,
    beta: NDArray[np.float64] | Sequence[float] = (-1.0, 1.5, -1.5, 1.0, 1.0, 1.0),
    classification: bool = True,
    non_lin: bool = False,
    squared_terms: bool = False,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Generate a simple two-dimensional synthetic dataset.

    Args:
        n: Number of observations.
        beta: Coefficients used to generate the target.
        classification: Whether to convert the target into binary labels.
        non_lin: Whether to add a multiplicative interaction term.
        squared_terms: Whether to add quadratic terms in each covariate.
        seed: Random seed for reproducibility. If ``None`` a fresh
            generator is used and results are non-deterministic.

    Returns:
        A tuple containing:
            - The feature matrix without intercept.
            - The generated target values or class labels.
            - The feature matrix with an intercept column.
    """
    rng = np.random.default_rng(seed)
    beta_array = np.asarray(beta, dtype=float)

    x = rng.standard_normal((n, 4))
    x_with_intercept = np.hstack([np.ones((x.shape[0], 1)), x])

    y = (
        beta_array[0]
        + beta_array[1] * x_with_intercept[:, 1]
        + beta_array[2] * x_with_intercept[:, 2]
    )

    if non_lin:
        y += beta_array[3] * x_with_intercept[:, 1] * x_with_intercept[:, 2]

    if squared_terms:
        y += (
            beta_array[4] * x_with_intercept[:, 1] ** 2
            + beta_array[5] * x_with_intercept[:, 2] ** 2
        )

    if classification:
        probabilities = 1.0 / (1.0 + np.exp(-y))
        y = rng.binomial(1, probabilities).astype(float)
    else:
        y += np.random.normal(scale=0.5 , size=len(y))

    return x.astype(float), y.astype(float), x_with_intercept.astype(float)


def create_data_unif(
    n: int,
    beta: Sequence[float] = (10, 1, 1, 1, 1),
    dep_level: float = 0.5,
    classification: bool = False,
    non_lin: bool = False,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a synthetic dataset with uniformly distributed covariates.

    Args:
        n: Number of observations.
        beta: Coefficients used to generate the target.
        dep_level: Dependency level between `x1` and `x3`.
        classification: Whether to convert the target into binary labels.
        non_lin: Whether to use a nonlinear target function.
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing:
            - The generated target values or class labels.
            - The design matrix including an intercept column.
    """
    rng = np.random.default_rng(seed)
    beta_array = np.asarray(beta, dtype=float)

    x0 = np.ones(n)
    x1 = rng.uniform(-10.0, 10.0, n)
    x2 = rng.uniform(-10.0, 10.0, n)
    x3 = rng.uniform(-10.0, 10.0, n)
    x4 = rng.uniform(-10.0, 10.0, n)

    x3 = dep_level * x1 + (1.0 - dep_level) * x3

    if non_lin:
        y = (
            beta_array[0]
            + beta_array[1] * x1
            + beta_array[2] * x2
            + beta_array[3] * x1**2
            + beta_array[4] * x2**2
            + x1 * x2
        )
    else:
        y = beta_array[0] + beta_array[1] * x1 + beta_array[2] * x2

    y = y + rng.normal(scale=0.01, size=n)

    if classification:
        y = y - y.min()
        y = y / max(y.max(), 1e-12)
        y = (y > np.median(y)).astype(float)

    x = np.column_stack((x0, x1, x2, x3, x4))

    return y.astype(float), x.astype(float)


def create_bsr_data(
    n: int,
    urange: tuple[float, float] = (-3, 3),
    func: int = 1,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate benchmark-style synthetic regression data.

    Args:
        n: Number of observations.
        urange: Range used for uniform sampling of the second feature.
        func: Index selecting the target-generating function.
        seed: Random seed for reproducibility.

    Returns:
        A tuple containing:
            - The generated target values.
            - The feature matrix with two columns.
    """
    rng: Generator = np.random.default_rng(seed)
    lower, upper = urange

    x1 = np.linspace(-1.5, 1.5, n)
    x2 = rng.uniform(lower, upper, n)

    if func == 1:
        y = 2.5 * x1**4 - 1.3 * x1**3 + 0.5 * x2**2 - 1.7 * x2
    elif func == 2:
        y = 8.0 * x1**2 + 8.0 * x2**3 - 15.0
    elif func == 3:
        y = 0.2 * x1**3 - 0.5 * x1 + 0.5 * x2**3 - 1.2 * x2
    elif func == 4:
        y = 1.5 * np.exp(x1) + 5.0 * np.cos(x2)
    elif func == 5:
        y = 6.0 * np.sin(x1) * np.cos(x2)
    elif func == 6:
        y = 1.35 * x1 * x2 + 5.5 * np.sin((x1 - 1.0) * (x2 - 1.0))
    elif func == 7:
        noise = rng.normal(scale=0.02, size=n)
        y = (
            x1
            + 0.3 * np.sin(2.0 * np.pi * (x1 + noise))
            + 0.3 * np.sin(4.0 * np.pi * (x1 + noise))
            + noise
        )
    elif func == 8:
        noise = rng.normal(scale=1.0, size=n)
        y = 10.0 * np.sin(2.0 * np.pi * x1) + noise
    else:
        raise ValueError("func must be in {1, 2, 3, 4, 5, 6, 7, 8}")

    x = np.column_stack((x1, x2))

    return y.astype(float), x.astype(float)