"""Nyström kernel ridge regression solved by BLOKS left PCG (eqs. 5-7).

The linear system is

    K_nm[i, j] = k(x_i, x_tilde_j)
    K_e[j, j'] = k(x_tilde_j, x_tilde_j')
    H = K_nm.T @ K_nm + lambda * n * K_e
    rhs = K_nm.T @ y

``H`` is never formed. Products use ``h_matvec``. Preconditioner blocks are
recomputed from the center coordinates on every PCG iteration.

``K_e`` is not jittered inside ``h_matvec``. The preconditioner adds
``KE_JITTER * I`` to each block before Cholesky only.
"""

from typing import Callable, NamedTuple, Optional

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from bloks.kernels import k_gauss
from bloks.pcg import left_pcg
from bloks.precond import KE_JITTER, preconditioner_from_centers


class BloksResult(NamedTuple):
    a: jax.Array
    centers: jax.Array
    indices: jax.Array
    n_iter: int
    rel_residual: float


def h_matvec(vector, K_nm, K_e, lam):
    """Matrix-free product ``H p = K_nm.T (K_nm p) + lambda n (K_e p)``.

    ``n`` is the number of rows of ``K_nm``. This does not form ``H``.
    """
    vector = jnp.asarray(vector, dtype=jnp.float64)
    K_nm = jnp.asarray(K_nm, dtype=jnp.float64)
    K_e = jnp.asarray(K_e, dtype=jnp.float64)
    n = K_nm.shape[0]
    return K_nm.T @ (K_nm @ vector) + float(lam) * n * (K_e @ vector)


def sample_centers(X, m, key):
    """Sample ``m`` training rows without replacement.

    Returns ``(indices, centers)`` with ``centers = X[indices]``.
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (n, d)")
    n = int(X.shape[0])
    m = int(m)
    if m < 1 or m > n:
        raise ValueError(f"m must satisfy 1 <= m <= n (got m={m}, n={n})")
    indices = jax.random.choice(key, n, shape=(m,), replace=False)
    return indices, X[indices]


def predict(X_test, centers, coefficients, sigma, kernel=k_gauss):
    """Nyström predictor ``f(x) = sum_i a_i k(x, x_tilde_i)``."""
    gram = kernel(X_test, centers, sigma)
    return gram @ jnp.asarray(coefficients, dtype=jnp.float64)


def bloks_solve(
    X,
    y,
    lam,
    m,
    m0,
    sigma,
    tmax,
    tol,
    key,
    kernel: Optional[Callable] = None,
    jitter: float = KE_JITTER,
):
    """Solve Nyström KRR with left BLOKS PCG.

    Centers are ``m`` training rows drawn without replacement. ``m0`` must
    divide ``m``. Returns coefficients ``a`` together with the centers used
    by the predictor.
    """
    if kernel is None:
        kernel = k_gauss
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (n, d)")
    if y.shape != (X.shape[0],):
        raise ValueError("y must have shape (n,)")
    lam = float(lam)
    if not lam > 0.0:
        raise ValueError("lambda must be positive")
    m = int(m)
    m0 = int(m0)
    if m0 < 1:
        raise ValueError("m0 must be positive")
    if m % m0 != 0:
        raise ValueError(f"m0 must divide m (got m={m}, m0={m0})")
    sigma = float(sigma)
    if not sigma > 0.0:
        raise ValueError("sigma must be positive")

    n = int(X.shape[0])
    indices, centers = sample_centers(X, m, key)
    # Stored only as the matvec's kernel factors, not as preconditioner blocks.
    K_nm = kernel(X, centers, sigma)
    K_e = kernel(centers, centers, sigma)
    rhs = K_nm.T @ y

    def matvec(vector):
        return h_matvec(vector, K_nm, K_e, lam)

    def preconditioner(residual):
        # Rebuild each m0 block from center coordinates; do not keep factors.
        return preconditioner_from_centers(
            residual, centers, kernel, sigma, m0, lam, n, jitter=jitter
        )

    solved = left_pcg(matvec, rhs, preconditioner, tmax, tol)
    return BloksResult(
        a=solved.a,
        centers=centers,
        indices=indices,
        n_iter=solved.n_iter,
        rel_residual=solved.rel_residual,
    )
