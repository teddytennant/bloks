"""Sketched block-Jacobi preconditioner (Section 3, Algorithms 5 and 6).

The m Nyström centers are split into ``q = m / m0`` contiguous blocks. For each
block the action of ``M`` is the four triangular solves in Algorithm 5, then
the whole vector is scaled by ``1 / (n q)``.

JAX's Cholesky factor is lower-triangular. The implementation converts it to
the paper's upper factor ``T = L.T`` (so ``K = T.T @ T``) and solves against
that upper factor. Factors of one ``m0 x m0`` block are computed and discarded
before the next block is touched; they are not cached across iterations.

Jitter
------
Each block ``K_e_j`` is replaced by ``K_e_j + KE_JITTER * I`` immediately
before Cholesky (default ``KE_JITTER = 1e-6``) so the factor exists when a
block is numerically singular. The Nyström operator ``H`` is not jittered;
pass ``jitter=0`` only if the caller already shifted the blocks.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

# Documented SPD shift applied to each center-Gram block before Cholesky.
KE_JITTER = 1e-6


def _as_f64(array):
    return jnp.asarray(array, dtype=jnp.float64)


def upper_cholesky(matrix):
    """Upper Cholesky factor ``T`` with ``matrix = T.T @ T``."""
    lower = jnp.linalg.cholesky(matrix)
    return lower.T


def _four_solves(block_residual, block_kernel, lam, jitter):
    """Algorithm 5 on one block, before the global ``1 / (n q)`` scale.

    ``r <- T^{-T} r``, ``r <- A^{-T} r``, ``r <- A^{-1} r``, ``r <- T^{-1} r``,
    where ``T`` is the upper Cholesky factor of the jittered block and ``A`` is
    the upper Cholesky factor of ``(1/m0) T @ T.T + lambda I``.
    """
    m0 = block_kernel.shape[0]
    eye = jnp.eye(m0, dtype=block_kernel.dtype)
    shifted = block_kernel + jitter * eye
    gram = upper_cholesky(shifted)
    # A = chol_upper((1/m0) * T @ T.T + lambda I)
    scaled = (1.0 / m0) * (gram @ gram.T) + lam * eye
    factor = upper_cholesky(scaled)
    residual = solve_triangular(gram, block_residual, lower=False, trans=1)
    residual = solve_triangular(factor, residual, lower=False, trans=1)
    residual = solve_triangular(factor, residual, lower=False, trans=0)
    residual = solve_triangular(gram, residual, lower=False, trans=0)
    return residual


def _check_block_size(m, m0):
    if m0 < 1:
        raise ValueError("m0 must be positive")
    if m < 1:
        raise ValueError("m must be positive")
    if m % m0 != 0:
        raise ValueError(f"m0 must divide m (got m={m}, m0={m0})")


def apply_preconditioner(residual, center_kernel, m0, lam, n, jitter=KE_JITTER):
    """Apply the left BLOKS preconditioner ``M`` to ``residual``.

    Parameters
    ----------
    residual:
        Vector in ``R^m``.
    center_kernel:
        ``K_e`` of shape ``(m, m)``, kernel on the Nyström centers in block order.
    m0:
        Block size. Must divide ``m``.
    lam:
        Ridge penalty ``lambda > 0``.
    n:
        Number of training rows. The action is scaled by ``1 / (n q)``.
    jitter:
        Multiple of ``I`` added to each block before Cholesky. Default ``1e-6``.

    The block factors are recomputed from ``center_kernel`` on every call and
    dropped after that block's solves. This is ``M r`` with
    ``M = diag(H_tilde_1^{-1}, ..., H_tilde_q^{-1})`` and

        H_tilde_j = (n q / m0) K_j K_j + lambda n q K_j

    on the jittered blocks. The code does not form those inverses.
    """
    residual = _as_f64(residual)
    center_kernel = _as_f64(center_kernel)
    if center_kernel.ndim != 2 or center_kernel.shape[0] != center_kernel.shape[1]:
        raise ValueError("center_kernel must be square")
    m = int(center_kernel.shape[0])
    m0 = int(m0)
    _check_block_size(m, m0)
    if residual.shape != (m,):
        raise ValueError("residual must have shape (m,)")
    lam = float(lam)
    if not lam > 0.0:
        raise ValueError("lambda must be positive")
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    jitter = float(jitter)
    q = m // m0
    # One block factor at a time: the next iteration overwrites nothing retained.
    output = jnp.zeros_like(residual)
    for block in range(q):
        start = block * m0
        stop = start + m0
        solved = _four_solves(
            residual[start:stop],
            center_kernel[start:stop, start:stop],
            lam,
            jitter,
        )
        output = output.at[start:stop].set(solved)
    return output / (n * q)


def preconditioner_from_centers(residual, centers, kernel, sigma, m0, lam, n, jitter=KE_JITTER):
    """Like ``apply_preconditioner``, but rebuilds each block Gram from points.

    Used by the solver so Cholesky factors (and the block Grams they come from)
    are recomputed every iteration rather than stored.
    """
    residual = _as_f64(residual)
    centers = _as_f64(centers)
    m = int(centers.shape[0])
    m0 = int(m0)
    _check_block_size(m, m0)
    if residual.shape != (m,):
        raise ValueError("residual must have shape (m,)")
    lam = float(lam)
    if not lam > 0.0:
        raise ValueError("lambda must be positive")
    n = int(n)
    if n < 1:
        raise ValueError("n must be positive")
    jitter = float(jitter)
    q = m // m0
    output = jnp.zeros_like(residual)
    for block in range(q):
        start = block * m0
        stop = start + m0
        points = centers[start:stop]
        block_kernel = kernel(points, points, sigma)
        solved = _four_solves(residual[start:stop], block_kernel, lam, jitter)
        output = output.at[start:stop].set(solved)
    return output / (n * q)
