"""Left preconditioned conjugate gradient (Algorithms 3 and 6).

Starts at ``a = 0``, so the initial residual is the right-hand side. The
preconditioned residual ``z = M r`` is cached, which is the same recurrence as
Algorithm 3 and the implementation in Algorithm 6. ``H`` is never formed: the
caller supplies a matrix-free product.
"""

from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


class PCGResult(NamedTuple):
    a: jax.Array
    n_iter: int
    rel_residual: float


def left_pcg(matvec, rhs, preconditioner, tmax, tol):
    """Solve ``H a = rhs`` by left PCG.

    Parameters
    ----------
    matvec:
        Callable computing ``H p`` without forming ``H``.
    rhs:
        Right-hand side.
    preconditioner:
        Callable computing ``M r``. Use the identity for unpreconditioned CG.
    tmax:
        Maximum number of iterations.
    tol:
        Stop when ``||r||_2 / ||rhs||_2 < tol``.

    Returns
    -------
    PCGResult
        Coefficient vector ``a``, iteration count, and the relative residual.
    """
    rhs = jnp.asarray(rhs, dtype=jnp.float64)
    if rhs.ndim != 1:
        raise ValueError("rhs must be a vector")
    tmax = int(tmax)
    if tmax < 0:
        raise ValueError("tmax must be non-negative")
    tol = float(tol)
    if tol <= 0.0:
        raise ValueError("tol must be positive")

    a = jnp.zeros_like(rhs)
    residual = rhs
    preconditioned = preconditioner(residual)
    direction = preconditioned
    rz = jnp.dot(residual, preconditioned)
    rhs_norm = jnp.linalg.norm(rhs)
    rel_residual = 1.0 if float(rhs_norm) == 0.0 else float(jnp.linalg.norm(residual) / rhs_norm)
    if rel_residual < tol or tmax == 0:
        return PCGResult(a, 0, rel_residual)

    n_iter = 0
    for step in range(tmax):
        product = matvec(direction)
        denominator = jnp.dot(direction, product)
        alpha = rz / denominator
        a = a + alpha * direction
        residual = residual - alpha * product
        n_iter = step + 1
        rel_residual = float(jnp.linalg.norm(residual) / rhs_norm)
        if rel_residual < tol:
            break
        preconditioned = preconditioner(residual)
        rz_new = jnp.dot(residual, preconditioned)
        beta = rz_new / rz
        direction = preconditioned + beta * direction
        rz = rz_new
    return PCGResult(a, n_iter, rel_residual)
