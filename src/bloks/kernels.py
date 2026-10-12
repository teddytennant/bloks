"""Kernel evaluations used by the Nyström system (paper appendix D.2)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp


def _prepare(x, xp):
    x = jnp.asarray(x, dtype=jnp.float64)
    xp = jnp.asarray(xp, dtype=jnp.float64)
    x_vec = x.ndim == 1
    xp_vec = xp.ndim == 1
    if x_vec:
        x = x[None, :]
    if xp_vec:
        xp = xp[None, :]
    if x.ndim != 2 or xp.ndim != 2:
        raise ValueError("kernel inputs must be vectors or row-stacked matrices")
    if x.shape[1] != xp.shape[1]:
        raise ValueError("feature dimensions differ")
    return x, xp, x_vec and xp_vec


def _check_sigma(sigma):
    sigma = float(sigma)
    if not sigma > 0.0:
        raise ValueError("sigma must be positive")
    return sigma


def k_gauss(x, xp, sigma):
    """Gaussian kernel ``exp(-||x-x'||^2 / (2 sigma^2))``.

    Vector inputs return a scalar. Matrix inputs of shape ``(n, d)`` and
    ``(m, d)`` return an ``(n, m)`` Gram matrix.
    """
    sigma = _check_sigma(sigma)
    x, xp, squeeze = _prepare(x, xp)
    diff = x[:, None, :] - xp[None, :, :]
    sq = jnp.sum(diff * diff, axis=-1)
    # Clip only tiny negative roundoff so the diagonal is exactly zero distance.
    sq = jnp.maximum(sq, 0.0)
    out = jnp.exp(-sq / (2.0 * sigma * sigma))
    if squeeze:
        return out[0, 0]
    return out


def k_lap(x, xp, sigma):
    """Laplacian kernel ``exp(-||x-x'||_1 / sigma)``."""
    sigma = _check_sigma(sigma)
    x, xp, squeeze = _prepare(x, xp)
    dist = jnp.sum(jnp.abs(x[:, None, :] - xp[None, :, :]), axis=-1)
    out = jnp.exp(-dist / sigma)
    if squeeze:
        return out[0, 0]
    return out
