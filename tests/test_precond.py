"""Numerical checks for kernels, the H matvec, and the block preconditioner."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest
from jax.scipy.linalg import solve_triangular

from bloks import KE_JITTER, apply_preconditioner, h_matvec, k_gauss, k_lap


def _upper_chol(matrix):
    return jnp.linalg.cholesky(matrix).T


def _independent_triangular(residual, center_kernel, m0, lam, n, jitter):
    """Paper upper-factor solves, written separately from the library path."""
    m = center_kernel.shape[0]
    q = m // m0
    pieces = []
    eye = jnp.eye(m0, dtype=center_kernel.dtype)
    for block in range(q):
        start = block * m0
        stop = start + m0
        block_kernel = center_kernel[start:stop, start:stop] + jitter * eye
        gram = _upper_chol(block_kernel)
        factor = _upper_chol((1.0 / m0) * (gram @ gram.T) + lam * eye)
        local = residual[start:stop]
        local = solve_triangular(gram, local, lower=False, trans=1)
        local = solve_triangular(factor, local, lower=False, trans=1)
        local = solve_triangular(factor, local, lower=False, trans=0)
        local = solve_triangular(gram, local, lower=False, trans=0)
        pieces.append(local)
    return jnp.concatenate(pieces) / (n * q)


def _explicit_block_inverse(residual, center_kernel, m0, lam, n, jitter):
    """Block-diagonal inverse of H_tilde on the same jittered blocks."""
    m = center_kernel.shape[0]
    q = m // m0
    pieces = []
    eye = jnp.eye(m0, dtype=center_kernel.dtype)
    for block in range(q):
        start = block * m0
        stop = start + m0
        block_kernel = center_kernel[start:stop, start:stop] + jitter * eye
        h_tilde = (n * q / m0) * (block_kernel @ block_kernel) + lam * n * q * block_kernel
        pieces.append(jnp.linalg.solve(h_tilde, residual[start:stop]))
    return jnp.concatenate(pieces)


def _rel(got, expected):
    return float(jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected))


def test_kernel_formulas_and_psd():
    left = jnp.array([1.0, -2.0, 0.5])
    right = jnp.array([0.25, 0.5, -1.5])
    sigma = 1.7
    diff = left - right
    gauss = jnp.exp(-jnp.sum(diff * diff) / (2.0 * sigma * sigma))
    lap = jnp.exp(-jnp.sum(jnp.abs(diff)) / sigma)
    assert abs(float(k_gauss(left, right, sigma) - gauss)) < 1e-12
    assert abs(float(k_lap(left, right, sigma) - lap)) < 1e-12
    # Matrix evaluation agrees with the scalar formula on the same pair.
    pair = jnp.stack([left, right])
    gauss_mat = k_gauss(pair, pair, sigma)
    lap_mat = k_lap(pair, pair, sigma)
    assert abs(float(gauss_mat[0, 1] - gauss)) < 1e-12
    assert abs(float(lap_mat[0, 1] - lap)) < 1e-12

    points = jax.random.normal(jax.random.PRNGKey(2), (9, 3))
    for kernel, lengthscale in ((k_gauss, 0.8), (k_lap, 1.1)):
        gram = kernel(points, points, lengthscale)
        assert jnp.allclose(gram, gram.T, atol=1e-12)
        eigenvalues = jnp.linalg.eigvalsh(gram)
        assert float(jnp.min(eigenvalues)) >= -1e-8


def test_h_matvec_matches_explicit():
    key = jax.random.PRNGKey(7)
    k_x, k_c, k_p = jax.random.split(key, 3)
    n, d, m = 40, 3, 12
    X = jax.random.normal(k_x, (n, d))
    centers = jax.random.normal(k_c, (m, d))
    vector = jax.random.normal(k_p, (m,))
    lam = 0.03
    K_nm = k_gauss(X, centers, 0.9)
    K_e = k_gauss(centers, centers, 0.9)
    explicit = K_nm.T @ K_nm + lam * n * K_e
    got = h_matvec(vector, K_nm, K_e, lam)
    assert _rel(got, explicit @ vector) < 1e-12


@pytest.mark.parametrize("m0", [12, 4])
def test_preconditioner_matches_triangular_and_inverse(m0):
    key = jax.random.PRNGKey(11)
    k_c, k_r = jax.random.split(key)
    m, n = 12, 37
    centers = jax.random.normal(k_c, (m, 3))
    residual = jax.random.normal(k_r, (m,))
    center_kernel = k_gauss(centers, centers, 0.7)
    lam = 1e-2
    got = apply_preconditioner(residual, center_kernel, m0, lam, n)
    triangular = _independent_triangular(residual, center_kernel, m0, lam, n, KE_JITTER)
    inverse = _explicit_block_inverse(residual, center_kernel, m0, lam, n, KE_JITTER)
    assert _rel(got, triangular) < 1e-8
    assert _rel(got, inverse) < 1e-8
    # A second vector, so the match is not a one-vector accident.
    other = jax.random.normal(jax.random.PRNGKey(12), (m,))
    got_other = apply_preconditioner(other, center_kernel, m0, lam, n)
    inverse_other = _explicit_block_inverse(other, center_kernel, m0, lam, n, KE_JITTER)
    assert _rel(got_other, inverse_other) < 1e-8


def test_q1_matches_falkon_factor():
    key = jax.random.PRNGKey(19)
    k_c, k_r = jax.random.split(key)
    m = 12
    n = 41
    centers = jax.random.normal(k_c, (m, 2))
    residual = jax.random.normal(k_r, (m,))
    lam = 2e-2
    center_kernel = k_lap(centers, centers, 1.3)
    shifted = center_kernel + KE_JITTER * jnp.eye(m)
    gram = _upper_chol(shifted)
    factor = _upper_chol((1.0 / m) * (gram @ gram.T) + lam * jnp.eye(m))
    eye = jnp.eye(m)
    gram_inv = solve_triangular(gram, eye, lower=False)
    factor_inv = solve_triangular(factor, eye, lower=False)
    # Eq. 34: B = n^{-1/2} T^{-1} A^{-1}, M = B B.T.
    falkon = (1.0 / jnp.sqrt(n)) * (gram_inv @ factor_inv)
    expected = falkon @ (falkon.T @ residual)
    got = apply_preconditioner(residual, center_kernel, m0=m, lam=lam, n=n)
    assert _rel(got, expected) < 1e-8


def test_m0_must_divide_m():
    center_kernel = jnp.eye(12)
    residual = jnp.ones(12)
    with pytest.raises(ValueError):
        apply_preconditioner(residual, center_kernel, m0=5, lam=1e-2, n=20)
