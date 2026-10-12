"""Numerical checks for left PCG, the predictor, and center sampling."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import pytest

from bloks import (
    apply_preconditioner,
    bloks_solve,
    h_matvec,
    k_gauss,
    left_pcg,
    predict,
    sample_centers,
)


def _rel(got, expected):
    return float(jnp.linalg.norm(got - expected) / jnp.linalg.norm(expected))


def _toy_regression():
    key = jax.random.PRNGKey(4)
    k_x, k_noise = jax.random.split(key)
    n, d = 48, 2
    X = jax.random.uniform(k_x, (n, d))
    noise = 0.02 * jax.random.normal(k_noise, (n,))
    y = jnp.sin(2.0 * jnp.pi * X[:, 0]) + 0.3 * X[:, 1] ** 2 + noise
    return X, y


def test_left_pcg_matches_direct_and_predictor():
    X, y = _toy_regression()
    n = X.shape[0]
    m = 12
    sigma = 0.45
    lam = 1e-2
    key = jax.random.PRNGKey(5)
    indices, centers = sample_centers(X, m, key)
    K_nm = k_gauss(X, centers, sigma)
    K_e = k_gauss(centers, centers, sigma)
    explicit = K_nm.T @ K_nm + lam * n * K_e
    rhs = K_nm.T @ y
    direct = jnp.linalg.solve(explicit, rhs)
    X_test = jnp.array([[0.2, 0.3], [0.7, 0.1], [0.05, 0.9]])

    for m0 in (12, 4):
        solved = bloks_solve(
            X,
            y,
            lam=lam,
            m=m,
            m0=m0,
            sigma=sigma,
            tmax=40,
            tol=1e-10,
            key=key,
            kernel=k_gauss,
        )
        assert jnp.array_equal(solved.indices, indices)
        assert _rel(solved.a, direct) < 1e-5
        pred = predict(X_test, solved.centers, solved.a, sigma, k_gauss)
        pred_direct = k_gauss(X_test, centers, sigma) @ direct
        assert _rel(pred, pred_direct) < 1e-5
        # Noisy smooth observations: BLOKS stays on the direct Nyström predictor.
        assert float(jnp.max(jnp.abs(pred - pred_direct))) < 1e-5


def _clustered_points(n, d, n_clusters, spread, scale, key):
    key_means, key_assign, key_noise = jax.random.split(key, 3)
    means = jax.random.normal(key_means, (n_clusters, d)) * scale
    assign = jax.random.randint(key_assign, (n,), 0, n_clusters)
    noise = jax.random.normal(key_noise, (n, d)) * spread
    return means[assign] + noise


def test_bloks_reaches_tolerance_faster_than_cg():
    # Points sit in tight clusters. A lengthscale small compared with the
    # cluster separation, together with a small ridge, makes H ill-conditioned.
    # q = 2 block-Jacobi still reaches 1e-6 in fewer iterations than CG.
    n, d, m, m0 = 100, 5, 36, 18
    X = _clustered_points(n, d, n_clusters=4, spread=0.08, scale=2.5, key=jax.random.PRNGKey(2))
    y = jax.random.normal(jax.random.PRNGKey(6), (n,))
    sigma = 0.5
    lam = 1e-6
    indices = jax.random.choice(jax.random.PRNGKey(1), n, shape=(m,), replace=False)
    centers = X[indices]
    K_nm = k_gauss(X, centers, sigma)
    K_e = k_gauss(centers, centers, sigma)
    rhs = K_nm.T @ y

    def matvec(vector):
        return h_matvec(vector, K_nm, K_e, lam)

    def bloks(residual):
        return apply_preconditioner(residual, K_e, m0, lam, n)

    tol = 1e-6
    tmax = 250
    preconditioned = left_pcg(matvec, rhs, bloks, tmax, tol)
    unpreconditioned = left_pcg(matvec, rhs, lambda residual: residual, tmax, tol)
    assert preconditioned.rel_residual < tol
    assert unpreconditioned.rel_residual < tol
    assert preconditioned.n_iter < unpreconditioned.n_iter, (
        f"BLOKS {preconditioned.n_iter} (rel {preconditioned.rel_residual}) "
        f"vs CG {unpreconditioned.n_iter} (rel {unpreconditioned.rel_residual})"
    )


def test_centers_are_training_subset():
    key = jax.random.PRNGKey(8)
    X = jax.random.normal(key, (30, 4))
    indices, centers = sample_centers(X, 7, jax.random.PRNGKey(9))
    assert indices.shape == (7,)
    assert int(jnp.unique(indices).size) == 7
    assert int(jnp.min(indices)) >= 0
    assert int(jnp.max(indices)) < X.shape[0]
    assert jnp.allclose(centers, X[indices])
    solved = bloks_solve(
        X,
        jnp.ones(30),
        lam=1e-2,
        m=6,
        m0=3,
        sigma=1.0,
        tmax=5,
        tol=1e-8,
        key=jax.random.PRNGKey(3),
    )
    assert jnp.allclose(solved.centers, X[solved.indices])
    assert int(jnp.unique(solved.indices).size) == 6


def test_m0_must_divide_m():
    X = jnp.ones((10, 2))
    y = jnp.ones(10)
    with pytest.raises(ValueError):
        bloks_solve(
            X,
            y,
            lam=1e-2,
            m=8,
            m0=3,
            sigma=1.0,
            tmax=2,
            tol=1e-6,
            key=jax.random.PRNGKey(0),
        )
