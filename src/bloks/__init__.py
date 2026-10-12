"""BLOKS sketched block preconditioning for Nyström kernel ridge regression.

Toy JAX solver for Vitale, Meanti, Rando, Ayuso de Dios, Rosasco,
arXiv:2610.13382. Preconditioner factors are recomputed on each application
and discarded. ``KE_JITTER`` (1e-6) is added to each center-Gram block before
Cholesky; see ``bloks.precond``.
"""

import jax

jax.config.update("jax_enable_x64", True)

from bloks.kernels import k_gauss, k_lap
from bloks.pcg import PCGResult, left_pcg
from bloks.precond import KE_JITTER, apply_preconditioner
from bloks.solve import BloksResult, bloks_solve, h_matvec, predict, sample_centers

__all__ = [
    "KE_JITTER",
    "BloksResult",
    "PCGResult",
    "apply_preconditioner",
    "bloks_solve",
    "h_matvec",
    "k_gauss",
    "k_lap",
    "left_pcg",
    "predict",
    "sample_centers",
]
