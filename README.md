implementation in JAX of BLOKS sketched block preconditioning for Nyström kernel ridge regression (arXiv:2610.13382).

```bash
pip install -e .
python -m pytest
```

CPU toy regression only, no Taxi/HIGGS/CIFAR runs, blocks recomputed each iteration rather than stored.

Each center-Gram block is shifted by `1e-6 * I` before its Cholesky factor so the block is SPD; that jitter is not added inside the Nyström matvec.
