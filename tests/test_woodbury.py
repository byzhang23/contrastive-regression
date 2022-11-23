import numpy as onp
import jax.numpy as jnp
import sys
sys.path.append("../models")
from linear_cr_bz import woodbury_inversion


# Test Woodbury matrix inversion
for _ in range(50):
    p = 10
    d = 2
    S = onp.random.normal(size=(d, p)).astype(onp.float32)
    sigma_sq = 1e-1
    woodbury_inv = woodbury_inversion(
        A_diag=sigma_sq * jnp.ones(p),
        U=S.T,
        C_diag=jnp.ones(d),
        V=S
    )
    full_inv = jnp.linalg.solve(
        S.T @ S + sigma_sq * jnp.eye(p),
        jnp.eye(p)
    )
    assert onp.allclose(full_inv, woodbury_inv, atol=1e-4)
print("Test passsed.")