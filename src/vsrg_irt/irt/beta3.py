"""
Beta3-IRT (Chen et al., AISTATS 2019) - continuous bounded-response IRT, the baseline.

Each response p in (0, 1) (normalized score) is Beta-distributed with shape parameters
driven by a power-ratio item characteristic curve:

    p_ij ~ Beta(alpha_ij, beta_ij)
    alpha_ij = (theta_i / delta_j) ** a_j
    beta_ij = ((1 - theta_i) / (1 - delta_j)) ** a_j
    theta_i, delta_j ~ Beta(1, 1)
    a_j ~ Normal(1, sigma0^2)

Ability theta and difficulty delta are both bounded in (0, 1) - the (0, 1) scale itself identifies the model,
with no external anchor. The discrimination a_j is a free-sign power factor: a_j > 1 gives sigmoidal ICCs,
a_j = 1 parabolic, 0 < a_j < 1 anti-sigmoidal, and a_j < 0 decreasing curves.

KNOWN LIMITATION: the free-sign discrimination makes the model non-identifiable by sign symmetry - an item
seeded with the wrong sign cannot recover.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist

from .base import IRTModel
from .registry import register


def _squeeze(y, n, eps):
    return jnp.clip((y * (n - 1) + 0.5) / n, eps, 1.0 - eps)


@register
class Beta3(IRTModel):
    name = "beta3"
    valid_response = frozenset({"score"})
    difficulty = "unit"

    disc_sd: float = 1.0    # sigma0 in a_j ~ Normal(1, sigma0^2)

    def model(self, data: dict, cfg):
        P, I = data["n_persons"], data["n_items"]
        p_idx, i_idx, y = data["person_idx"], data["item_idx"], data["response"]
        eps = getattr(cfg, "squeeze_eps", 1e-4)

        with numpyro.plate("persons", P):
            theta = numpyro.sample("theta", dist.Beta(1.0, 1.0))    # (0, 1)

        with numpyro.plate("items", I):
            delta = numpyro.sample("delta", dist.Beta(1.0, 1.0))    # (0, 1)
            a = numpyro.sample("a", dist.Normal(1.0, self.disc_sd)) # free sign, mean 1

        th = jnp.clip(theta[p_idx], 1e-6, 1.0 - 1e-6)
        dl = jnp.clip(delta[i_idx], 1e-6, 1.0 - 1e-6)
        a_obs = a[i_idx]

        log_alpha = a_obs * (jnp.log(th) - jnp.log(dl))
        log_beta = a_obs * (jnp.log(1.0 - th) - jnp.log(1.0 - dl))

        alpha = jnp.clip(jnp.exp(log_alpha), 1e-3, 1e6)
        beta = jnp.clip(jnp.exp(log_beta), 1e-3, 1e6)

        y_sq = _squeeze(y, data["n_obs"], eps)
        with numpyro.plate("obs", data["n_obs"]):
            numpyro.sample("y", dist.Beta(alpha, beta), obs=y_sq)

    def extract_difficulty(self, samples) -> np.ndarray:
        return np.asarray(samples["delta"])
