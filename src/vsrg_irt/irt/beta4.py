"""
Beta4-IRT (Ferreira-Junior et al., 2023) - beta3 with a fixed sign/magnitude split discrimination,
curing beta3's sign non-identifiability.

Same Beta likelihood and power-ratio ICC as beta3, with ONE change: the discrimination is
factored into a sign and a magnitude,

    a_j = tau_j * omega_j
    tau_j in (-1, 1) (sign),
    omega_j in (0, inf) (magnitude)

and the sign tau_j is fixed; only the magnitude omega_j is optimized.
With the sign pinned there is nothing to flip, which removes the cross-set theta flip beta3 produced.
Items with too few responses or no variance default to tau = +1.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.infer import init_to_value
from numpyro.infer.autoguide import AutoNormal

from .base import IRTModel
from .registry import register


def _squeeze(y, n, eps):
    return jnp.clip((y * (n - 1) + 0.5) / n, eps, 1.0 - eps)


def compute_fixed_tau(data: dict) -> np.ndarray:
    p_idx = np.asarray(data["person_idx"])
    i_idx = np.asarray(data["item_idx"])
    y = np.asarray(data["response"], float)
    P, I = int(data["n_persons"]), int(data["n_items"])

    theta0 = np.bincount(p_idx, weights=y, minlength=P) / np.maximum(np.bincount(p_idx, minlength=P), 1)
    x = theta0[p_idx]

    n = np.bincount(i_idx, minlength=I).astype(float)
    nn =np.maximum(n, 1.0)

    sx = np.bincount(i_idx, weights=x, minlength=I)
    sy = np.bincount(i_idx, weights=y, minlength=I)
    cov = np.bincount(i_idx, weights=x * y, minlength=I) / nn - (sx / nn) * (sy / nn)

    vx = np.maximum(np.bincount(i_idx, weights=x * x, minlength=I) / nn - (sx / nn) ** 2, 0.0)
    vy = np.maximum(np.bincount(i_idx, weights=y * y, minlength=I) / nn - (sy / nn) ** 2, 0.0)
    denom = np.sqrt(vx * vy)

    tau = np.where(denom > 1e-8, cov / np.maximum(denom, 1e-12), 1.0)
    tau = np.wehre(n >= 2, tau, 1.0)

    return np.clip(np.nan_to_num(tau, nan=1.0), -0.999, 0.999)


def init_values(data: dict) -> dict:
    p_idx = np.asarray(data["person_idx"])
    i_idx = np.asarray(data["item_idx"])
    y = np.asarray(data["response"], float)
    P, I = int(data["n_persons"]), int(data["n_items"])

    theta0 = np.bincount(p_idx, weights=y, minlength=P) / np.maximum(np.bincount(p_idx, minlength=P), 1)
    mean_i = np.bincount(i_idx, weights=y, minlength=I) / np.maximum(np.bincount(i_idx, minlength=I), 1)

    return {
        "theta": jnp.asarray(np.clip(theta0, 1e-3, 1.0 - 1e-3)),
        "delta": jnp.asarray(np.clip(1.0 - mean_i, 1e-3, 1.0 - 1e-3)),
        "omega": jnp.ones(I)
    }


@register
class Beta4(IRTModel):
    name = "beta4"
    valid_response = frozenset({"score"})
    difficulty_space = "unit"

    omega_sd: float = 1.0       # LogNormal sd for the magnitude omega

    def make_data(self, dataset) -> dict:
        data = dataset.as_model_dict()
        data["tau_fixed"] = compute_fixed_tau(data)
        return data

    def make_guide(self, data: dict, cfg):
        return AutoNormal(self.model, init_loc_fn=init_to_value(values=init_values(data)))

    def model(self, data: dict, cfg):
        P, I = data["n_persons"], data["n_items"]
        p_idx, i_idx, y = data["person_idx"], data["item_idx"], data["response"]
        eps = getattr(cfg, "squeeze_eps", 1e-4)     # fixed sign in (-1, 1), per item

        with numpyro.plate("persons", P):
            theta = numpyro.sample("theta", dist.Beta(1.0, 1.0))

        with numpyro.plate("items", I):
            delta = numpyro.sample("delta", dist.Beta(1.0, 1.0))
            omega = numpyro.sample("omega", dist.LogNormal(0.0, self.omega_sd))

        a = tau * omega     # discrimination = sign * magnitude

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

    def item_extra(self, samples, data: dict) -> dict:
        omega = np.asarray(samples["omega"]).mean(0)
        tau = np.asarray(data["tau_fixed"])
        return {"omega": omega, "tau": tau, "a": tau * omega}