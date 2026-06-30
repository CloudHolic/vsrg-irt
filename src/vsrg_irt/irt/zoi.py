"""
ZOI-IRT - Zero-and-One-Inflated Beta IRT

Use this when responses pile up at the boundaries 0 and/or 1 on top of spreading over (0, 1):
The response is a three-part mixture per (person, item):

    P(x = 0)        = sigmoid(gamma0 - alpha * theta)
    P(x = 1)        = sigmoid(alpha * theta - gamma1)
    P(0 < x < 1)    = the remaining mass, sahped by a Beta body

with a discrimination `alpha` shared by the boundary gates and the body, a body location `beta`,
dispersion `omega`, and ordered thresholds gamma0 < gamma1.

Ability theta ~ N(0, 1) anchors the latent scale on the real line, so this model's difficulty lives on R.
Item difficulty is the "overall" theta* where the full mixture mean (boundary + body) crosses 0.5, found by bisection;
it reduces to the body location -beta/alpha when uninfalted but stays correct when boundary mass dominates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import random
import numpy as np

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from scipy.special import expit

from .base import IRTModel
from .registry import register


class ZeroOneInflatedBeta(dist.Distribution):
    arg_constraints = {
        "mu": constraints.open_interval(0.0, 1.0),
        "phi": constraints.positive,
        "gate0": constraints.unit_interval,
        "gate1": constraints.unit_interval
    }
    support = constraints.unit_interval
    reparameterized_params: list = []

    def __init__(self, mu, phi, gate0, gate1, *, validate_args=None):
        self.mu, self.phi, self.gate0, self.gate1 = mu, phi, gate0, gate1
        batch_shape = jax.lax.broadcast_shapes(
            jnp.shape(mu), jnp.shape(phi), jnp.shape(gate0), jnp.shape(gate1))
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @property
    def _beta(self):
        return dist.Beta(self.mu * self.phi, (1.0 - self.mu) * self.phi)

    def log_prob(self, value):
        eps = jnp.finfo(jnp.result_type(float, 0.0)).tiny
        gate0 = jnp.clip(self.gate0, eps, 1.0)
        gate1 = jnp.clip(self.gate1, eps, 1.0)
        interior = jnp.clip(1.0 - self.gate0 - self.gate1, eps, 1.0)

        is_zero = value <= 0.0
        is_one = value >= 1.0
        is_interior = (~is_zero) & (~is_one)

        safe = jnp.clip(value, eps, 1.0 - eps)
        log_body = jnp.log(interior) + self._beta.log_prob(safe)

        return jnp.where(is_zero, jnp.log(gate0),
                         jnp.where(is_one, jnp.log(gate1),
                                   jnp.where(is_interior, log_body, -jnp.inf)))

    def sample(self, key, sample_shape=()):
        shape = sample_shape + self.batch_shape
        k_u, k_b = random.split(key)
        u = random.uniform(k_u, shape)
        body = self._beta.expand(self.batch_shape).sample(k_b, sample_shape)
        return jnp.where(u < self.gate0, 0.0,
                         jnp.where(u < self.gate0 + self.gate1, 1.0, body))


@register
class Zoi(IRTModel):
    name = "zoi"
    valid_response = frozenset({"score", "acc"})
    difficulty_space = "real"

    def model(self, data: dict, cfg):
        P, I = data["n_persons"], data["n_items"]
        p_idx, i_idx, y = data["person_idx"], data["item_idx"], data["response"]

        with numpyro.plate("persons", P):
            theta = numpyro.sample("theta", dist.Normal(0.0, 1.0))      # anchor

        with numpyro.plate("items", I):
            alpha = numpyro.sample("alpha", dist.LogNormal(0.0, 0.5))   # > 0, shared
            beta = numpyro.sample("beta", dist.Normal(0.0, 2.0))        # body easiness
            omega = numpyro.sample("omega", dist.Normal(2.0, 1.0))      # dispersion
            gamma0 = numpyro.sample("gamma0", dist.Normal(-1.0, 2.0))   # 0-threshold
            gap = numpyro.sample("gap", dist.LogNormal(0.0, 0.7))       # gamma1 = gamma0 + gap

        gamma1 = gamma0 + gap

        at = alpha[i_idx] * theta[p_idx]
        lin = at + beta[i_idx]
        sh1 = jnp.exp((lin + omega[i_idx]) / 2.0)
        sh2 = jnp.exp((omega[i_idx] - lin) / 2.0)
        mu = jnp.clip(sh1 / (sh1 + sh2), 1e-6, 1.0 - 1e-6)
        phi = sh1 + sh2

        gate0 = jax.nn.sigmoid(gamma0[i_idx] - at)
        gate1 = jax.nn.sigmoid(at - gamma1[i_idx])

        with numpyro.plate("obs", data["n_obs"]):
            numpyro.sample("y", ZeroOneInflatedBeta(mu, phi, gate0, gate1), obs=y)

    def extract_difficulty(self, samples) -> np.ndarray:
        a = np.asarray(samples["alpha"])
        b = np.asarray(samples["beta"])
        g0 = np.asarray(samples["gamma0"])
        g1 = g0 + np.asarray(samples["gap"])

        lo = np.full_like(a, -12.0)
        hi = np.full_like(b, 12.0)

        for _ in range(40):
            m = 0.5 * (lo + hi)
            at = a * m
            E = expit(at - g1) + (expit(g1 - at) - expit(g0 - at)) * expit(at + b)
            hi = np.where(E > 0.5, m, hi)
            lo = np.where(E > 0.5, lo, m)

        return 0.5 * (lo + hi)