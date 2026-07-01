from __future__ import annotations

import numpy as np
import polars as pl

from jax import random
from numpyro.infer import SVI, Trace_ELBO, MCMC, NUTS
from numpyro.optim import ClippedAdam
from scipy.stats import spearmanr, pearsonr

from .. import config
from ..specs import FitConfig
from ..data import Dataset
from .base import IRTModel


# Fit -------------------------------------------------------------------------

def fit(model: IRTModel, data: dict, cfg: FitConfig, *, progress: bool=True):
    """Run SVI for one (model, data, cfg). Returns (guide, svi_result)."""
    guide = model.make_guide(data, cfg)
    svi = SVI(model.model, guide, ClippedAdam(cfg.lr), Trace_ELBO())
    res = svi.run(random.PRNGKey(cfg.seed), cfg.num_steps, data, cfg, progress_bar=progress)
    return guide, res


def _summarize(draws: np.ndarray, ci_mass: float):
    lo = (1.0 - ci_mass) / 2.0
    return (draws.mean(0), draws.std(0),
            np.quantile(draws, lo, axis=0), np.quantile(draws, 1.0 - lo, axis=0))


# Weak-label tables -----------------------------------------------------------

def item_intervals(model: IRTModel, guide, params, dataset: Dataset,
                   data: dict, cfg: FitConfig, *, n_draws: int=4000) -> pl.DataFrame:
    """Per-item difficulty posterior."""
    s = guide.sample_posterior(random.PRNGKey(cfg.seed + 1), params, sample_shape=(n_draws,))
    mean, sd, lo, hi = _summarize(model.extract_difficulty(s), cfg.ci_mass)

    out = dataset.items.with_columns(
        pl.col("rate_group").replace_strict(config.RATE_GROUP_CLOCK, return_dtype=pl.Float64).alias("rate"),
        pl.Series("n_resp", np.bincount(dataset.item_idx, minlength=dataset.n_items)),
        pl.Series("delta_mean", mean),
        pl.Series("delta_std", sd),
        pl.Series("delta_lo", lo),
        pl.Series("delta_hi", hi)
    )

    extra = {k: pl.Series(k, np.asarray(v)) for k, v in model.item_extra(s, data).items()}
    if extra:
        out = out.with_columns(**extra)

    return out.with_columns(**{k: pl.lit(v) for k, v in cfg.tag().items()})


def person_intervals(model: IRTModel, guide, params, dataset: Dataset,
                     cfg: FitConfig, *, n_draws: int=4000) -> pl.DataFrame:
    """Per-user theta posterior."""
    s = guide.sample_posterior(random.PRNGKey(cfg.seed + 1), params, sample_shape=(n_draws,))
    mean, sd, lo, hi = _summarize(model.extract_ability(s), cfg.ci_mass)

    out = dataset.users.with_columns(
        pl.Series("n_resp", np.bincount(dataset.person_idx, minlength=dataset.n_persons)),
        pl.Series("theta_mean", mean),
        pl.Series("theta_std", sd),
        pl.Series("theta_lo", lo),
        pl.Series("theta_hi", hi)
    )

    return out.with_columns(**{k: pl.lit(v) for k, v in cfg.tag().items()})


def theta_agreement(persons_a: pl.DataFrame, persons_b: pl.DataFrame) -> dict:
    """Cross-set reliability: join two-person tables on user_id, report theta agreement."""
    m = persons_a.join(persons_b, on="user_id", suffix="_b")
    if m.height < 3:
        return {"n_overlap": m.height, "spearman": float("nan"), "pearson": float("nan")}

    a = m["theta_mean"].to_numpy()
    b = m["theta_mean_b"].to_numpy()

    return {"n_overlap": m.height, "spearman": float(spearmanr(a, b).statistic), "pearson": float(pearsonr(a, b).statistic)}


# VI-width calibration --------------------------------------------------------

def nuts_reference(model: IRTModel, dataset: Dataset, data: dict, cfg: FitConfig,
                   *, num_warmup: int=600, num_samples: int=600) -> pl.DataFrame:
    """NUTS difficulty intervals on the same cell, as a width reference for the VI fit."""
    mcmc = MCMC(NUTS(model.model), num_warmup=num_warmup, num_samples=num_samples, progress_bar=True)
    mcmc.run(random.PRNGKey(cfg.seed + 2), data, cfg)

    d = model.extract_difficulty(mcmc.get_samples())
    lo_q = (1.0 - cfg.ci_mass) / 2.0

    return dataset.items.with_columns(
        pl.Series("delta_mean", d.mean(0)),
        pl.Series("delta_lo", np.quantile(d, lo_q, axis=0)),
        pl.Series("delta_hi", np.quantile(d, 1.0 - lo_q, axis=0))
    )


def interval_scale_factor(vi: pl.DataFrame, ref: pl.DataFrame) -> float:
    m = vi.join(ref, on=["beatmap_id", "rate_group"], suffix="_ref")

    w_vi = (m["delta_hi"] - m["delta_lo"]) / 2.0
    w_ref = (m["delta_hi_ref"] - m["delta_lo_ref"]) / 2.0

    return float(np.median((w_ref / w_vi).to_numpy()))


def apply_scale(vi: pl.DataFrame, factor: float) -> pl.DataFrame:
    half = (vi["delta_hi"] - vi["delta_lo"]) / 2.0 * factor
    return vi.with_columns(
        (vi["delta_mean"] - half).alias("delta_lo"),
        (vi["delta_mean"] + half).alias("delta_hi"),
        (vi["delta_sd"] * factor).alias("delta_sd"),
        pl.lit(factor).alias("width_factor")
    )
