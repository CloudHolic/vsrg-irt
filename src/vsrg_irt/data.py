"""Prepared IRT data + a disk cache keyed by DataSpec."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import polars as pl

from . import config
from .db import load_response
from .specs import DataSpec


@dataclass
class Dataset:
    person_idx: np.ndarray      # (n_obs,)  int
    item_idx: np.ndarray        # (n_obs,)  int
    response: np.ndarray        # (n_obs,)  float64
    n_persons: int
    n_items: int
    n_obs: int
    items: pl.DataFrame         # columns:  beatmap_id, rate_group  (row order = item code)
    users: pl.DataFrame         # column:   user_id                 (row order = person code)

    def as_model_dict(self) -> dict:
        return {
            "person_idx": jnp.asarray(self.person_idx),
            "item_idx": jnp.asarray(self.item_idx),
            "response": jnp.asarray(self.response),
            "n_persons": self.n_persons,
            "n_items": self.n_items,
            "n_obs": self.n_obs
        }


def prepare(df: pl.DataFrame, spec: DataSpec) -> Dataset:
    while True:
        n0 = df.height
        df = df.filter(
            (pl.len().over(["beatmap_id", "rate_group"]) >= spec.min_item)
            & (pl.len().over("user_id") >= spec.min_user)
        )
        if df.height == n0:
            break

    if df.height == 0:
        raise ValueError(f"No rows survive the >= filters for {spec.cache_name}")

    users = df.select("user_id").unique().sort("user_id").with_row_index("person_idx")
    items = (df.select(["beatmap_id", "rate_group"]).unique()
             .sort(["beatmap_id", "rate_group"]).with_row_index("item_idx"))
    df = df.join(users, on="user_id").join(items, on=["beatmap_id", "rate_group"])

    return Dataset(
        person_idx=df["person_idx"].to_numpy().astype(np.int64),
        item_idx=df["item_idx"].to_numpy().astype(np.int64),
        response=df["response"].to_numpy().astype(np.float64),
        n_persons=int(users.height),
        n_items=int(items.height),
        n_obs=int(df.height),
        items=items.select(["beatmap_id", "rate_group"]),
        users=users.select("user_id")
    )


# Disk cache (npz) -------------------------------------------------------------

def _cache_path(spec: DataSpec):
    return config.DATA_CACHE_DIR / f"{config.DATA_TAG}_{spec.cache_name}.npz"


def _save(spec: DataSpec, d: Dataset) -> None:
    config.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        _cache_path(spec),
        person_idx=d.person_idx, item_idx=d.item_idx, response=d.response,
        n_persons=d.n_persons, n_items=d.n_items, n_obs=d.n_obs,
        item_beatmap_id=d.items["beatmap_id"].to_numpy(),
        item_rate_group=d.items["rate_group"].to_numpy().astype(str),
        user_id=d.users["user_id"].to_numpy()
    )


def _load(spec: DataSpec) -> Dataset:
    z = np.load(_cache_path(spec), allow_pickle=False)
    return Dataset(
        person_idx=z["person_idx"], item_idx=z["item_idx"], response=z["response"],
        n_persons=int(z["n_persons"]), n_items=int(z["n_items"]), n_obs=int(z["n_obs"]),
        items=pl.DataFrame({"beatmap_id": z["item_beatmap_id"], "rate_group": z["item_rate_group"]}),
        users=pl.DataFrame({"user_id": z["user_id"]})
    )


def load_dataset(spec: DataSpec, *, refresh: bool=False, allow_db: bool=True) -> Dataset:
    """Prepared Dataset for a spec, from cache when available."""
    if not refresh and _cache_path(spec).exists():
        return _load(spec)

    if not allow_db:
        raise FileNotFoundError(
            f"no cached data for {spec.cache_name} and no connection given; "
            f"warm the cache first (load_dataset(spec, allow_db=True))"
        )

    d = prepare(load_response(spec), spec)
    _save(spec, d)
    return d
