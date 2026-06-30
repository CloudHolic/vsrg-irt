"""Prepared IRT data + a disk cache keyed by DataSpec."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pandas as pd

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
    items: pd.DataFrame         # columns:  beatmap_id, rate_group  (row order = item code)
    users: pd.DataFrame         # column:   user_id                 (row order = person code)

    def as_model_dict(self) -> dict:
        return {
            "person_idx": jnp.asarray(self.person_idx),
            "item_idx": jnp.asarray(self.item_idx),
            "response": jnp.asarray(self.response),
            "n_persons": self.n_persons,
            "n_items": self.n_items,
            "n_obs": self.n_obs
        }


def prepare(df: pd.DataFrame, spec: DataSpec) -> Dataset:
    df = df.copy()
    df["item"] = list(zip(df["beatmap_id"], df["rate_group"]))

    while True:
        n0 = len(df)
        df = df[df["item"].map(df["item"].value_counts()) >= spec.min_item]
        df = df[df["user_id"].map(df["user_id"].value_counts()) >= spec.min_user]
        if len(df) == n0:
            break

    if df.empty:
        raise ValueError(f"No rows survive the >= filters for {spec.cache_name}")

    user_codes, users = pd.factorize(df["user_id"], sort=True)
    item_codes, items = pd.factorize(df["item"], sort=True)

    return Dataset(
        person_idx=np.asarray(user_codes),
        item_idx=np.asarray(item_codes),
        response=df["response"].to_numpy(np.float64),
        n_persons=int(len(users)),
        n_items=int(len(items)),
        n_obs=int(len(df)),
        items=pd.DataFrame(items.tolist(), columns=["beatmap_id", "rate_group"]),
        users=pd.DataFrame({"user_id": np.asarray(users)})
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
        items=pd.DataFrame({"beatmap_id": z["item_beatmap_id"], "rate_group": z["item_rate_group"]}),
        users=pd.DataFrame({"user_id": z["user_id"]})
    )


def load_dataset(spec: DataSpec, *, conn=None, refresh: bool=False) -> Dataset:
    """Prepared Dataset for a spec, from cache when available."""
    if not refresh and _cache_path(spec).exists():
        return _load(spec)

    if conn is None:
        raise FileNotFoundError(
            f"no cached data for {spec.cache_name} and no connection given; "
            f"warm the cache first (load_dataset(spec, conn=...))"
        )

    d = prepare(load_response(conn, spec), spec)
    _save(spec, d)
    return d
