from __future__ import annotations

import polars as pl

from . import config
from .specs import DataSpec


def load_response(spec: DataSpec) -> pl.DataFrame:
    """Raw (user_id, beatmap_id, rate_group, response) rows for one DataSpec."""
    where = ["mania_keys = " + str(int(spec.key))]

    if spec.sample == "random":
        where.append("in_random")

    sql = (
        f"SELECT user_id, beatmap_id, rate_group, response "
        f"FROM {config.VIEWS[spec.response]} WHERE {' AND '.join(where)}"
    )

    return pl.read_database_uri(sql, config.DSN, engine="connectorx")
