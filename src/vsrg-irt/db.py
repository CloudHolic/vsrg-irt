from __future__ import annotations

import pandas as pd
import psycopg2

from . import config
from .specs import DataSpec


def connect(dsn: str | None = None):
    return psycopg2.connect(dsn or config.DSN)


def load_response(conn, spec: DataSpec) -> pd.DataFrame:
    """Raw (user_id, beatmap_id, rate_group, response) rows for one DataSpec."""
    where = ["mania_keys = %(key)s"]
    if spec.sample == "random":
        where.append("in_random")

    sql = (
        f"SELECT user_id, beatmap_id, rate_group, response "
        f"FROM {config.VIEWS[spec.response]} WHERE {' AND '.join(where)}"
    )

    cur = conn.cursor()
    try:
        cur.execute(sql, {"key": spec.key})
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        cur.close()
