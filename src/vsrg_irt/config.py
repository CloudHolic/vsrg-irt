from __future__ import annotations

import os
from pathlib import Path

# DB ---------------------------------------------------------------------------

DSN: str = os.environ.get(
    "DSN",
    "postgresql://thesis:thesis@homeserver.tail26c9db.ts.net:5432/thesis"
)


# osu! mirror -----------------------------------------------------------------

OSU_MIRRORS: list[str] = [
    "https://osu.direct/api/osu/{beatmap_id}",
    "https://catboy.best/osu/{beatmap_id}",
]
USER_AGENT: str = os.environ.get(
    "THESIS_UA", "osu-mania-thesis/1.0 (research; contact: cloudholic@pm.me)")


# Paths ------------------------------------------------------------------------

ROOT: Path = Path(os.environ.get("THESIS_ROOT", ".")).resolve()
CACHE_DIR: Path = Path(os.environ.get("THESIS_CACHE", ROOT / "cache")).resolve()
RESULT_DIR: Path = Path(os.environ.get("THESIS_RESULT", ROOT / "result")).resolve()
OSU_CACHE_DIR: Path =  CACHE_DIR / "osu"    # raw .osu files
DATA_CACHE_DIR: Path = CACHE_DIR / "data"   # prepared IRT arrays (.npz)

# Bump this when the underlying monthly DB dump changes;
# it is part of the prepared-data cache filename, so old caches are invalidated automatically.
DATA_TAG: str = os.environ.get("THESIS_DATA_TAG", "2026H1")


# Domain defaults --------------------------------------------------------------

RATE_GROUP_CLOCK: dist[str, float] = {"NM": 1.0, "DT": 1.5, "HT": 0.75}
VIEWS: dict[str, str] = {"score": "v_irt_score", "acc": "v_irt_acc"}
MANIA_KEYS: tuple[int, ...] = tuple(range(1, 19))


def ensure_dirs() -> None:
    for d in (CACHE_DIR, RESULT_DIR, OSU_CACHE_DIR, DATA_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)