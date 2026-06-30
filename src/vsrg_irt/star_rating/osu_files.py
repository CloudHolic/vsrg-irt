from __future__ import annotations

import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPException
from pathlib import Path
from typing import Iterable

from .. import config


_HEADER = b"osu file format"


class OsuFetchError(Exception):
    """Raised when a .osu file cannot be retrieved."""


class _RateLimiter:
    def __init__(self, rate_per_sec: float):
        self.min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self.min_interval <= 0.0:
            return

        with self._lock:
            now = time.monotonic()
            t = max(self._next, now)
            self._next = t + self.min_interval

        wait = t - time.monotonic()
        if wait > 0:
            time.sleep(wait)


def osu_path(beatmap_id: int) -> Path:
    return config.OSU_CACHE_DIR / f"{int(beatmap_id)}.osu"


def _backoff(attempt: int, base: float, cap: float) -> float:
    return random.uniform(0.0, min(cap, base * (2.0 ** attempt)))


def _retry_after(err: urllib.error.HTTPError) -> float | None:
    val = err.headers.get("Retry-After") if err.headers else None
    if val is None:
        return None

    try:
        return float(val)
    except ValueError:
        return None


def _try_mirror(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        body = f.read()

    if not body.lstrip().startswith(_HEADER):
        raise OsuFetchError("response is not a .osu file (error/placeholder page)")

    return body


def fetch_osu(beatmap_id: int, *, mirrors: Iterable[str] | None = None,
              limiter: "_RateLimiter | None" = None, retries: int=5,
              base_delay: float=1.0, max_delay: float=30.0,
              timeout: float=20.0, refresh: bool=False) -> Path:
    path = osu_path(beatmap_id)
    if path.exists() and not refresh and path.stat().st_size > 0:
        return path

    mirrors = list(mirrors or config.OSU_MIRRORS)
    last: Exception | None = None

    for tmpl in mirrors:
        url = tmpl.format(beatmap_id=int(beatmap_id))

        for attempt in range(retries + 1):
            if limiter is not None:
                limiter.acquire()

            try:
                data = _try_mirror(url, timeout)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".osu.part")
                tmp.write_bytes(data)
                tmp.replace(path)
                return path
            except urllib.error.HTTPError as e:
                last = e
                if e.code == 404 or not (e.code == 429 or 500 <= e.code < 600):
                    break

                delay = _retry_after(e) if e.code == 429 else None
                if delay is None:
                    delay = _backoff(attempt, base_delay, max_delay)
            except (urllib.error.URLError, TimeoutError, HTTPException, OsuFetchError) as e:
                last = e
                delay = _backoff(attempt, base_delay, max_delay)

            if attempt < retries:
                time.sleep(delay)

    raise OsuFetchError(f"{beatmap_id}: all mirrors failed ({last})")


def prefetch(beatmap_ids: Iterable[int], *, jobs: int=8, rate: float=10.0, **fetch_kw) -> tuple[list[int], dict[int, str]]:
    ids = sorted({int(b) for b in beatmap_ids})
    limiter = _RateLimiter(rate)
    ok: list[int] = []
    failed: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(fetch_osu, b, limiter=limiter, **fetch_kw): b for b in ids}
        for fut in as_completed(futs):
            b = futs[fut]
            try:
                fut.result()
                ok.append(b)
            except OsuFetchError as e:
                failed[b] = str(e)

    if failed:
        print(f"prefetch: {len(ok)} ok, {len(failed)} failed (e.g. "
              f"{next(iter(failed.items()))})", flush=True)

    return ok, failed


def make_sr_fn(beatmap_ids: Iterable[int] | None = None, *, prefetch_jobs: int=8, **fetch_kw):
    from .algorithm import calculate_star_rating

    if beatmap_ids is not None:
        prefetch(beatmap_ids, jobs=prefetch_jobs, **fetch_kw)

    def sr(beatmap_id: int, clock_rate: float) -> float:
        path = fetch_osu(beatmap_id, **fetch_kw)
        return float(calculate_star_rating(str(path), clock_rate))

    return sr
