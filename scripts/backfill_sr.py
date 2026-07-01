"""
Backfill null star_rating in the grid's *_items.csv.

uv run scripts/backfill_sr.py
uv run scripts/backfill_sr.py --result-dir result --jobs 8 --refresh
"""

from __future__ import annotations

import argparse
import glob
import os

import polars as pl

from vsrg_irt import config
from vsrg_irt.star_rating import prefetch, make_sr_fn, OsuFetchError


def _null_pairs(df: pl.DataFrame) -> pl.DataFrame:
    return (df.filter(pl.col("star_rating").is_null())
              .select(["beatmap_id", "rate"]).unique())


def main() -> None:
    ap = argparse.ArgumentParser(description="refill null star_rating in *_items.csv")
    ap.add_argument("--result-dir", default=None, help="default: config.RESULT_DIR")
    ap.add_argument("--jobs", type=int, default=8, help="parallel download workers")
    ap.add_argument("--pattern", default="*_items.csv")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download even if a .osu is already cached")
    args = ap.parse_args()

    result_dir = args.result_dir or str(config.RESULT_DIR)
    paths = sorted(glob.glob(os.path.join(result_dir, args.pattern)))
    if not paths:
        print(f"no items CSVs ({args.pattern}) in {result_dir}")
        return

    # 1. load files; gather distinct (beatmap_id, rate) whose SR is null
    frames: dict[str, pl.DataFrame] = {}
    needs = []
    for p in paths:
        df = pl.read_csv(p)
        if "star_rating" not in df.columns:
            continue

        frames[p] = df
        if df["star_rating"].null_count():
            needs.append(_null_pairs(df))

    if not needs:
        print("no null star_rating rows - nothing to backfill")
        return

    need = pl.concat(needs).unique()
    bids = sorted({int(b) for b in need["beatmap_id"].to_list()})
    n_files_with_null = sum(1 for f in frames.values() if f["star_rating"].null_count())
    print(f"[scan] {need.height} distinct (map,rate) need SR across {n_files_with_null} files; "
          f"{len(bids)} maps to (re)download")

    # 2. download the maps (rate-limited + retried inside prefetch); dead maps land in `failed`
    ok, failed = prefetch(bids, jobs=args.jobs, refresh=args.refresh)
    ok_set = set(ok)
    sr_fn = make_sr_fn()   # no prefetch here; sr() is a cache hit for what we just downloaded

    # 3. recompute SR for each recoverable (map, rate)
    rows = []
    for bid, rate in zip(need["beatmap_id"].to_list(), need["rate"].to_list()):
        bid = int(bid)
        if bid not in ok_set:
            continue

        try:
            rows.append((bid, float(rate), float(sr_fn(bid, float(rate)))))
        except OsuFetchError:
            pass

    patch = (pl.DataFrame(rows, schema=["beatmap_id", "rate", "sr_new"], orient="row")
             if rows else
             pl.DataFrame(schema={"beatmap_id": pl.Int64, "rate": pl.Float64, "sr_new": pl.Float64}))
    print(f"[fetch] recovered {patch.height} maps; {len(failed)} still failing (dead on all mirrors)")

    if patch.height == 0:
        print("nothing recovered; no files written")
        return

    # 4. patch each file in place - coalesce fills only the null cells, keeps existing SR
    total_before = total_after = 0
    for p, df in frames.items():
        before = df["star_rating"].null_count()
        if before == 0:
            continue

        merged = (df.join(patch, on=["beatmap_id", "rate"], how="left")
                    .with_columns(pl.coalesce(["star_rating", "sr_new"]).alias("star_rating"))
                    .drop("sr_new"))
        after = merged["star_rating"].null_count()

        total_before += before
        total_after += after
        if after < before:
            merged.write_csv(p)
            print(f"  {os.path.basename(p)}: {before} -> {after} null")

    print(f"[done] filled {total_before - total_after} rows; {total_after} still null")


if __name__ == "__main__":
    main()