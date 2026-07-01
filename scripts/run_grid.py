"""
Run the IRT grid for one or more keys and write weak-label CSVs.

uv run scripts/run_grid.py --keys 4 --jobs 6 --threads 2
"""

from __future__ import annotations

import argparse
from vsrg_irt.cpu import configure_cpu


def parse_keys(tokens, *, valid=range(1, 19)) -> list[int]:
    out: set[int] = set()
    for tok in tokens:
        for part in str(tok).replace(",", " ").split():
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                if lo > hi:
                    lo, hi = hi, lo
                out.update(range(lo, hi + 1))
            else:
                out.add(int(part))
    bad = sorted(k for k in out if k not in valid)
    if bad:
        raise ValueError(f"keys out of range {min(valid)}..{max(valid)}: {bad}")
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", nargs="+", default=None)
    ap.add_argument("--jobs", type=int, default=1, help="parallel cells (processes)")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads per worker")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--no-sr", action="store_true", help="skip star-rating attach")
    args = ap.parse_args()

    configure_cpu(args.threads)

    import polars as pl
    from vsrg_irt import config
    from vsrg_irt.irt.grid import run_grid
    from vsrg_irt.star_rating import make_sr_fn, OsuFetchError

    config.ensure_dirs()

    def attach(items: pl.DataFrame, sr_fn) -> pl.DataFrame:
        """Attach a star_rating column."""
        out = []
        for bid, rate in zip(items["beatmap_id"].to_list(), items["rate"].to_list()):
            try:
                out.append(sr_fn(int(bid), float(rate)))
            except OsuFetchError:
                out.append(None)

        return items.with_columns(pl.Series("star_rating", out, dtype=pl.Float64))

    keys = parse_keys(args.keys, valid=config.MANIA_KEYS) if args.keys else list(config.MANIA_KEYS)
    for key in keys:
        print(f"=== {key}K ===", flush=True)
        results = run_grid(key, jobs=args.jobs, threads_per_worker=args.threads, num_steps=args.steps)

        sr_fn = None
        if not args.no_sr:
            bids = {int(b) for r in results.values()
                    for b in r["items"]["beatmap_id"].to_list()}
            sr_fn = make_sr_fn(bids)

        for (model, response, sample), r in sorted(results.items()):
            items = (attach(r["items"], sr_fn) if sr_fn is not None
                     else r["items"].with_columns(pl.lit(None, dtype=pl.Float64).alias("star_rating")))
            stem = f"{key}Keys_{model}_{response}_{sample}"
            items.write_csv(config.RESULT_DIR / f"{stem}_items.csv")
            if "persons" in r:
                r["persons"].write_csv(config.RESULT_DIR / f"{stem}_persons.csv")

            print(f"   wrote {stem}", flush=True)


if __name__ == "__main__":
    main()