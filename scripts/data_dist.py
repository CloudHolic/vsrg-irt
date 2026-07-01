"""
Data-distribution / connectivity diagnostics for the osu! dumped.

uv run scripts/data_dist.py --response score --native
uv run scripts/data_dist.py --response acc --all --keys 4-10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from vsrg_irt import config


# Style (colorblind-safe) ------------------------------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.dpi": 130, "axes.titleweight": "bold",
                     "font.size": 11, "figure.titlesize": 15})
SAMPLE_PALETTE = {"all": "#4C72B0", "top": "#C44E52", "random": "#55A868"}
COUNT_THRESHOLDS = [20, 50]
MIN_KEY_CELLS = 5000


class Ctx:
    view: str
    native_only: bool
    is_accuracy: bool
    out_dir: Path
    keys_override: list[int] | None

    @property
    def tag(self) -> str:
        return f"{self.view} ({'native' if self.native_only else 'all'})"


CTX = Ctx()


def _where(*preds: str) -> str:
    p = [x for x in preds if x]
    return ("WHERE " + " AND ".join(p)) if p else ""


def _native_pred() -> str:
    return "NOT is_convert" if CTX.native_only else ""


def _read(sql: str) -> pd.DataFrame:
    return pl.read_database_uri(sql, config.DSN, engine="connectorx").to_pandas()


def load_responses() -> pd.DataFrame:
    q = f"""
        SELECT mania_keys, rate_group, in_top, in_random, response
        FROM {CTX.view} {_where(_native_pred())}
    """
    df = _read(q)
    df["response"] = pd.to_numeric(df["response"], errors="coerce")
    return df.dropna(subset=["response"])


def load_item_counts() -> pd.DataFrame:
    q = f"""
        SELECT beatmap_id, rate_group, mania_keys,
               COUNT(*)                          AS n_all,
               COUNT(*) FILTER (WHERE in_top)    AS n_top,
               COUNT(*) FILTER (WHERE in_random) AS n_random
        FROM {CTX.view} {_where(_native_pred())}
        GROUP BY beatmap_id, rate_group, mania_keys
    """
    return _read(q)


def load_user_counts() -> pd.DataFrame:
    q = f"""
        SELECT user_id,
               bool_or(in_top)    AS in_top,
               bool_or(in_random) AS in_random,
               COUNT(*)           AS n_items
        FROM {CTX.view} {_where(_native_pred())}
        GROUP BY user_id
    """
    return _read(q)


def pick_keys(df: pd.DataFrame) -> list[int]:
    if CTX.keys_override is not None:
        return list(CTX.keys_override)
    vc = df["mania_keys"].value_counts()
    keys = sorted(int(k) for k, n in vc.items() if n >= MIN_KEY_CELLS)
    return keys or sorted(int(k) for k in vc.index[:2])


def with_sample_groups(df: pd.DataFrame) -> pd.DataFrame:
    base = df.assign(sample="all")
    top = df[df["in_top"]].assign(sample="top")
    rnd = df[df["in_random"]].assign(sample="random")
    return pd.concat([base, top, rnd], ignore_index=True)


def acc_tail(x: pd.Series, eps: float = 1e-4) -> pd.Series:
    return -np.log10(np.clip(1.0 - x, eps, 1.0))


def connectivity_report():
    q = f"SELECT user_id, beatmap_id, rate_group FROM {CTX.view} {_where(_native_pred())}"
    e = _read(q)
    e["item"] = e["beatmap_id"].astype(str) + "|" + e["rate_group"].astype(str)

    u_idx = {u: i for i, u in enumerate(e["user_id"].unique())}
    i_idx = {it: j for j, it in enumerate(e["item"].unique())}
    nu, ni = len(u_idx), len(i_idx)
    rows = e["user_id"].map(u_idx).to_numpy()
    cols = e["item"].map(i_idx).to_numpy() + nu

    N = nu + ni
    data = np.ones(len(e) * 2)
    R = np.concatenate([rows, cols]); C = np.concatenate([cols, rows])
    adj = coo_matrix((data, (R, C)), shape=(N, N)).tocsr()
    ncomp, labels = connected_components(adj, directed=False)
    sizes = pd.Series(labels).value_counts()
    largest = sizes.iloc[0] / N

    lines = [
        f"view={CTX.tag}",
        f"nodes={N:,} (users {nu:,} + items {ni:,})",
        f"components={ncomp:,}",
        f"largest_CC={largest:.4f} of nodes",
        f"top_component_sizes={sizes.head().to_dict()}",
    ]
    (CTX.out_dir / "01_connectivity.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"  components={ncomp:,}  largest_CC={largest:.4f}")
    return ncomp, largest


def plot_item_count_dist(df_items: pd.DataFrame, keys: list[int]):
    sub = df_items[df_items["mania_keys"].isin(keys)]
    ncols = min(3, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
    axes = axes.flatten()

    for i, k in enumerate(keys):
        ax = axes[i]
        d = sub[sub["mania_keys"] == k]

        for col, color, lab in [("n_top", "#C44E52", "top"), ("n_random", "#55A868", "random")]:
            n = d[col]; n = n[n > 0]
            if len(n) == 0:
                continue
            ax.hist(np.log10(n.clip(lower=1)), bins=40, histtype="step", lw=1.8,
                    color=color, label=f"{lab} (med {int(n.median())})")

        for t in COUNT_THRESHOLDS:
            ax.axvline(np.log10(t), color="#888", ls="--", lw=1.0)

        nr = d["n_random"]
        surv = " ".join(f">={t}:{(nr >= t).mean():.0%}" for t in COUNT_THRESHOLDS)

        ax.set_title(f"{k}K - random {surv}", fontsize=11)
        ax.set_xlabel("log10(responses per item)")
        ax.set_ylabel("item count")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{10**v:.0f}"))
        ax.legend(fontsize=8)

    for j in range(len(keys), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f"Responses per item, top vs random - {CTX.tag}", y=1.02)
    fig.tight_layout()
    fig.savefig(CTX.out_dir / "02_item_count_dist.png", bbox_inches="tight")

    plt.close(fig)


def coverage_table(df_items: pd.DataFrame, keys: list[int]) -> pd.DataFrame:
    rows = []
    for k in keys:
        d = df_items[df_items["mania_keys"] == k]
        r = {"mania_keys": k, "n_items": len(d)}
        for t in COUNT_THRESHOLDS:
            r[f"all>={t}"] = int((d["n_all"] >= t).sum())
            r[f"random>={t}"] = int((d["n_random"] >= t).sum())
            r[f"top_only>={t}"] = int(((d["n_all"] >= t) & (d["n_random"] < t)).sum())
        rows.append(r)

    out = pd.DataFrame(rows)
    out.to_csv(CTX.out_dir / "03_coverage_by_sample.csv", index=False)

    return out


def plot_user_count_dist(df_users: pd.DataFrame):
    long = with_sample_groups(df_users)
    fig, ax = plt.subplots(figsize=(9, 5))

    for s in ["all", "top", "random"]:
        n = long.loc[long["sample"] == s, "n_items"]
        if len(n) == 0:
            continue
        ax.hist(np.log10(n.clip(lower=1)), bins=40, histtype="step", lw=2,
                color=SAMPLE_PALETTE[s], label=f"{s} (median {int(n.median())})")

    ax.set_title(f"Responses (items) per user - {CTX.tag}")
    ax.set_xlabel("log10(items per user)")
    ax.set_ylabel("user count")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, p: f"{10**v:.0f}"))
    ax.legend()

    fig.tight_layout()
    fig.savefig(CTX.out_dir / "04_user_count_dist.png", bbox_inches="tight")
    plt.close(fig)


def plot_response_distributions(df: pd.DataFrame, keys: list[int]):
    sub = df[df["mania_keys"].isin(keys)].copy()
    long = with_sample_groups(sub)
    metric = "accuracy" if CTX.is_accuracy else "score_norm"

    g = sns.displot(
        data=long, x="response", hue="sample", hue_order=list(SAMPLE_PALETTE),
        palette=SAMPLE_PALETTE, col="mania_keys", col_order=keys, col_wrap=3,
        kind="kde", common_norm=False, fill=False,
        facet_kws=dict(sharex=True, sharey=False), height=2.8, aspect=1.4, warn_singular=False)
    g.set_titles("{col_name}K")
    g.set_axis_labels(metric, "density")
    g.figure.suptitle(f"Response distribution by key - {CTX.tag}", y=1.02)
    g.figure.savefig(CTX.out_dir / "05_response_kde.png", bbox_inches="tight")
    plt.close(g.figure)

    long2 = long.assign(resp_tail=acc_tail(long["response"]))
    g2 = sns.displot(
        data=long2, x="resp_tail", hue="sample", hue_order=list(SAMPLE_PALETTE),
        palette=SAMPLE_PALETTE, col="mania_keys", col_order=keys, col_wrap=3,
        kind="kde", common_norm=False, fill=False,
        facet_kws=dict(sharex=True, sharey=False), height=2.8, aspect=1.4, warn_singular=False)
    g2.set_titles("{col_name}K")
    g2.set_axis_labels(f"-log10(1 - {metric})  (-> max)", "density")
    g2.figure.suptitle(f"{metric} upper tail by key - {CTX.tag}", y=1.02)
    g2.figure.savefig(CTX.out_dir / "06_response_tail_kde.png", bbox_inches="tight")

    plt.close(g2.figure)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", choices=["score", "acc"], default="score")

    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--native", dest="native", action="store_true", default=True,
                     help="native charts only (NOT is_convert) [default]")
    grp.add_argument("--all", dest="native", action="store_false", help="include converts")

    ap.add_argument("--keys", nargs="+", default=None, help="e.g. 4-10  (default: auto)")
    args = ap.parse_args()

    from vsrg_irt.specs import parse_keys

    CTX.view = config.VIEWS[args.response]
    CTX.native_only = args.native
    CTX.is_accuracy = args.response == "acc"
    CTX.keys_override = parse_keys(args.keys, valid=config.MANIA_KEYS) if args.keys else None
    CTX.out_dir = config.RESULT_DIR / f"dist_{CTX.view}_{'native' if CTX.native_only else 'all'}"
    CTX.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] responses from {CTX.tag} ...")

    resp = load_responses()
    keys = pick_keys(resp)
    print(f"[load] {len(resp):,} cells - keys used = {keys}")

    print("[1/3] connectivity ...")
    connectivity_report()

    print("[2/3] item / user counts ...")
    items = load_item_counts()
    plot_item_count_dist(items, keys)
    cov = coverage_table(items, keys)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(cov.to_string(index=False))
    plot_user_count_dist(load_user_counts())

    print("[3/3] response tail / distribution ...")
    plot_response_distributions(resp, keys)
    print(f"\ndone. {CTX.out_dir.resolve()}")


if __name__ == "__main__":
    main()