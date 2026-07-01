"""
IRT & star-rating diagnostics, comparing zoi / beta3 / beta4.

Reads the grid's *_items.csv / *_persons.csv for one key and renders:

    A. calibration difficulty vs star rating    - IRT difficulty vs SR
    B. standardized residual vs star rating     - z = (difficulty - g(SR)) / posterior SD
    C. random vs all                            - sample-selection shift (random vs all fit)
    D. cross-set theta agreement                - cross-set ability agreement; beta3 vs beta4
    E. weak label width vs response count       - posterior SD of difficulty vs item response count

uv run scripts/make_plots.py --key 4
"""

from __future__ import annotations

import argparse
import glob
import os
import warnings
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vsrg_irt import config


# Column names
C_BEATMAP = "beatmap_id"
C_RATE = "rate_group"
C_SR = "star_rating"
C_DIFF_MEAN = "delta_mean"
C_DIFF_SD = "delta_std"
C_N_RESP = "n_resp"
C_USER = "user_id"
C_THETA_MEAN = "theta_mean"

DIFFICULTY_SPACE = {"zoi": "real", "beta3": "unit", "beta4": "unit"}

SERIES_STYLE = {
    ("zoi", "score"):   dict(c="#4C72B0", marker="o", ls="-",  label="ZOI score"),
    ("zoi", "acc"):     dict(c="#DD8452", marker="s", ls="--", label="ZOI acc"),
    ("beta3", "score"): dict(c="#55A868", marker="^", ls=":",  label="beta3 score"),
    ("beta4", "score"): dict(c="#8172B3", marker="D", ls="-.", label="beta4 score"),
}

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 11,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.labelsize": 11,
    "legend.fontsize": 9, "axes.grid": True, "grid.alpha": 0.3,
    "axes.unicode_minus": False,
})


# Frame / numeric helpers
def _has(df, *cols):
    return df is not None and all(c in df.columns for c in cols)


def _np(df, col) -> np.ndarray:
    return df.get_column(col).to_numpy().astype(float)


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _rank_data(a):
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")

    r = np.empty(len(a), float)
    r[order] = np.arange(len(a), dtype=float)
    return r


def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan

    try:
        return float(spearmanr(a[m], b[m]).statistic)
    except Exception:
        return float(np.corrcoef(_rank_data(a[m]), _rank_data(b[m]))[0, 1])


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


def _pava(ys, ws):
    val, wt, cnt = [], [], []
    for v, w in zip(ys, ws):
        cv, cw, cc = float(v), float(w), 1
        while val and val[-1] > cv:
            pv, pw, pc = val.pop(), wt.pop(), cnt.pop()
            cv = (pv * pw + cv * cw) / (pw + cw)
            cw += pw
            cc += pc

        val.append(cv)
        wt.append(cw)
        cnt.append(cc)

    out = np.empty(len(ys))

    i = 0
    for v, c in zip(val, cnt):
        out[i:i + c] = v
        i += c

    return out


def _isotonic(x, y, w=None):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    g = np.full_like(y, np.nan)

    if m.sum() < 5:
        return g

    try:
        ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
        g[m] = ir.fit_transform(x[m], y[m], sample_weight=w[m])
        return g
    except Exception:
        order = np.argsort(x[m], kind="mergesort")
        fit = _pava(y[m][order], w[m][order])
        gm = np.empty(m.sum())
        gm[order] = fit
        g[m] = gm
        return g


def _inv_var_w(sd):
    sd = np.asarray(sd, float)
    med = np.median(sd[np.isfinite(sd) & (sd > 0)]) if np.isfinite(sd).any() else 1.0
    floor = (med if np.isfinite(med) else 1.0) * 1e-2
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, floor)
    return 1.0 / np.maximum(sd, floor) ** 2


def _roll_med(y, w):
    """Centered rolling median over a 1D array."""
    y = np.asarray(y, float)
    n = len(y)
    w = max(1, w)
    h = w // 2

    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(y[max(0, i - h):min(n, i + h + 1)])

    return out


def _diff_link(model, mean, sd):
    """Difficulty mean/sd on the link scale."""
    mean = np.asarray(mean, float)
    sd = None if sd is None else np.asarray(sd, float)
    if DIFFICULTY_SPACE.get(model) == "unit":
        m = np.clip(mean, 1e-6, 1 - 1e-6)
        return _logit(m), (None if sd is None else sd / (m * (1 - m)))

    return mean, sd


# Loading
MODELS = ["zoi", "beta3", "beta4"]
RESPS = ["score", "acc"]
SAMPLES = ["random", "all"]
VALID = {("zoi", "score"), ("zoi", "acc"), ("beta3", "score"), ("beta4", "score")}
UNIT_MODELS = ["beta3", "beta4"]


@dataclass
class Cell:
    items: "pl.DataFrame | None"
    persons: "pl.DataFrame | None"


def _find(result_dir, key, model, resp, sample, kind):
    pats = [
        f"{key}Keys_{model}_{resp}_{sample}_{kind}.csv",
        f"{key}keys_{model}_{resp}_{sample}_{kind}.csv",
        f"*{key}*{model}*{resp}*{sample}*{kind}*.csv",
    ]

    for p in pats:
        hits = glob.glob(os.path.join(result_dir, p))
        if hits:
            return sorted(hits)[0]

    return None


def load_cells(result_dir, key):
    cells = {}

    for model in MODELS:
        for resp in RESPS:
            if (model, resp) not in VALID:
                continue
            for sample in SAMPLES:
                ip = _find(result_dir, key, model, resp, sample, "items")
                pp = _find(result_dir, key, model, resp, sample, "persons")

                if ip is None and pp is None:
                    continue

                cells[(model, resp, sample)] = Cell(
                    pl.read_csv(ip) if ip else None,
                    pl.read_csv(pp) if pp else None
                )

    return cells


def items_of(cells, model, resp, sample):
    c = cells.get((model, resp, sample))
    return c.items if c else None


def persons_of(cells, model, resp, sample):
    c = cells.get((model, resp, sample))
    return c.persons if c else None


# Figure A. - Difficulty vs star rating
def _calib_series(ax, it, model, resp, summary):
    if not _has(it, C_SR, C_DIFF_MEAN):
        return False

    sr = _np(it, C_SR)
    ym, ys = _diff_link(model, _np(it, C_DIFF_MEAN), _np(it, C_DIFF_SD) if _has(it, C_DIFF_SD) else None)
    st = SERIES_STYLE[(model, resp)]
    rho = _spearman(sr, ym)

    ax.scatter(sr, ym, s=16, c=st["c"], marker=st["marker"], alpha=0.5,
               label=f"{st['label']} (Spearman={rho:.2f})")
    g = _isotonic(sr, ym, _inv_var_w(ys) if ys is not None else None)
    o = np.argsort(sr)

    ax.plot(sr[o], g[o], color=st["c"], ls=st["ls"], lw=2)
    summary[f"calibration_{model}_{resp}_spearman_difficulty_vs_sr"] = rho

    return True


def fig_calibration(cells, summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    drew = any([_calib_series(ax, items_of(cells, "zoi", r, "all"), "zoi", r, summary)
                for r in RESPS])

    ax.set_title("ZOI difficulty (theta*) vs star rating")
    ax.set_xlabel("star rating"); ax.set_ylabel("theta*  (N(0,1) scale)")
    ax.legend(loc="upper left") if drew else ax.text(0.5, 0.5, "no ZOI all-sample items",
                                                     ha="center", transform=ax.transAxes)
    ax = axes[1]

    drew = any([_calib_series(ax, items_of(cells, m, "score", "all"), m, "score", summary)
               for m in UNIT_MODELS])
    ax.set_title("beta3 vs beta4 difficulty (logit delta) vs star rating")
    ax.set_xlabel("star rating"); ax.set_ylabel("logit delta  ((0,1) scale)")
    ax.legend(loc="upper left") if drew else ax.text(0.5, 0.5, "no beta3/beta4 score items",
                                                     ha="center", transform=ax.transAxes)

    fig.suptitle("Calibration: IRT difficulty vs star rating (weighted isotonic link, all-sample)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    return fig


# Figure B - Standardized residual vs star rating
def fig_residual(cells, summary):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    drew = False

    for (model, resp) in [("zoi", "score"), ("zoi", "acc"), ("beta3", "score"), ("beta4", "score")]:
        it = items_of(cells, model, resp, "all")
        if not _has(it, C_SR, C_DIFF_MEAN, C_DIFF_SD):
            continue
        sr = _np(it, C_SR)
        ym, ys = _diff_link(model, _np(it, C_DIFF_MEAN), _np(it, C_DIFF_SD))
        g = _isotonic(sr, ym, _inv_var_w(ys))
        z = (ym - g) / ys
        st = SERIES_STYLE[(model, resp)]
        ax.scatter(sr, z, s=16, c=st["c"], marker=st["marker"], alpha=0.55,
                   label=f"{st['label']}  (|z|>2: {np.nanmean(np.abs(z) > 2)*100:.0f}%)")
        summary[f"residual_{model}_{resp}_outlier_fraction_absz_gt_2"] = float(np.nanmean(np.abs(z) > 2))
        drew = True

    ax.axhline(0, color="k", lw=1)

    for h in (-2, 2):
        ax.axhline(h, color="grey", ls="--", lw=1)

    ax.set_title("Standardized residual  z = (difficulty - g(SR)) / posterior SD  vs star rating")
    ax.set_xlabel("star rating"); ax.set_ylabel("z  (dimensionless)")

    if drew:
        ax.legend(loc="upper right")
        ax.text(0.01, 0.98, "|z|>2 : difficulty disagrees with SR beyond its own posterior interval",
                transform=ax.transAxes, va="top", fontsize=9, color="#555")
    else:
        ax.text(0.5, 0.5, "requires delta_sd", ha="center", transform=ax.transAxes)

    fig.tight_layout()
    return fig


# Figure C - Difficulty: random vs all (sample-selection MNAR)
def _merge_all_random(a, b):
    keys = [k for k in (C_BEATMAP, C_RATE) if k in a.columns and k in b.columns]
    if not keys or C_DIFF_MEAN not in a.columns or C_DIFF_MEAN not in b.columns:
        return None

    aa = a.select(keys + [pl.col(C_DIFF_MEAN).alias("v_all")])
    bb = b.select(keys + [pl.col(C_DIFF_MEAN).alias("v_rnd")])

    return aa.join(bb, on=keys, how="inner")


def _mnar_series(ax, cells, model, resp, summary, fmt):
    ia, ir = items_of(cells, model, resp, "all"), items_of(cells, model, resp, "random")
    if ia is None or ir is None:
        return False

    m = _merge_all_random(ia, ir)
    if m is None or m.height < 3:
        return False

    v_all = m.get_column("v_all").to_numpy().astype(float)
    v_rnd = m.get_column("v_rnd").to_numpy().astype(float)
    shift = float(np.median(v_all - v_rnd))
    st = SERIES_STYLE[(model, resp)]

    ax.scatter(v_rnd, v_all, s=16, c=st["c"], marker=st["marker"], alpha=0.5,
               label=f"{st['label']}  median shift={shift:{fmt}}")
    summary[f"mnar_{model}_{resp}_median_shift_all_minus_random"] = shift

    return True


def fig_mnar(cells, summary):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    ax = axes[0]

    drew = any([_mnar_series(ax, cells, "zoi", r, summary, "+.2f")
                for r in RESPS])
    if drew:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]); hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="k", ls="--", lw=1, label="identity")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "no zoi all/random pair", ha="center", transform=ax.transAxes)

    ax.set_title("ZOI theta*: random vs all")
    ax.set_xlabel("theta*  (random fit)"); ax.set_ylabel("theta*  (all fit)")
    ax = axes[1]

    drew = any([_mnar_series(ax, cells, m, "score", summary, "+.3f")
                for m in UNIT_MODELS])
    if drew:
        ax.plot([0, 1], [0, 1], color="k", ls="--", lw=1, label="identity")
        ax.legend(loc="upper left")
    else:
        ax.text(0.5, 0.5, "no beta3/beta4 all/random pair", ha="center", transform=ax.transAxes)

    ax.set_title("beta3 / beta4 delta: random vs all")
    ax.set_xlabel("delta  (random fit)"); ax.set_ylabel("delta  (all fit)")

    fig.suptitle("Sample-selection MNAR (within-user chart choice & failure censoring NOT captured here)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    return fig


# Figure D - Cross-set ability (theta) agreement; beta4 fix vs beta3 break
def _theta_table(cells, model, resp):
    p = persons_of(cells, model, resp, "all")
    if not _has(p, C_USER, C_THETA_MEAN):
        return None

    keep = [C_USER, C_THETA_MEAN] + ([C_N_RESP] if C_N_RESP in p.columns else [])
    return p.select(keep).rename({C_USER: "user_id", C_THETA_MEAN: "theta"})


def fig_cross_set(cells, summary):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    zs = _theta_table(cells, "zoi", "score")
    za = _theta_table(cells, "zoi", "acc")
    b3 = _theta_table(cells, "beta3", "score")
    b4 = _theta_table(cells, "beta4", "score")

    def panel(ax, A, B, x_label, y_label, use_rank, title, key):
        if A is None or B is None:
            ax.text(0.5, 0.5, "missing persons", ha="center", transform=ax.transAxes)
            ax.set_title(title)
            return

        m = A.rename({"theta": "a"}).join(B.rename({"theta": "b"}), on="user_id", how="inner")
        if m.height < 3:
            ax.text(0.5, 0.5, "insufficient overlap", ha="center", transform=ax.transAxes)
            ax.set_title(title)
            return

        x = m.get_column("a").to_numpy().astype(float)
        y = m.get_column("b").to_numpy().astype(float)
        rho = _spearman(x, y)
        nc = "n_resp" if "n_resp" in m.columns else None
        px, py = (_rank_data(x), _rank_data(y)) if use_rank else (x, y)
        sc = ax.scatter(px, py, s=14, alpha=0.55,
                        c=(np.log10(m.get_column(nc).to_numpy().astype(float) + 1) if nc else "#4C72B0"),
                        cmap=("viridis" if nc else None))

        if nc:
            fig.colorbar(sc, ax=ax, label="log10(user response count + 1)")

        ax.set_xlabel(x_label); ax.set_ylabel(y_label)
        sub = f"Spearman={rho:.2f}" + ("" if use_rank else f", Pearson={_pearson(x, y):.2f}")
        ax.set_title(f"{title}\n{sub}")
        summary[key] = rho

    panel(axes[0], zs, za, "ZOI score theta", "ZOI acc theta", False,
          "Reference reliability (both N(0,1))", "crossset_zoiScore_vs_zoiAcc_theta_spearman")
    panel(axes[1], zs, b4, "ZOI score theta (rank)", "beta4 score theta (rank)", True,
          "beta4 fix: agrees with ZOI?", "crossset_zoiScore_vs_beta4_theta_spearman")
    panel(axes[2], zs, b3, "ZOI score theta (rank)", "beta3 score theta (rank)", True,
          "beta3 baseline: sign flip", "crossset_zoiScore_vs_beta3_theta_spearman")

    fig.suptitle("Cross-set ability (theta) agreement vs the trusted ZOI reference  "
                 "(panel 2 positive & > panel 3 => beta4 fixes beta3's sign flip)",
                 fontsize=13.5, fontweight="bold")
    fig.tight_layout()
    return fig


# Figure E - Weak-label width: posterior SD vs response count
def fig_label_quality(cells, interval_scale):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    drew = False

    for (model, resp) in [("zoi", "score"), ("zoi", "acc"), ("beta3", "score"), ("beta4", "score")]:
        it = items_of(cells, model, resp, "all")
        if not _has(it, C_DIFF_MEAN, C_DIFF_SD, C_N_RESP):
            continue

        _, sd = _diff_link(model, _np(it, C_DIFF_MEAN), _np(it, C_DIFF_SD))
        n = _np(it, C_N_RESP)

        med = np.nanmedian(sd)
        if not np.isfinite(med) or med == 0:
            continue

        y = sd / med
        st = SERIES_STYLE[(model, resp)]
        ax.scatter(np.log10(n + 1), y, s=14, c=st["c"], marker=st["marker"], alpha=0.45, label=st["label"])

        o = np.argsort(n)
        trend = _roll_med(y[o], max(5, len(o) // 10))
        ax.plot(np.log10(n[o] + 1), trend, color=st["c"], ls=st["ls"], lw=2)
        drew = True

    if drew:
        ax.axhline(interval_scale, color="grey", ls=":", lw=1.5,
                   label=f"x interval_scale={interval_scale} (NUTS-corrected weak-label width)")
        ax.set_yscale("log"); ax.legend(loc="upper right")
    else:
        ax.text(0.5, 0.5, "requires delta_sd and n_resp", ha="center", transform=ax.transAxes)

    ax.set_title("Weak-label width: posterior SD of difficulty vs item response count")
    ax.set_xlabel("log10(item response count + 1)")
    ax.set_ylabel("posterior SD / series median")

    fig.tight_layout()
    return fig


BUILDERS = [
    ("calibration_difficulty_vs_starrating", lambda c, s, a: fig_calibration(c, s)),
    ("standardized_residual_vs_starrating",  lambda c, s, a: fig_residual(c, s)),
    ("mnar_difficulty_random_vs_all",        lambda c, s, a: fig_mnar(c, s)),
    ("crossset_theta_agreement",             lambda c, s, a: fig_cross_set(c, s)),
    ("weaklabel_width_vs_response_count",    lambda c, s, a: fig_label_quality(c, a.interval_scale)),
]


def main():
    ap = argparse.ArgumentParser(description="IRT vs star-rating diagnostics (5 figures)")
    ap.add_argument("--result-dir", default=None)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--interval-scale", type=float, default=1.5)
    args = ap.parse_args()

    result_dir = args.result_dir
    if result_dir is None:
        result_dir = str(config.RESULT_DIR)

    out_dir = args.out_dir or os.path.join(result_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    cells = load_cells(result_dir, args.key)
    if not cells:
        print(f"[!] no CSVs for key={args.key} in {result_dir}")
        return
    print(f"[load] key={args.key}: cells = {sorted('.'.join(k) for k in cells)}")

    summary = {}
    for stem, fn in BUILDERS:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fig = fn(cells, summary, args)
            fig.savefig(os.path.join(out_dir, f"{args.key}Keys_{stem}.png"), bbox_inches="tight")
            plt.close(fig)
            print(f"   [ok] {stem}")
        except Exception as e:
            print(f"   [skip] {stem}: {type(e).__name__}: {e}")

    spath = os.path.join(out_dir, f"{args.key}Keys_diagnostics_summary.txt")
    with open(spath, "w", encoding="utf-8") as f:
        f.write(f"key={args.key}\n")
        for k, v in summary.items():
            f.write(f"{k}\t{v}\n")

    print(f"[summary] {os.path.abspath(spath)}")


if __name__ == "__main__":
    main()
