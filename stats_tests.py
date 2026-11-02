"""
Summary tables and statistical tests for the results CSVs.

    !python stats_tests.py --results results --baseline-of ens+TS

Writes to <results>/tables/:
    summary_<file>.csv     mean and std over seeds per (dataset, model, test set, ...)
    wilcoxon_<file>.csv    proposed model vs each other model, per metric:
                           median difference, 95% bootstrap CI, Wilcoxon p, Holm-adjusted p

About Wilcoxon with few pairs: with 5 seeds on one dataset, a two-sided
Wilcoxon signed-rank test cannot go below p = 0.0625. The script therefore
also pools pairs across datasets / test sets (e.g. 5 seeds x 3 datasets = 15
pairs) in the "pooled" rows. Report the CI and effect size alongside p.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Metrics where lower is better (used only to label the direction).
LOWER_BETTER = {"fpr", "fpr_at_thr", "ece", "brier", "nll", "aurc",
                "risk_at_cov_0.9", "risk_at_cov_0.95"}
METRICS = ["f1", "f1_at_thr", "fpr", "fpr_at_thr", "precision_at_thr", "auroc", "auprc",
           "tpr_at_fpr_1e-2", "tpr_at_fpr_1e-3", "ece", "brier", "nll", "aurc",
           "risk_at_cov_0.95", "shift_auroc_entropy", "shift_auroc_mi",
           "detection_rate_unseen", "unseen_auroc_entropy", "unseen_auroc_mi"]
GROUP_COLS = ["dataset", "source", "test_set", "base_rate", "held_out"]


def holm(pvals):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def bootstrap_ci(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, (n, len(d)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def compare(df, proposed, metrics, groups):
    """Paired comparison of `proposed` vs every other model, pairs = seeds x groups."""
    rows = []
    for other in sorted(set(df["model"]) - {proposed}):
        for met in metrics:
            a = df[df.model == proposed].set_index(groups + ["seed"])[met]
            b = df[df.model == other].set_index(groups + ["seed"])[met]
            pair = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
            if len(pair) < 3:
                continue
            d = (pair["a"] - pair["b"]).to_numpy()
            try:
                p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
            except ValueError:
                p = np.nan
            lo, hi = bootstrap_ci(d)
            rows.append({"proposed": proposed, "other": other, "metric": met,
                         "lower_is_better": met in LOWER_BETTER, "n_pairs": len(d),
                         "mean_diff": d.mean(), "median_diff": np.median(d),
                         "ci95_low": lo, "ci95_high": hi, "p_wilcoxon": p})
    out = pd.DataFrame(rows)
    if len(out):
        ok = out["p_wilcoxon"].notna()
        out.loc[ok, "p_holm"] = holm(out.loc[ok, "p_wilcoxon"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--proposed", default="ens+TS")
    args = ap.parse_args()
    os.makedirs(f"{args.results}/tables", exist_ok=True)

    files = sorted(glob.glob(f"{args.results}/results_*.csv"))
    if not files:
        raise SystemExit(f"No results_*.csv in {args.results}")

    for f in files:
        name = os.path.basename(f)[:-4]
        df = pd.read_csv(f)
        groups = [c for c in GROUP_COLS if c in df.columns]
        df[groups] = df[groups].fillna("-")
        metrics = [m for m in METRICS if m in df.columns and df[m].notna().any()]

        summary = df.groupby(groups + ["model"])[metrics].agg(["mean", "std"])
        summary.columns = [f"{m}_{s}" for m, s in summary.columns]
        summary.round(4).to_csv(f"{args.results}/tables/summary_{name}.csv")

        if args.proposed not in set(df["model"]):
            print(f"{name}: '{args.proposed}' not found, skipping tests")
            continue
        # per group (e.g. one dataset): pairs = seeds
        per = []
        for key, g in df.groupby(groups):
            res = compare(g, args.proposed, metrics, groups)
            if len(res):
                for c, v in zip(groups, key if isinstance(key, tuple) else [key]):
                    res[c] = v
                per.append(res)
        # pooled across groups: pairs = seeds x groups
        pooled = compare(df, args.proposed, metrics, groups)
        pooled[groups] = "pooled"
        out = pd.concat(per + [pooled], ignore_index=True)
        out.round(5).to_csv(f"{args.results}/tables/wilcoxon_{name}.csv", index=False)
        print(f"{name}: wrote summary and tests ({len(out)} comparisons)")


if __name__ == "__main__":
    main()
