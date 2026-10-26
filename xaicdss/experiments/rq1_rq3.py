"""RQ1: do the explanation methods disagree?
RQ3: is low agreement a warning that the prediction is wrong, or that the
patient is unusual?

Scores a random sample of test patients and runs simple statistical tests.

    python -m xaicdss.experiments.rq1_rq3 --n 2000
"""

import argparse

import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config, pipeline
from ..data import TARGET


def _score_chunk(kind, rows):
    system = pipeline.load_system()
    engine = pipeline.engine_for(system, kind)
    X = system["space"].encode(system["test"].iloc[rows])
    out = []
    for r, x in zip(rows, X):
        a = engine.assess(x)
        out.append({"row": r, "p": a.probability, "agreement": a.agreement, "overlap": a.overlap,
                    "stability": a.stability, "reliability": a.score, "ood": a.ood_percentile,
                    "p_low": a.probability_range[0], "p_high": a.probability_range[1]})
    return out


def score_patients(kind, n, seed=config.SEED, jobs=4):
    test = pipeline.load_system()["test"]
    rows = np.random.default_rng(seed).choice(len(test), size=min(n, len(test)), replace=False)
    chunks = np.array_split(rows, jobs * 4)
    res = Parallel(n_jobs=jobs)(delayed(_score_chunk)(kind, c) for c in chunks)
    df = pd.DataFrame([r for chunk in res for r in chunk])
    df["y"] = test[TARGET].to_numpy()[df["row"]]
    return df


def analyse(df, kind):
    # Decision threshold: flag the top 20% highest-risk patients, a common
    # way to pick who gets follow-up when only 11% are readmitted.
    thr = np.quantile(df["p"], 0.80)
    df["flagged"] = (df["p"] >= thr).astype(int)
    df["wrong"] = (df["flagged"] != df["y"]).astype(int)
    df["sq_error"] = (df["y"] - df["p"]) ** 2

    lines = [f"# RQ1 and RQ3 results, {kind.upper()} model ({len(df):,} test patients)", ""]

    # RQ1
    d = df["reliability"]
    lines += [
        "## RQ1: do the methods disagree?", "",
        "| Measure | Mean | 10th pct | Median | 90th pct |", "|---|---|---|---|---|",
    ]
    for c in ["agreement", "overlap", "stability", "reliability"]:
        v = df[c]
        lines.append(f"| {c} | {v.mean():.3f} | {v.quantile(.1):.3f} | {v.median():.3f} | {v.quantile(.9):.3f} |")
    lines += [
        "",
        f"Top-{config.TOP_K} features named by every method identically: "
        f"{(df['overlap'] == 1).mean():.1%} of patients. "
        f"Average top-{config.TOP_K} overlap (Jaccard): {df['overlap'].mean():.2f}.",
        "",
    ]

    # RQ3
    w, r = df[df["wrong"] == 1]["reliability"], df[df["wrong"] == 0]["reliability"]
    mw = mannwhitneyu(w, r)
    auc = roc_auc_score(df["wrong"], -df["reliability"])
    rho_err = spearmanr(df["reliability"], df["sq_error"])
    rho_ood = spearmanr(df["reliability"], df["ood"])
    q = pd.qcut(df["reliability"], 5, labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"])
    by_q = df.groupby(q, observed=True).agg(n=("y", "size"), error_rate=("wrong", "mean"),
                                            mean_risk=("p", "mean"), unusual=("ood", "mean"))
    lines += [
        "## RQ3: is low reliability a warning sign?", "",
        f"Wrong = patient flagged (top 20% risk, p >= {thr:.3f}) but not readmitted, or not flagged but readmitted.", "",
        f"- Mean reliability when wrong: {w.mean():.3f}; when right: {r.mean():.3f} "
        f"(Mann-Whitney p = {mw.pvalue:.2g})",
        f"- AUROC of low reliability for spotting wrong predictions: {auc:.3f} (0.5 = no signal)",
        f"- Spearman correlation, reliability vs squared error: {rho_err.statistic:+.3f} (p = {rho_err.pvalue:.2g})",
        f"- Spearman correlation, reliability vs how unusual the patient is: {rho_ood.statistic:+.3f} "
        f"(p = {rho_ood.pvalue:.2g})",
        "",
        "| Reliability fifth | Patients | Error rate | Mean risk | Unusualness (pct) |", "|---|---|---|---|---|",
    ]
    for name, row in by_q.iterrows():
        lines.append(f"| {name} | {int(row.n)} | {row.error_rate:.1%} | {row.mean_risk:.3f} | {row.unusual:.0f} |")
    lines += [
        "",
        "Note: error rate depends strongly on predicted risk (patients near the threshold are the ones "
        "that get misclassified), so compare the mean-risk column before reading the error column as an "
        "effect of reliability alone.",
    ]

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(df["reliability"], bins=40, color="#3b6ea5")
    ax[0].set(title="Reliability score across patients", xlabel="reliability", ylabel="patients")
    ax[1].bar(range(5), by_q["error_rate"], color="#c0504d")
    ax[1].set_xticks(range(5), ["Q1\nlow", "Q2", "Q3", "Q4", "Q5\nhigh"])
    ax[1].set(title="Error rate by reliability fifth", ylabel="error rate")
    fig.tight_layout()
    fig.savefig(config.RESULTS / f"rq3_{kind}.png", dpi=130)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--model", default="xgb", choices=["xgb", "ebm"])
    ap.add_argument("--reuse", action="store_true", help="re-analyse saved scores without rescoring")
    args = ap.parse_args()
    scores = config.RESULTS / f"patient_scores_{args.model}.csv"
    if args.reuse:
        df = pd.read_csv(scores)
    else:
        df = score_patients(args.model, args.n)
        df.to_csv(scores, index=False)
    report = analyse(df, args.model)
    (config.RESULTS / f"rq1_rq3_{args.model}.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
