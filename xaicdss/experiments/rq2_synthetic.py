"""RQ2: why do methods disagree? A controlled test with artificial data
where we know the true answer.

We make 8 features. The outcome truly depends on x0, x2, x3 and x4 only.
x1 has NO real effect, but we make it more and more similar (correlated) to
x0. A perfect explainer should always give x1 zero credit. We measure, as
the correlation grows:

- accuracy:  how well each method's attribution matches the true one
             (Spearman correlation with beta_j * (x_j - mean))
- fake credit: share of attribution wrongly given to x1
- agreement: how much the methods agree with each other

    python -m xaicdss.experiments.rq2_synthetic
"""

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .. import config, models
from ..data import TARGET, FeatureSpace
from ..explainers import explainers_for
from ..reliability import agreement_score

BETA = np.array([1.0, 0.0, 0.8, -0.6, 0.5, 0.0, 0.0, 0.0])
FEATS = [f"x{i}" for i in range(len(BETA))]
RHOS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]


def make_data(rho, n, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(BETA)))
    X[:, 1] = rho * X[:, 0] + np.sqrt(1 - rho ** 2) * X[:, 1]
    logit = X @ BETA - 1.0
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(X, columns=FEATS)
    df[TARGET] = y
    return df


def run(n_train=8000, n_explain=150, seed=config.SEED):
    space = FeatureSpace(numeric=list(FEATS), categorical=[], levels={})
    rows = []
    for rho in RHOS:
        train = make_data(rho, n_train, seed)
        test = make_data(rho, n_explain, seed + 1)
        Xc = space.encode(test)
        truth = BETA * (Xc - space.encode(train).mean(0))
        for kind in ("xgb", "ebm"):
            m = models.train(kind, train, space)
            exps = explainers_for(m, space.encode(train))
            attrs = {e.name: np.vstack([e.explain(m, x) for x in Xc]) for e in exps}
            for name, A in attrs.items():
                acc = np.nanmean([spearmanr(a, t).statistic for a, t in zip(A, truth)])
                fake = np.mean(np.abs(A[:, 1]) / (np.abs(A).sum(1) + 1e-12))
                rows.append({"rho": rho, "model": kind, "method": name, "accuracy": acc, "fake_credit": fake})
            agr = np.mean([agreement_score({k: v[i] for k, v in attrs.items()}) for i in range(len(Xc))])
            rows.append({"rho": rho, "model": kind, "method": "(agreement between methods)", "agreement": agr})
        print(f"rho={rho} done", flush=True)
    return pd.DataFrame(rows)


def report(df):
    res = df[df["method"] != "(agreement between methods)"]
    agr = df[df["method"] == "(agreement between methods)"]
    lines = [
        "# RQ2: explanations when features become similar (synthetic data)", "",
        "Truth: outcome depends on x0, x2, x3, x4. x1 has no effect but is correlated with x0 at level rho.", "",
        "## Share of credit wrongly given to x1 (lower is better)", "",
        res.pivot_table(index="rho", columns=["model", "method"], values="fake_credit").round(3).to_markdown(), "",
        "## Accuracy: Spearman with true attribution (higher is better)", "",
        res.pivot_table(index="rho", columns=["model", "method"], values="accuracy").round(3).to_markdown(), "",
        "## Agreement between methods (0 to 1)", "",
        agr.pivot_table(index="rho", columns="model", values="agreement").round(3).to_markdown(),
    ]
    return "\n".join(lines)


def plot(df):
    res = df[df["method"] != "(agreement between methods)"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, kind in zip(axes, ("xgb", "ebm")):
        for name, g in res[res["model"] == kind].groupby("method"):
            ax.plot(g["rho"], g["fake_credit"], marker="o", label=name)
        ax.set(title=f"{kind.upper()}: credit given to the fake feature", xlabel="correlation with real feature (rho)")
        ax.legend()
    axes[0].set_ylabel("share of attribution")
    fig.tight_layout()
    fig.savefig(config.RESULTS / "rq2_synthetic.png", dpi=130)


def main():
    config.ensure_dirs()
    df = run()
    df.to_csv(config.RESULTS / "rq2_synthetic.csv", index=False)
    plot(df)
    text = report(df)
    (config.RESULTS / "rq2_synthetic.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
