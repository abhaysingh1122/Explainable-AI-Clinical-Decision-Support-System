"""End-to-end pipeline: raw data -> split -> models -> calibration ->
reliability engine -> saved artifacts and the cost-of-readability table.

    python -m xaicdss.pipeline
"""

import time

import joblib
import numpy as np
import pandas as pd

from . import config, data, models
from .data import TARGET
from .explainers import explainers_for
from .reliability import OODDetector, ReliabilityEngine


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def versions():
    import interpret, lime, shap, sklearn, xgboost
    return {
        "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "interpret": interpret.__version__,
        "shap": shap.__version__, "lime": getattr(lime, "__version__", "0.2.x"),
    }


def run(n_stability=config.N_STABILITY_MODELS):
    config.ensure_dirs()
    log(f"Library versions: {versions()}")
    df = data.build_features(data.load_raw())
    splits = data.patient_split(df)
    (config.RESULTS / "data_split.md").write_text(data.leakage_report(df, splits))
    log("Split done, leakage check passed")

    space = data.FeatureSpace.fit(splits["train"])
    ycal = splits["calib"][TARGET].to_numpy()
    fitted = {}
    for kind in ("lr", "ebm", "xgb"):
        t = time.time()
        fitted[kind] = models.train(kind, splits["train"], space).calibrate(splits["calib"], ycal)
        log(f"Trained {kind} in {time.time() - t:.0f}s")

    stability = {}
    for kind in ("ebm", "xgb"):
        stability[kind] = []
        for i in range(n_stability):
            boot = models.resample_patients(splits["train"], seed=1000 + i)
            m = models.train(kind, boot, space, seed=1000 + i).calibrate(splits["calib"], ycal)
            stability[kind].append(m)
        log(f"Trained {n_stability} resampled {kind} models")

    ood = OODDetector(space, splits["train"])
    rng = np.random.default_rng(config.SEED)
    bg_rows = rng.choice(len(splits["train"]), 5000, replace=False)
    train_codes = space.encode(splits["train"].iloc[bg_rows])

    cost_of_readability(fitted, splits)
    fairness(fitted, splits)

    joblib.dump({
        "space": space, "models": fitted, "stability": stability, "ood": ood,
        "train_codes": train_codes, "test": splits["test"], "versions": versions(),
    }, config.ARTIFACTS / "system.joblib", compress=3)
    log(f"Saved {config.ARTIFACTS / 'system.joblib'}")


def cost_of_readability(fitted, splits):
    test = splits["test"]
    y = test[TARGET].to_numpy()
    preds = {k: m.predict_proba(test) for k, m in fitted.items()}
    rows = []
    for k, m in fitted.items():
        r = {"Model": m.name, **models.metrics(y, preds[k])}
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(config.RESULTS / "cost_of_readability.csv", index=False)
    cmp = models.bootstrap_compare(y, preds["xgb"], preds["ebm"], test["patient_nbr"].to_numpy())
    cmp_lr = models.bootstrap_compare(y, preds["ebm"], preds["lr"], test["patient_nbr"].to_numpy())

    lines = [
        "# Cost of readability (test set, patient-level split)",
        "",
        f"Test set: {len(test):,} encounters, {test['patient_nbr'].nunique():,} patients, "
        f"{y.mean():.2%} readmitted within 30 days. All models calibrated on a separate set of patients.",
        "",
        "| Model | AUROC | AUPRC | Brier | ECE |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['Model']} | {r['AUROC']:.4f} | {r['AUPRC']:.4f} | {r['Brier']:.4f} | {r['ECE']:.4f} |")
    lines += [
        "",
        f"Chance-level AUPRC equals the readmission rate ({y.mean():.3f}).",
        "",
        "## Paired patient-level bootstrap (1,000 resamples)",
        "",
        f"- XGBoost AUROC 95% CI: {cmp['a_ci'][0]:.4f} to {cmp['a_ci'][1]:.4f}",
        f"- EBM AUROC 95% CI: {cmp['b_ci'][0]:.4f} to {cmp['b_ci'][1]:.4f}",
        f"- **Cost of readability (XGBoost minus EBM): {cmp['diff']:+.4f} AUROC "
        f"(95% CI {cmp['diff_ci'][0]:+.4f} to {cmp['diff_ci'][1]:+.4f}, p = {cmp['p_two_sided']:.3f})**",
        f"- EBM minus logistic regression: {cmp_lr['diff']:+.4f} AUROC "
        f"(95% CI {cmp_lr['diff_ci'][0]:+.4f} to {cmp_lr['diff_ci'][1]:+.4f}, p = {cmp_lr['p_two_sided']:.3f})",
        "",
        "Rudin's claim [2] is that readability usually costs little accuracy. A difference whose "
        "confidence interval sits near zero supports it on this task.",
    ]
    (config.RESULTS / "cost_of_readability.md").write_text("\n".join(lines))
    log(f"Cost of readability: {cmp['diff']:+.4f} AUROC")


def fairness(fitted, splits):
    test = splits["test"]
    frames = []
    for k in ("ebm", "xgb"):
        t = models.subgroup_auroc(test, fitted[k].predict_proba(test))
        t.insert(0, "model", fitted[k].name)
        frames.append(t)
    pd.concat(frames).to_csv(config.RESULTS / "subgroup_performance.csv", index=False)


def load_system(path=config.ARTIFACTS / "system.joblib"):
    return joblib.load(path)


def engine_for(system, kind="xgb", n_stability=None):
    m = system["models"][kind]
    stab = system["stability"].get(kind, [])
    if n_stability is not None:
        stab = stab[:n_stability]
    return ReliabilityEngine(m, explainers_for(m, system["train_codes"]), stab, system["ood"])


if __name__ == "__main__":
    run()
