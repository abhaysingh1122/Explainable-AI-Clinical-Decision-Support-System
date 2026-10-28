"""One file to run the whole system from the terminal (no frontend needed).

    python run.py                      explain the two demo patients
    python run.py --patient 3923       explain one test patient
    python run.py --random             explain a random test patient
    python run.py --model ebm ...      use the readable EBM instead of XGBoost
    python run.py --train              rebuild data split and models
    python run.py --experiments        rerun RQ1, RQ2 and RQ3 experiments
    python run.py --all                train, run experiments, then explain demo patients
    python run.py --test               run the automatic tests

Research prototype for demonstration only. Not for use with real patients.
"""

import argparse
import subprocess
import sys
import warnings

import numpy as np

from xaicdss import config, counterfactual, pipeline
from xaicdss.data import pretty

warnings.filterwarnings("ignore")

DEMO_PATIENTS = [3923, 10613]
MODEL_NAMES = {"xgb": "XGBoost (complex)", "ebm": "EBM (readable)"}


def line(char="-"):
    print(char * 70)


def fmt(v):
    return f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)


def explain(system, kind, row):
    """Print the risk, reliability, main factors and what-if for one patient."""
    space, test = system["space"], system["test"]
    patient = test.iloc[[row]]
    x = space.encode(patient)[0]
    engine = pipeline.engine_for(system, kind)
    a = engine.assess(x)

    line("=")
    print(f"Patient #{row}  |  model: {MODEL_NAMES[kind]}")
    line("=")
    print(f"Predicted 30-day readmission risk : {a.probability:.0%}  "
          f"(retrained models: {a.probability_range[0]:.0%} to {a.probability_range[1]:.0%})")
    print(f"Explanation reliability          : {a.score:.2f}  [{a.band}]")
    print(f"   agreement between methods     : {a.agreement:.2f}")
    print(f"   same top-{config.TOP_K} factors            : {a.overlap:.2f}")
    print(f"   stable when retrained         : {a.stability:.2f}")
    print(f"How unusual is this patient      : {a.ood_percentile:.0f} / 100")
    if a.band == "Low":
        print("\n  WARNING: the explanation methods disagree for this patient.")
        print("  Treat the reasons below with caution.")
    if a.ood_percentile > 95:
        print("\n  WARNING: this patient is unlike most patients the model was trained on.")

    print("\nMain factors (average of all methods):")
    for j in np.argsort(-np.abs(a.consensus))[:6]:
        f = space.features[j]
        arrow = "raises risk" if a.consensus[j] > 0 else "lowers risk"
        print(f"  {'+' if a.consensus[j] > 0 else '-'} {pretty(f)} = {fmt(patient[f].iloc[0])}  ({arrow})")
    if a.sign_conflicts:
        print("  Methods disagree on the direction of: " + ", ".join(pretty(f) for f in a.sign_conflicts))

    print("\nTop 3 factors according to each method:")
    for name, attr in a.attributions.items():
        top = [pretty(space.features[j]) for j in np.argsort(-np.abs(attr))[:3]]
        print(f"  {name:<10}: {', '.join(top)}")

    print("\nWhat if? (what the model is sensitive to, not treatment advice)")
    steps = counterfactual.search(engine.model, x)
    if not steps:
        print("  No realistic change to the adjustable factors lowers the predicted risk.")
    for s in steps:
        print(f"  {pretty(s['feature'])}: {fmt(s['from'])} -> {fmt(s['to'])}  =>  risk {s['risk_after']:.0%}")

    actual = "readmitted" if patient["readmitted_30d"].iloc[0] else "not readmitted"
    print(f"\nActual outcome (for checking only): {actual}")
    print()


def run_module(module, *args):
    subprocess.run([sys.executable, "-m", module, *args], check=True)


def main():
    ap = argparse.ArgumentParser(description="Explainable AI clinical decision support system")
    ap.add_argument("--patient", type=int, help="test patient number to explain")
    ap.add_argument("--random", action="store_true", help="explain a random test patient")
    ap.add_argument("--model", choices=["xgb", "ebm"], default="xgb")
    ap.add_argument("--train", action="store_true", help="rebuild the data split and models")
    ap.add_argument("--experiments", action="store_true", help="rerun the RQ1-RQ3 experiments")
    ap.add_argument("--all", action="store_true", help="train, run experiments, explain demo patients")
    ap.add_argument("--test", action="store_true", help="run the automatic tests")
    args = ap.parse_args()

    if args.test:
        sys.exit(subprocess.run([sys.executable, "-m", "pytest", "-q"]).returncode)

    model_file = config.ARTIFACTS / "system.joblib"
    if args.train or args.all or not model_file.exists():
        print("Training models (about 6 minutes)...")
        pipeline.run()

    if args.experiments or args.all:
        print("Running experiments (about 25 minutes)...")
        run_module("xaicdss.experiments.rq2_synthetic")
        run_module("xaicdss.experiments.rq1_rq3", "--model", "xgb")
        run_module("xaicdss.experiments.rq1_rq3", "--model", "ebm")
        print(f"Results saved in {config.RESULTS}")

    if (args.train or args.experiments) and args.patient is None and not args.random:
        return

    print("Loading trained system...")
    try:
        system = pipeline.load_system()
    except Exception as e:  # usually a library version mismatch
        print(f"Could not load saved models ({e}). Rebuilding them...")
        pipeline.run()
        system = pipeline.load_system()

    if args.random:
        rows = [int(np.random.randint(len(system["test"])))]
    elif args.patient is not None:
        if not 0 <= args.patient < len(system["test"]):
            sys.exit(f"Patient number must be between 0 and {len(system['test']) - 1}.")
        rows = [args.patient]
    else:
        rows = DEMO_PATIENTS
    for row in rows:
        explain(system, args.model, row)


if __name__ == "__main__":
    main()
