"""Clinician-facing demo interface.

    streamlit run app/streamlit_app.py

Research prototype for demonstration only. Not for use with real patients.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xaicdss import config, counterfactual, pipeline  # noqa: E402
from xaicdss.data import pretty  # noqa: E402

st.set_page_config(page_title="XAI-CDSS readmission risk", layout="wide")


@st.cache_resource
def load():
    return pipeline.load_system()


@st.cache_resource
def engine(kind):
    return pipeline.engine_for(load(), kind)


BAND_COLOUR = {"High": "green", "Moderate": "orange", "Low": "red"}


def show_value(v):
    return f"{v:g}" if isinstance(v, (int, float, np.floating)) else str(v)


def log_feedback(entry):
    config.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.AUDIT_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


def read_log():
    if not config.AUDIT_LOG.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in config.AUDIT_LOG.read_text().splitlines() if l.strip()])


system = load()
space, test = system["space"], system["test"]

st.sidebar.title("XAI-CDSS")
st.sidebar.warning("Research prototype. Not for clinical use.")
kind = st.sidebar.radio("Model", ["xgb", "ebm"],
                        format_func=lambda k: "XGBoost (complex)" if k == "xgb" else "EBM (readable)")
start = int(st.query_params.get("patient", 3923))  # 3923: a 68% risk patient, like the report's example
row = st.sidebar.number_input("Test patient #", min_value=0, max_value=len(test) - 1, value=start, step=1)
if st.sidebar.button("Random patient"):
    st.session_state["row"] = int(np.random.randint(len(test)))
row = st.session_state.pop("row", row)

page = st.sidebar.radio("Page", ["Patient", "Evidence", "Feedback log"])

if page == "Patient":
    patient = test.iloc[[row]]
    x = space.encode(patient)[0]
    eng = engine(kind)
    with st.spinner("Explaining..."):
        a = eng.assess(x)

    st.title("30-day readmission risk")
    st.caption(f"Test patient #{row} · encounter {patient['encounter_id'].iloc[0]} · "
               f"actual outcome (hidden from clinicians in real use): "
               f"{'readmitted' if patient['readmitted_30d'].iloc[0] else 'not readmitted'}")

    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted risk", f"{a.probability:.0%}",
              help="Calibrated probability of readmission within 30 days. Average patient: 11%.")
    c1.caption(f"Retrained models range: {a.probability_range[0]:.0%} to {a.probability_range[1]:.0%}")
    c2.metric("Explanation reliability", f"{a.score:.2f}", help="0 to 1. How consistent the explanation is.")
    c2.markdown(f"**:{BAND_COLOUR[a.band]}[{a.band}]**")
    c3.metric("How unusual is this patient", f"{a.ood_percentile:.0f} / 100",
              help="Percentile compared with training patients. Above 95 means very unusual.")

    if a.band == "Low":
        st.error("The explanation methods disagree for this patient. Treat the reasons below with caution "
                 "and rely more on your own clinical judgement.")
    if a.ood_percentile > 95:
        st.warning("This patient is unlike most patients the model was trained on.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Main factors")
        top = np.argsort(-np.abs(a.consensus))[:8]
        dfc = pd.DataFrame({
            "factor": [f"{pretty(space.features[j])} = {show_value(patient[space.features[j]].iloc[0])}" for j in top],
            "effect": a.consensus[top],
        })
        dfc["direction"] = np.where(dfc["effect"] > 0, "raises risk", "lowers risk")
        chart = alt.Chart(dfc).mark_bar().encode(
            x=alt.X("effect:Q", title="average effect across methods (scaled)"),
            y=alt.Y("factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=400)),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raises risk", "lowers risk"],
                                                           range=["#c0504d", "#3b6ea5"])),
        )
        st.altair_chart(chart, use_container_width=True)
        if a.sign_conflicts:
            st.info("Methods disagree on the direction of: " + ", ".join(pretty(f) for f in a.sign_conflicts))

        with st.expander("What each method says (top 5 per method)"):
            cols = st.columns(len(a.attributions))
            for col, (name, attr) in zip(cols, a.attributions.items()):
                col.markdown(f"**{name}**")
                for j in np.argsort(-np.abs(attr))[:5]:
                    col.write(f"{'▲' if attr[j] > 0 else '▼'} {pretty(space.features[j])}")

    with right:
        st.subheader("Why this reliability score")
        st.progress(a.agreement, text=f"Agreement between methods: {a.agreement:.2f}")
        st.progress(a.overlap, text=f"Same top-{config.TOP_K} factors: {a.overlap:.2f}")
        st.progress(a.stability, text=f"Stable when retrained: {a.stability:.2f}")

        st.subheader("What if?")
        steps = counterfactual.search(eng.model, x)
        if steps:
            st.write(f"Starting risk {a.probability:.0%}. The model's risk would change like this:")
            for s in steps:
                st.write(f"- {pretty(s['feature'])}: **{show_value(s['from'])} → {show_value(s['to'])}** "
                         f"→ risk {s['risk_after']:.0%}")
            st.caption("Shows what the model is sensitive to. Not a treatment recommendation.")
        else:
            st.write("No realistic change to the adjustable factors lowers the predicted risk.")

    with st.expander("Readable model: how one factor changes risk (EBM graph)"):
        ebm = system["models"]["ebm"].estimator
        feat = st.selectbox("Factor", space.features, format_func=pretty)
        g = ebm.explain_global().data(list(ebm.feature_names_in_).index(feat))
        names = list(g["names"])
        scores = list(g["scores"])
        if len(names) == len(scores) + 1:  # numeric bins: use left edges
            names = names[:-1]
        st.bar_chart(pd.DataFrame({"effect on log-odds": scores}, index=[show_value(n) for n in names]))

    st.subheader("Do you agree?")
    with st.form("feedback", clear_on_submit=True):
        verdict = st.radio("Your view", ["Agree with the risk", "Disagree: risk is too high",
                                         "Disagree: risk is too low"], horizontal=True)
        reason = st.selectbox("Main reason", ["", "Clinical picture differs", "Explanation does not make sense",
                                              "Missing information in record", "Patient circumstances", "Other"])
        note = st.text_area("Note (optional)")
        who = st.text_input("Your name or role")
        if st.form_submit_button("Save to audit log"):
            log_feedback({
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "clinician": who, "test_row": int(row), "encounter_id": int(patient["encounter_id"].iloc[0]),
                "model": kind, "risk": round(a.probability, 4), "reliability": round(a.score, 3),
                "band": a.band, "verdict": verdict, "reason": reason, "note": note,
            })
            st.success("Saved.")

elif page == "Evidence":
    st.title("Evidence behind the system")
    for name in ["cost_of_readability.md", "rq1_rq3_xgb.md", "rq1_rq3_ebm.md", "rq2_synthetic.md", "data_split.md"]:
        p = config.RESULTS / name
        if p.exists():
            with st.expander(name.replace(".md", "").replace("_", " "), expanded=name.startswith("cost")):
                st.markdown(p.read_text())
                png = p.with_suffix(".png")
                alt_png = config.RESULTS / name.replace("rq1_rq3", "rq3").replace(".md", ".png")
                for img in {png, alt_png}:
                    if img.exists():
                        st.image(str(img))
else:
    st.title("Clinician feedback log")
    log = read_log()
    if log.empty:
        st.write("No feedback recorded yet.")
    else:
        st.dataframe(log, use_container_width=True)
