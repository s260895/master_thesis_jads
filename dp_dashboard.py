"""Interactive privacy-utility dashboard for the DP fraud-detection experiments.

Reads results/dp_sweep_results.csv (generate it with `python -m src.experiment`)
and lets you explore how model utility and membership-inference attack success
move as the differential-privacy budget epsilon changes.

Run:  streamlit run dp_dashboard.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RESULTS = Path(__file__).resolve().parent / "results" / "dp_sweep_results.csv"

st.set_page_config(page_title="DP Privacy-Utility Dashboard", layout="wide")
st.title("Differential Privacy vs Membership-Inference Attacks")
st.caption(
    "Fraud-detection classifiers under Laplace-mechanism noise. "
    "Companion dashboard to the JADS MSc thesis experiments."
)

if not RESULTS.exists():
    st.error(
        "No results found. Generate them first:\n\n"
        "```\npython -m src.experiment\n```"
    )
    st.stop()


@st.cache_data
def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS)
    df["epsilon"] = df["epsilon"].astype(float)
    return df


df = load_results()
finite_eps = sorted(df.loc[np.isfinite(df["epsilon"]), "epsilon"].unique())
classifiers = sorted(df["classifier"].unique())

UTILITY_METRICS = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
ATTACK_METRICS = ["salem_attack_auc", "yeom_attack_auc"]

with st.sidebar:
    st.header("Controls")
    eps_choice = st.select_slider(
        "Privacy budget ε (lower = more private)",
        options=finite_eps + ["clean (no DP)"],
        value=finite_eps[len(finite_eps) // 2],
    )
    selected_clfs = st.multiselect("Classifiers", classifiers, default=classifiers)
    utility_metric = st.selectbox("Utility metric", UTILITY_METRICS, index=0)
    attack_metric = st.selectbox("Attack metric", ATTACK_METRICS, index=0)

view = df[df["classifier"].isin(selected_clfs)].copy()
eps_value = np.inf if eps_choice == "clean (no DP)" else float(eps_choice)
at_eps = view[view["epsilon"] == eps_value]

# ---- headline tiles at the chosen epsilon ----------------------------------
st.subheader(
    "At ε = ∞ (clean baseline)" if np.isinf(eps_value) else f"At ε = {eps_value:g}"
)
cols = st.columns(3)
best_util = at_eps.loc[at_eps[utility_metric].idxmax()] if len(at_eps) else None
worst_leak = at_eps.loc[at_eps[attack_metric].idxmax()] if len(at_eps) else None
if best_util is not None:
    cols[0].metric(
        f"Best {utility_metric}",
        f"{best_util[utility_metric]:.3f}",
        best_util["classifier"],
    )
    cols[1].metric(
        f"Highest {attack_metric.replace('_', ' ')}",
        f"{worst_leak[attack_metric]:.3f}",
        worst_leak["classifier"],
    )
    cols[2].metric(
        "Leakage margin over random (0.5)",
        f"+{max(worst_leak[attack_metric] - 0.5, 0):.3f}",
    )

st.dataframe(
    at_eps.set_index("classifier")[UTILITY_METRICS + ATTACK_METRICS].round(3),
    use_container_width=True,
)

# ---- curves over epsilon ----------------------------------------------------
finite_view = view[np.isfinite(view["epsilon"])]
left, right = st.columns(2)

with left:
    st.subheader(f"Utility: {utility_metric} vs ε")
    fig = px.line(
        finite_view,
        x="epsilon",
        y=utility_metric,
        color="classifier",
        log_x=True,
        markers=True,
    )
    if not np.isinf(eps_value):
        fig.add_vline(x=eps_value, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader(f"Privacy leakage: {attack_metric} vs ε")
    fig = px.line(
        finite_view,
        x="epsilon",
        y=attack_metric,
        color="classifier",
        log_x=True,
        markers=True,
    )
    fig.add_hline(
        y=0.5, line_dash="dot", line_color="green", annotation_text="random guess"
    )
    if not np.isinf(eps_value):
        fig.add_vline(x=eps_value, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

# ---- privacy-utility pareto -------------------------------------------------
st.subheader("Privacy-utility tradeoff (each point = one classifier at one ε)")
pareto = view.copy()
pareto["eps_label"] = pareto["epsilon"].map(
    lambda e: "clean" if np.isinf(e) else f"{e:g}"
)
fig = go.Figure()
for clf in selected_clfs:
    sub = pareto[pareto["classifier"] == clf].sort_values("epsilon")
    fig.add_trace(
        go.Scatter(
            x=sub[attack_metric],
            y=sub[utility_metric],
            mode="lines+markers+text",
            text=sub["eps_label"],
            textposition="top center",
            name=clf,
        )
    )
fig.update_layout(
    xaxis_title=f"{attack_metric} (lower = more private)",
    yaxis_title=f"{utility_metric} (higher = more useful)",
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Attack AUC of 0.5 means the attacker cannot tell training members from "
    "non-members; 1.0 means full membership leakage. The sweet spot is top-left: "
    "high utility, low leakage."
)
