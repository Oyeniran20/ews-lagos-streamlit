import streamlit as st
from src import theme
from src.model import get_pipeline

st.set_page_config(
    page_title="Lagos Early Warning Framework",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject_css()

with st.sidebar:
    st.markdown("### 🌦️ Early Warning Framework")
    st.caption("MSc Thesis Demo · Lagos State, Nigeria")
    st.markdown("---")
    st.markdown(
        "**Koleoso Busola Khadijat**  \nTU2022000025 · MSc Computing  \n"
        "Anchor University Lagos  \nSupervisor: Dr D. Aleburu"
    )
    st.markdown("---")
    st.caption(
        "Use the pages in this sidebar to try the live prediction demo, "
        "explore the thesis's official Chapter 4 results, or read about "
        "the methodology."
    )

theme.hero(
    "Ensemble Predictive Analytics · MSc Thesis",
    "Early Warning Framework for Extreme Weather in Lagos State",
    "A soft-voting ensemble of Logistic Regression, Random Forest, and XGBoost "
    "that predicts tomorrow's extreme heat and flash-flood risk, one day ahead, "
    "from today's weather, land cover, and socioeconomic data.",
)

st.markdown("### What this app is")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """<div class="ews-card"><h4>🔴 Live Prediction Demo</h4>
        <p>A real ensemble, retrained on real Lagos climate data right in this session,
        predicting next-day heat and flood risk for any covered LGA — either replaying
        real historical days or exploring your own custom weather scenario.</p></div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """<div class="ews-card"><h4>📊 Results Dashboard</h4>
        <p>The thesis's official Chapter 4 evaluation — AUC-ROC, confusion matrices,
        feature importance, and the risk-tier breakdown — quoted directly from the
        submitted thesis, computed on the full study dataset.</p></div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """<div class="ews-card"><h4>📖 Methodology</h4>
        <p>How the framework is built: data sources, feature engineering, the
        soft-voting ensemble formula, evaluation design, and the three-tier
        risk communication system.</p></div>""",
        unsafe_allow_html=True,
    )

st.markdown("")
st.markdown("### The soft-voting ensemble")
st.latex(r"P_{ensemble}(x) = \frac{P_{LR}(x) + P_{RF}(x) + P_{XGB}(x)}{3}")
theme.note(
    "Three structurally different base learners — a linear model, a bagged-tree "
    "ensemble, and a boosted-tree ensemble — are trained independently, then combined "
    "by simply averaging their predicted probabilities. No meta-learner, no hidden "
    "weighting: the combination rule is fully transparent and auditable, which is "
    "part of why it was chosen over stacking for this thesis."
)

st.markdown("### A note on honesty in this app")
theme.note(
    "This app trains its <b>Live Prediction Demo</b> on the sample dataset shipped in "
    "this repository (8 of Lagos's 20 LGAs, 1990&ndash;1999). Its own metrics, shown on that "
    "page, are real and computed live &mdash; but they are not the thesis's official "
    "results. The <b>Results Dashboard</b> page instead quotes the thesis's official "
    "Chapter 4 evaluation verbatim, run on the complete engineered dataset. The two are "
    "kept clearly separate throughout this app; nothing here is invented."
)

with st.spinner("Warming up the demo model…"):
    pipe = get_pipeline("data/lagos_features.csv")

st.markdown("### At a glance")
k1, k2, k3, k4 = st.columns(4)
k1.metric("LGAs in demo dataset", len(pipe["lgas"]))
k2.metric("Daily records", f"{len(pipe['raw_df']):,}")
k3.metric("Demo heat model AUC", f"{pipe['targets']['Y_heat']['metrics']['Ensemble']['0.5']['roc_auc']:.3f}")
k4.metric("Official thesis heat AUC", "0.987")

st.caption(
    "Built from the real thesis: \"Design and Development of an Early Warning "
    "Framework for Extreme Weather Conditions Using Ensemble Predictive Analytics "
    "in Lagos State, Nigeria.\""
)
