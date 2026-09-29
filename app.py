import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme
from src.model import get_pipeline, predict_row, build_scenario_row, SCENARIO_FIELDS
from src.theme import TIER_COLOR

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
        "This page has the live prediction demo and methodology. Use the sidebar "
        "to jump to the Results Dashboard for the thesis's official Chapter 4 evaluation."
    )

theme.hero(
    "Ensemble Predictive Analytics · MSc Thesis",
    "Early Warning Framework for Extreme Weather in Lagos State",
    "A soft-voting ensemble of Logistic Regression, Random Forest, and XGBoost "
    "that predicts tomorrow's extreme heat and flash-flood risk, one day ahead, "
    "from today's weather, land cover, and socioeconomic data.",
)

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
    "The <b>live prediction demo</b> below trains on the sample dataset shipped in "
    "this repository (8 of Lagos's 20 LGAs, 1990&ndash;1999). Its own metrics, shown "
    "further down, are real and computed live &mdash; but they are not the thesis's "
    "official results. The <b>Results Dashboard</b> page (see sidebar) instead quotes "
    "the thesis's official Chapter 4 evaluation verbatim, run on the complete "
    "engineered dataset. The two are kept clearly separate throughout this app; "
    "nothing here is invented."
)

with st.spinner("Warming up the demo model…"):
    pipe = get_pipeline("data/lagos_features.csv")
heat_res = pipe["targets"]["Y_heat"]
flood_res = pipe["targets"]["Y_flood"]

st.markdown("### At a glance")
k1, k2, k3, k4 = st.columns(4)
k1.metric("LGAs in demo dataset", len(pipe["lgas"]))
k2.metric("Daily records", f"{len(pipe['raw_df']):,}")
k3.metric("Demo heat model AUC", f"{pipe['targets']['Y_heat']['metrics']['Ensemble']['0.5']['roc_auc']:.3f}")
k4.metric("Official thesis heat AUC", "0.987")

st.markdown("---")

# ======================================================================
# Live Prediction Demo (folded in from the former separate page)
# ======================================================================
st.markdown("## 🌡️ Try the Ensemble, One Day Ahead")
st.caption(
    "A real soft-voting ensemble, trained live in this session on real Lagos "
    "climate data, predicting tomorrow's extreme heat and flash-flood risk."
)

mode = st.radio(
    "Mode",
    ["Historical Replay (real recorded day)", "Custom Scenario (what-if)"],
    horizontal=True,
)

lga = st.selectbox(
    "Local Government Area",
    pipe["lgas"],
    index=pipe["lgas"].index("Badagry") if "Badagry" in pipe["lgas"] else 0,
)

st.markdown("")

if mode.startswith("Historical"):
    df_lga = pipe["raw_df"][pipe["raw_df"]["lga"] == lga].dropna(
        subset=["Y_heat_tomorrow", "Y_flood_tomorrow"]
    ).sort_values("date").reset_index(drop=True)

    dates = df_lga["date"].dt.date.tolist()
    picked = st.select_slider("Pick a real recorded day", options=dates, value=dates[len(dates) // 2])
    row_series = df_lga[df_lga["date"].dt.date == picked].iloc[0]
    row = row_series.to_dict()

    st.markdown("#### Today's real recorded conditions")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Max temp", f"{row['temperature_2m_max_celsius']:.1f}°C")
    c2.metric("Humidity", f"{row['relative_humidity_pct']:.0f}%")
    c3.metric("Rainfall", f"{row['total_precipitation_mm']:.1f} mm")
    c4.metric("Wind speed", f"{row['wind_speed_10m_ms']:.1f} m/s")
    c5.metric("Heat index", f"{row['heat_index_celsius']:.1f}°C")

    heat_out = predict_row(heat_res, pipe["cat_categories"], row)
    flood_out = predict_row(flood_res, pipe["cat_categories"], row)
    actual_heat = int(row_series["Y_heat_tomorrow"])
    actual_flood = int(row_series["Y_flood_tomorrow"])
else:
    st.markdown("#### Adjust tomorrow's weather scenario")
    month = st.select_slider(
        "Month",
        options=list(range(1, 13)),
        value=7,
        format_func=lambda m: pd.Timestamp(2024, m, 1).strftime("%B"),
    )
    lga_med = pipe["lga_medians"].loc[lga]
    sliders = {}
    for (field, (label, lo, hi, step)), col in zip(SCENARIO_FIELDS.items(), st.columns(len(SCENARIO_FIELDS))):
        default = float(np.clip(lga_med.get(field, (lo + hi) / 2), lo, hi))
        sliders[field] = col.slider(label, min_value=lo, max_value=hi, value=round(default, 2), step=step)

    theme.note(
        "Fields not shown here (surface pressure, dewpoint, wind direction, population, "
        "land cover, poverty…) are held at this LGA's real historical median. Multi-day "
        "lag and rolling-average features are set equal to today's entered value, a "
        "disclosed simplifying assumption for this what-if explorer."
    )

    row = build_scenario_row(pipe, lga, month, sliders)
    heat_out = predict_row(heat_res, pipe["cat_categories"], row)
    flood_out = predict_row(flood_res, pipe["cat_categories"], row)
    actual_heat = actual_flood = None

st.markdown("---")
st.markdown("### 🚨 Tomorrow's predicted risk")


def gauge(value: float, tier: str, title: str) -> go.Figure:
    color = TIER_COLOR[tier]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        number={"suffix": "%", "font": {"size": 34, "color": theme.NAVY}},
        title={"text": title, "font": {"size": 16, "color": theme.NAVY}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": theme.TEXT_MUTED},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": theme.CARD_BG,
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#E4F0EA"},
                {"range": [30, 50], "color": "#FCEBD6"},
                {"range": [50, 100], "color": "#FBE3E4"},
            ],
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


g1, g2 = st.columns(2)
with g1:
    st.plotly_chart(gauge(heat_out["Ensemble"], heat_out["tier"], "Extreme Heat Risk"), use_container_width=True)
    st.markdown(theme.tier_pill(heat_out["tier"]), unsafe_allow_html=True)
    if actual_heat is not None:
        st.caption(f"What actually happened: {'🔴 Extreme heat day' if actual_heat else '🟢 Not an extreme heat day'}")
with g2:
    st.plotly_chart(gauge(flood_out["Ensemble"], flood_out["tier"], "Flash Flood Risk"), use_container_width=True)
    st.markdown(theme.tier_pill(flood_out["tier"]), unsafe_allow_html=True)
    if actual_flood is not None:
        st.caption(f"What actually happened: {'🔴 Flood-exceedance day' if actual_flood else '🟢 Not a flood-exceedance day'}")

st.markdown("### How the three base learners voted")
b1, b2 = st.columns(2)
for out, col, label in [(heat_out, b1, "Heat"), (flood_out, b2, "Flood")]:
    fig = go.Figure(go.Bar(
        x=["Logistic Regression", "Random Forest", "XGBoost", "Ensemble"],
        y=[out["Logistic Regression"], out["Random Forest"], out["XGBoost"], out["Ensemble"]],
        marker_color=[theme.TEAL, theme.TEAL, theme.TEAL, theme.NAVY],
        text=[f"{v:.0%}" for v in [out["Logistic Regression"], out["Random Forest"], out["XGBoost"], out["Ensemble"]]],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"{label} risk probability by model", yaxis_range=[0, 1], yaxis_tickformat=".0%",
        height=320, **theme.CHART_TEMPLATE,
    )
    col.plotly_chart(fig, use_container_width=True)

theme.note(
    "The Ensemble bar is the plain average of the other three — soft voting, exactly as "
    "described in Chapter 3 of the thesis: P<sub>ensemble</sub>(x) = "
    "(P<sub>LR</sub>(x) + P<sub>RF</sub>(x) + P<sub>XGB</sub>(x)) / 3."
)

st.markdown("---")

# ======================================================================
# Condensed methodology (folded in from the former separate About page)
# ======================================================================
st.markdown("## 📖 How This Framework Was Built")
st.caption("Condensed from Chapters 1–3 of the thesis. See the Results Dashboard page for the full Chapter 4 evaluation.")

theme.note(
    "Lagos State is Africa's largest urban agglomeration and one of the continent's "
    "most climatically exposed coastal cities. Rapid urbanisation has replaced "
    "absorptive wetland with impervious built-up land, raising both flood and "
    "urban-heat-island risk, while early-warning coverage stays largely manual and "
    "reactive. This framework combines machine learning, ERA5 climate history, and "
    "socioeconomic data to close that gap."
)

m1, m2, m3, m4 = st.columns(4)
groups = [
    ("🌍 Data sources", "ERA5 climate reanalysis (1990–2024), WorldPop population, NBS poverty (MPI), ESA WorldCover / WUDAPT land cover."),
    ("🧮 Feature engineering", "Heat Index, Antecedent Precipitation Index, 1/3/7-day lags, rolling windows, wind speed/direction, seasonal cycles."),
    ("🤖 Model architecture", "Logistic Regression, Random Forest, and XGBoost trained independently, combined by soft voting (see formula above)."),
    ("📏 Evaluation", "5 metrics (incl. AUC-ROC), 2 naive baselines (Persistence, Climatology), a chronological 80/20 train/test split."),
]
for col, (title, desc) in zip([m1, m2, m3, m4], groups):
    col.markdown(f'<div class="ews-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

st.markdown("")
st.markdown("#### From probability to a public alert")
t1, t2, t3 = st.columns(3)
tiers = [
    ("Low", "< 0.30", theme.GREEN, "Business as usual; routine monitoring continues"),
    ("Medium", "0.30 – 0.50", theme.AMBER, "Public advisories, community preparedness, LASEMA pre-positioning"),
    ("High", "≥ 0.50", theme.CORAL, "Full emergency response: evacuation planning, cooling centres, NEMA coordination"),
]
for col, (name, rng, color, desc) in zip([t1, t2, t3], tiers):
    col.markdown(
        f'<div style="background:{color};border-radius:14px;padding:1.1rem 1.2rem;color:white;height:100%;">'
        f'<div style="font-size:1.3rem;font-weight:800;font-family:Cambria,serif;">{name}</div>'
        f'<div style="font-weight:700;margin-bottom:0.5rem;">{rng}</div>'
        f'<div style="font-size:0.88rem;">{desc}</div></div>',
        unsafe_allow_html=True,
    )
st.caption(
    "This is the same tuned 0.30 / 0.50 thresholding used throughout this app: the "
    "gauges above and the Results Dashboard's confusion matrices both use it. Alerts "
    "aggregate to LGA level by a precautionary max rule: an LGA receives the highest "
    "alert found at any grid point inside it."
)

st.markdown("---")
st.caption(
    "Source: \"Design and Development of an Early Warning Framework for Extreme Weather "
    "Conditions Using Ensemble Predictive Analytics in Lagos State, Nigeria\" — Koleoso "
    "Busola Khadijat (TU2022000025), MSc Computing, Anchor University Lagos, "
    "Supervisor: Dr D. Aleburu."
)
