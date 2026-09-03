import streamlit as st
from src import theme

st.set_page_config(page_title="About · EWS Lagos", page_icon="📖", layout="wide")
theme.inject_css()

theme.hero(
    "Chapters 1–3 · Background & Methodology",
    "How This Framework Was Built",
    "The problem, the aim, the data, and the modelling pipeline behind the thesis — "
    "condensed from Chapters 1 through 3.",
)

st.markdown("### The problem")
theme.note(
    "Lagos State is Africa's largest urban agglomeration and one of the continent's most "
    "climatically exposed coastal cities. Rapid, often unplanned urbanisation has replaced "
    "absorptive wetland with impervious built-up land, raising both flood and urban-heat-island "
    "risk — while early warning coverage stays largely manual and reactive. NiMET collects weather "
    "data but has limited capacity for detailed, Lagos-specific forecasts; NIHSA's Annual Flood "
    "Outlook is seasonal, not day-to-day; and no current approach combines machine learning, ERA5 "
    "climate history, and socioeconomic data together."
)
s1, s2, s3 = st.columns(3)
s1.markdown(theme.stat_block("~20M", "People in the Lagos metropolitan area"), unsafe_allow_html=True)
s2.markdown(theme.stat_block("1.3M", "People affected by Nigeria's 2024 floods (320+ lives lost)"), unsafe_allow_html=True)
s3.markdown(theme.stat_block("1 in 3", "People worldwide lack access to local extreme-weather alerts"), unsafe_allow_html=True)

st.markdown("### Aim and objectives")
st.markdown(
    "**Aim:** to design, develop, and validate an integrated ensemble predictive analytics "
    "framework for an early warning system targeting extreme heat events and flood exceedance "
    "in Lagos State, Nigeria."
)
objs = [
    ("1. Understand", "Existing early warning and forecasting systems in Lagos, including NiMET, NIHSA, LASEMA, and NEMA, and the gaps that limit them."),
    ("2. Design", "An integrated framework: the data pipeline, feature engineering architecture, and multi-model ensemble structure."),
    ("3. Implement", "Logistic Regression, Random Forest, and XGBoost classifiers, combined into a soft-voting ensemble on ERA5 and socioeconomic data."),
    ("4. Evaluate", "Model performance against naive baselines, and translate outputs into a simple, colour-coded early warning alert system."),
]
o1, o2, o3, o4 = st.columns(4)
for col, (title, desc) in zip([o1, o2, o3, o4], objs):
    col.markdown(f'<div class="ews-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

st.markdown("### Study area & data sources")
c1, c2 = st.columns([1, 1.4])
with c1:
    st.markdown(
        """<div class="ews-card"><h4>🗺️ Lagos State</h4>
        <p>3,577 km² land area (~22% water bodies) · 20 LGAs, 37 LCDAs · ~15–16M state population,
        20M+ metro · ~30% of Nigeria's GDP.</p>
        <p style="margin-top:0.5rem;">Climate: 26–28°C mean annual temperature, 1,500–2,000 mm mean
        annual rainfall, 75–85% mean relative humidity. Humid tropical (Köppen Aw): a major wet
        season (Apr–Jul), minor wet season (Sep–Oct), harmattan-influenced dry season (Dec–Feb).</p>
        </div>""",
        unsafe_allow_html=True,
    )
with c2:
    sources = [
        ("🌍", "ERA5 Reanalysis (ECMWF/Copernicus)", "~31 km / 0.25° grid, hourly, 1990–2024, via the CDS API — temperature, dewpoint, precipitation, wind, pressure, 3-layer soil moisture."),
        ("👥", "WorldPop", "100 m population density grid (2020), Random Forest dasymetric disaggregation."),
        ("🏅", "NBS Multidimensional Poverty Index (2022)", "LGA-level MPI across 6 deprivation dimensions."),
        ("🟩", "ESA WorldCover / WUDAPT", "10 m impervious surface fraction and Local Climate Zone classification."),
    ]
    for icon, title, desc in sources:
        st.markdown(
            f'<div style="display:flex;gap:0.7rem;background:{theme.CARD_BG};border-radius:10px;'
            f'padding:0.6rem 0.9rem;margin-bottom:0.5rem;"><div style="font-size:1.3rem;">{icon}</div>'
            f'<div><b style="color:{theme.NAVY};">{title}</b><br>'
            f'<span style="color:{theme.TEXT_MUTED};font-size:0.85rem;">{desc}</span></div></div>',
            unsafe_allow_html=True,
        )

st.markdown("### Feature engineering")
groups = [
    ("🌡️ Thermal", "Heat Index, T2m 1/3/7-day lags, rolling max, monthly anomaly"),
    ("💧 Hydrological", "Precip 1/3/7-day lags, rolling 3/7/30-day sums, Antecedent Precipitation Index"),
    ("💨 Atmospheric", "Derived wind speed & direction, lagged surface pressure"),
    ("📅 Seasonal & Spatial", "Day-of-year cycles, wet-season flag, population, MPI, impervious fraction, LCZ"),
]
g1, g2, g3, g4 = st.columns(4)
for col, (title, desc) in zip([g1, g2, g3, g4], groups):
    col.markdown(f'<div class="ews-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

st.markdown("")
n1, n2 = st.columns(2)
with n1:
    theme.note(
        "<b>Extreme-day labelling rule.</b> Whole-record 95th percentile of daily max temperature "
        "(heat) and daily total precipitation (flood), computed separately per grid point &rarr; "
        "exactly 5% of days labelled extreme, by construction."
    )
with n2:
    theme.note(
        "<b>Redundancy screening.</b> Pearson/Spearman |r| &gt; 0.90 pairs pruned; permutation "
        "importance below the 5th percentile dropped &rarr; a parsimonious, theoretically motivated "
        "predictor set."
    )

st.markdown("### Model architecture: three base learners, combined by soft voting")
m1, m2, m3 = st.columns(3)
models = [
    ("Logistic Regression", "Fast, stable linear baseline; resistant to overfitting on a moderate-sized dataset."),
    ("Random Forest", "Breiman (2001): bootstrap-aggregated decision trees with random feature subsets at each split."),
    ("XGBoost", "Chen & Guestrin (2016): sequential boosted trees fitted to correct prior residual error."),
]
for col, (title, desc) in zip([m1, m2, m3], models):
    col.markdown(f'<div class="ews-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

st.latex(r"P_{ensemble}(x) = \frac{P_{LR}(x) + P_{RF}(x) + P_{XGB}(x)}{3}")
st.caption(
    "No separate meta-learner: each base learner trains once on the full training partition; only "
    "their test-set probabilities are averaged. An LSTM (TensorFlow/Keras) was planned as a fourth "
    "base learner but excluded — TensorFlow did not yet support the study's Python version at the "
    "time of writing."
)

st.markdown("### Evaluation framework")
e1, e2, e3 = st.columns(3)
with e1:
    st.markdown(
        '<div class="ews-card"><h4>5 metrics</h4><p>Accuracy, Precision, Recall, F1-score, AUC-ROC — '
        'chosen because extreme days are rare (~5%), so accuracy alone could hide a lazy model.</p></div>',
        unsafe_allow_html=True,
    )
with e2:
    st.markdown(
        '<div class="ews-card"><h4>2 naive baselines</h4><p><b>Persistence:</b> assume tomorrow looks '
        'like today. <b>Climatology:</b> always guess the long-run extreme-day rate. Any useful model '
        'must clearly beat both.</p></div>',
        unsafe_allow_html=True,
    )
with e3:
    st.markdown(
        '<div class="ews-card"><h4>2 decision thresholds</h4><p>0.50 (standard) and 0.30 (sensitive). '
        'A warning system favours catching real events over avoiding false alarms.</p></div>',
        unsafe_allow_html=True,
    )
theme.note(
    "<b>Temporally honest evaluation.</b> All performance figures come from a single, chronologically "
    "held-out test set: the most recent 20% of the date range, never used for fitting or tuning. "
    "Feature importance is read from Random Forest's built-in Gini-based scores for both hazards; a "
    "SHAP-based analysis is identified as a more rigorous next step."
)

st.markdown("### From probability to a public alert")
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
    "Alerts aggregate to LGA level by a precautionary max rule: an LGA receives the highest alert "
    "found at any grid point inside it."
)

st.markdown("#### Four dissemination channels")
d1, d2, d3, d4 = st.columns(4)
chans = [
    ("🛰️", "Institutional API", "NiMET, NIHSA, LASEMA, NEMA dashboards"),
    ("📻", "Mass media", "LTV & FRCN broadcast, English and Yoruba"),
    ("💬", "SMS networks", "MTN, Airtel, Glo — plain-language alerts"),
    ("🧑‍🤝‍🧑", "Community liaison", "Ward health workers, last-mile reach"),
]
for col, (icon, title, desc) in zip([d1, d2, d3, d4], chans):
    col.markdown(
        f'<div class="ews-card"><h4>{icon} {title}</h4><p>{desc}</p></div>', unsafe_allow_html=True
    )

st.markdown("### Theoretical grounding")
th1, th2, th3 = st.columns(3)
theories = [
    ("Sendai Framework for DRR 2015–2030", "Frames EWS as sociotechnical systems, not just forecasts. This study operationalises Sendai Pillars 1, 2, and 4."),
    ("Data-Driven Decision Support & CRISP-DM", "Simon's bounded rationality and the KDD framework justify algorithmic decision support; the pipeline maps onto all six CRISP-DM phases."),
    ("Urban Vulnerability Theory", "Blaikie et al.'s Pressure and Release model: Risk = Hazard × Vulnerability — why technical accuracy alone cannot protect Lagos's most exposed communities."),
]
for col, (title, desc) in zip([th1, th2, th3], theories):
    col.markdown(f'<div class="ews-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Source: \"Design and Development of an Early Warning Framework for Extreme Weather Conditions "
    "Using Ensemble Predictive Analytics in Lagos State, Nigeria\" — Koleoso Busola Khadijat "
    "(TU2022000025), MSc Computing, Anchor University Lagos, Supervisor: Dr D. Aleburu."
)
