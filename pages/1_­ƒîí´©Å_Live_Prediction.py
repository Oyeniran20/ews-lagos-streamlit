import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import theme
from src.model import get_pipeline, predict_row, build_scenario_row, SCENARIO_FIELDS
from src.theme import TIER_COLOR

st.set_page_config(page_title="Live Prediction · EWS Lagos", page_icon="🌡️", layout="wide")
theme.inject_css()

theme.hero(
    "Live Prediction Demo",
    "Try the Ensemble, One Day Ahead",
    "A real soft-voting ensemble, trained live in this session on real Lagos climate "
    "data, predicting tomorrow's extreme heat and flash-flood risk.",
)

theme.note(
    "This demo model is trained on the sample dataset shipped with this repository "
    "(8 LGAs, 1990&ndash;1999) &mdash; not the full dataset behind the thesis's official "
    "Chapter&nbsp;4 results (see the Results Dashboard page for those). Treat this page as "
    "a genuine, working illustration of the method, not a restatement of the thesis's "
    "reported accuracy."
)

pipe = get_pipeline("data/lagos_features.csv")
heat_res = pipe["targets"]["Y_heat"]
flood_res = pipe["targets"]["Y_flood"]

mode = st.radio(
    "Mode",
    ["Historical Replay (real recorded day)", "Custom Scenario (what-if)"],
    horizontal=True,
)

lga = st.selectbox("Local Government Area", pipe["lgas"], index=pipe["lgas"].index("Badagry") if "Badagry" in pipe["lgas"] else 0)

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
st.markdown("### Tomorrow's predicted risk")


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
