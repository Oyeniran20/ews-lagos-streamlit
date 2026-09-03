import plotly.graph_objects as go
import streamlit as st

from src import theme
from src import thesis_results as R

st.set_page_config(page_title="Results Dashboard · EWS Lagos", page_icon="📊", layout="wide")
theme.inject_css()

theme.hero(
    "Chapter 4 · Official Results",
    "The Thesis's Reported Evaluation",
    "Every number on this page is quoted directly from the submitted thesis and its "
    "defense deck — computed on the complete engineered dataset, evaluated on the "
    "2018–2024 test window with a tuned 0.30 alert threshold.",
)

theme.note(
    f"<b>Coverage.</b> {R.COVERAGE['note']} This evaluation covers "
    f"{R.COVERAGE['covered']} of {R.COVERAGE['total']} Lagos LGAs: "
    + ", ".join(R.COVERAGE["lgas"]) + "."
)

tab_heat, tab_flood, tab_feat, tab_risk, tab_eda = st.tabs(
    ["🌡️ Extreme Heat", "🌊 Flash Flood", "🔍 Feature Importance", "🚦 Risk Communication", "📈 EDA & Trends"]
)


def hazard_tab(tab, cfg, label):
    with tab:
        c1, c2 = st.columns([1.1, 1])
        with c1:
            st.markdown(f"#### AUC-ROC comparison (test set, 2018–2024)")
            fig = go.Figure(go.Bar(
                y=cfg["auc_labels"], x=cfg["auc_values"], orientation="h",
                marker_color=[theme.TEXT_MUTED if v <= 0.6 else theme.TEAL for v in cfg["auc_values"]],
                text=[f"{v:.3f}" for v in cfg["auc_values"]], textposition="outside",
            ))
            fig.update_layout(xaxis_range=[0, 1.08], height=330, **theme.CHART_TEMPLATE)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(cfg["auc_note"])

            st.markdown(f"#### Ensemble skill vs. best naive baseline")
            g1, g2 = st.columns(2)
            g1.markdown(theme.stat_block(cfg["skill_gain"], cfg["skill_gain_note"]), unsafe_allow_html=True)

        with c2:
            st.markdown("#### Confusion matrix (tuned threshold = 0.30)")
            cm = cfg["confusion"]
            cells = [
                ("True Negatives", cm["tn"], theme.CARD_BG, theme.NAVY),
                ("False Positives", cm["fp"], "#FCEBD6", theme.AMBER),
                ("False Negatives", cm["fn"], "#FBE3E4", theme.CORAL),
                ("True Positives", cm["tp"], "#E4F0EA", theme.GREEN),
            ]
            cc1, cc2 = st.columns(2)
            for i, (lbl, val, bg, fg) in enumerate(cells):
                target = cc1 if i % 2 == 0 else cc2
                target.markdown(
                    f'<div style="background:{bg};border-radius:12px;padding:0.8rem 1rem;margin-bottom:0.6rem;">'
                    f'<div style="font-size:1.5rem;font-weight:800;color:{fg};font-family:Cambria,serif;">{val:,}</div>'
                    f'<div style="font-size:0.8rem;color:{theme.TEXT_MUTED};">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )
            metric_cells = [
                ("Accuracy", cfg["accuracy"]), ("Precision", cfg["precision"]),
                ("Recall", cfg["recall"]), ("F1-score", cfg["f1"]),
            ]
            mc1, mc2 = st.columns(2)
            for i, (lbl, val) in enumerate(metric_cells):
                target = mc1 if i % 2 == 0 else mc2
                target.markdown(
                    f'<div style="background:{theme.NAVY};border-radius:12px;padding:0.7rem 1rem;margin-bottom:0.6rem;">'
                    f'<div style="font-size:1.35rem;font-weight:800;color:{theme.TEAL};font-family:Cambria,serif;">{val}</div>'
                    f'<div style="font-size:0.78rem;color:#C7D6E8;">{lbl}</div></div>',
                    unsafe_allow_html=True,
                )
            st.caption(cfg["metric_note"])


hazard_tab(tab_heat, R.HEAT, "Heat")
hazard_tab(tab_flood, R.FLOOD, "Flood")

with tab_feat:
    st.markdown("#### What the models actually learned (Random Forest importances)")
    f1, f2 = st.columns(2)
    for col, key, title, color in [(f1, "heat", "🌡️ Heat Model", theme.CORAL), (f2, "flood", "💧 Flood Model", theme.TEAL)]:
        block = R.FEATURE_IMPORTANCE[key]
        items = "".join(
            f'<div style="display:flex;gap:0.6rem;align-items:center;background:{theme.WHITE};'
            f'border:1px solid {theme.LINE};border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.4rem;">'
            f'<span style="font-weight:800;color:{color};font-family:Cambria,serif;">{i+1}</span>'
            f'<span style="font-size:0.92rem;">{feat}</span></div>'
            for i, feat in enumerate(block["features"])
        )
        col.markdown(
            f'<div class="ews-card"><h4>{title}</h4>{items}'
            f'<p style="margin-top:0.6rem;font-style:italic;">{block["note"]}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown("")
    theme.note(R.FEATURE_IMPORTANCE["shared_note"])

with tab_risk:
    c1, c2 = st.columns([1, 1.3])
    with c1:
        st.markdown(f"#### {R.RISK_TIERS['n_lga_days']:,} LGA-days, classified")
        fig = go.Figure(go.Pie(
            labels=R.RISK_TIERS["labels"], values=R.RISK_TIERS["values"], hole=0.55,
            marker_colors=[theme.GREEN, theme.AMBER, theme.CORAL],
            textinfo="label+percent",
        ))
        fig.update_layout(height=360, showlegend=False, **theme.CHART_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(R.RISK_TIERS["period_note"])
    with c2:
        st.markdown("#### Heat risk by LGA (High-risk day share)")
        rank_color = {"Highest": theme.CORAL, "High": theme.CORAL, "Moderate": theme.AMBER, "Minimal": theme.GREEN}
        for r in R.HEAT_RISK_BY_LGA:
            color = rank_color.get(r["tier"], theme.TEAL)
            pill = (f'<span class="ews-pill" style="background:{color};">{r["tier"]} risk</span>')
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.8rem;background:{theme.CARD_BG};'
                f'border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.5rem;">'
                f'<div style="width:8rem;font-weight:700;color:{theme.NAVY};">{r["lga"]}</div>'
                f'{pill}'
                f'<div style="color:{theme.TEXT_MUTED};font-size:0.85rem;">{r["detail"]}</div></div>',
                unsafe_allow_html=True,
            )
        st.caption("Flood risk, by contrast, is much more even across all 8 covered LGAs.")

with tab_eda:
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("#### Correlation between key variables")
        for c in R.CORRELATIONS:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:1rem;background:{theme.CARD_BG};'
                f'border-radius:10px;padding:0.6rem 1rem;margin-bottom:0.5rem;">'
                f'<div style="flex:1;font-weight:600;">{c["pair"]}</div>'
                f'<div style="font-weight:800;color:{theme.TEAL};font-family:Cambria,serif;font-size:1.1rem;">{c["r"]}</div>'
                f'<div style="flex:1.3;color:{theme.TEXT_MUTED};font-size:0.82rem;">{c["note"]}</div></div>',
                unsafe_allow_html=True,
            )
    with c2:
        st.markdown("#### Long-term trend (1990–2024)")
        st.markdown(theme.stat_block(R.TREND["warming"], R.TREND["warming_note"]), unsafe_allow_html=True)
        st.caption(R.TREND["warming_detail"])
        st.markdown("---")
        st.caption(R.TREND["rainfall_note"])

    st.markdown("")
    st.markdown("#### Three independent data sources agree on where urban Lagos is")
    st.caption(
        "WorldPop population density, WUDAPT Local Climate Zones, and ESA WorldCover impervious-surface "
        "maps all point to the same dense urban core (Ikeja, Oshodi-Isolo, Mushin, Somolu, Ajeromi-Ifelodun), "
        "while the eastern LGAs (Epe, Ikorodu, Ibeju-Lekki) show the highest poverty (MPI) scores yet sit at "
        "the edge of, or outside, current weather-data coverage."
    )
