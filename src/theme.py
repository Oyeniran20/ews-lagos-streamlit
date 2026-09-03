"""Shared visual design system for the app: palette, fonts, CSS injection,
and small reusable HTML component helpers. The palette mirrors the
thesis defense slide deck for a consistent look across both deliverables.
"""

import streamlit as st

NAVY = "#0B2340"
NAVY2 = "#0F2A47"
TEAL = "#1C7293"
CORAL = "#D64550"
AMBER = "#E8A33D"
GREEN = "#2E8B57"
TEXT_DARK = "#1B2430"
TEXT_MUTED = "#5B6B7C"
CARD_BG = "#F2F5F8"
CARD_BG2 = "#EAF0F6"
WHITE = "#FFFFFF"
LINE = "#D8E0E8"

TIER_COLOR = {"Low": GREEN, "Medium": AMBER, "High": CORAL}


def inject_css():
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Cambria:wght@400;700&family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
        <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {TEXT_DARK};
        }}
        h1, h2, h3 {{
            font-family: 'Cambria', Georgia, serif !important;
            color: {NAVY} !important;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1100px;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {NAVY} 0%, {NAVY2} 100%);
        }}
        section[data-testid="stSidebar"] * {{
            color: {WHITE} !important;
        }}
        section[data-testid="stSidebar"] .stMarkdown p {{
            color: #C6D3E0 !important;
        }}
        div[data-testid="stMetric"] {{
            background: {CARD_BG};
            border-radius: 14px;
            padding: 1rem 1.1rem;
            border: 1px solid {LINE};
        }}
        div[data-testid="stMetricValue"] {{
            color: {NAVY};
        }}
        .ews-hero {{
            background: linear-gradient(135deg, {NAVY} 0%, {NAVY2} 55%, {TEAL} 130%);
            border-radius: 20px;
            padding: 2.6rem 2.6rem;
            color: {WHITE};
            margin-bottom: 1.8rem;
        }}
        .ews-hero .kicker {{
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.78rem;
            font-weight: 700;
            color: {AMBER};
            margin-bottom: 0.6rem;
        }}
        .ews-hero h1 {{
            color: {WHITE} !important;
            font-size: 2.3rem;
            margin: 0 0 0.7rem 0;
            line-height: 1.2;
        }}
        .ews-hero p {{
            color: #D7E2EC;
            font-size: 1.02rem;
            max-width: 42rem;
            margin: 0;
        }}
        .ews-card {{
            background: {CARD_BG};
            border: 1px solid {LINE};
            border-radius: 16px;
            padding: 1.4rem 1.5rem;
            height: 100%;
        }}
        .ews-card h4 {{
            margin: 0 0 0.4rem 0;
            color: {NAVY};
            font-family: 'Cambria', Georgia, serif;
        }}
        .ews-card p {{
            color: {TEXT_MUTED};
            font-size: 0.92rem;
            margin: 0;
        }}
        .ews-pill {{
            display: inline-block;
            padding: 0.3rem 0.85rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.85rem;
            color: {WHITE};
        }}
        .ews-badge-row {{
            display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 0.8rem;
        }}
        .ews-note {{
            background: {CARD_BG2};
            border-left: 3px solid {TEAL};
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            font-size: 0.9rem;
            color: {TEXT_MUTED};
        }}
        .ews-stat {{
            text-align: center;
            padding: 1rem 0.5rem;
        }}
        .ews-stat .num {{
            font-size: 2.1rem;
            font-weight: 800;
            color: {TEAL};
            font-family: 'Cambria', Georgia, serif;
        }}
        .ews-stat .lbl {{
            color: {TEXT_MUTED};
            font-size: 0.85rem;
        }}
        footer {{visibility: hidden;}}
        #MainMenu {{visibility: hidden;}}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(kicker: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="ews-hero">
            <div class="kicker">{kicker}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tier_pill(tier: str) -> str:
    color = TIER_COLOR.get(tier, TEAL)
    return f'<span class="ews-pill" style="background:{color};">{tier} risk</span>'

def stat_block(number: str, label: str) -> str:
    return f'<div class="ews-stat"><div class="num">{number}</div><div class="lbl">{label}</div></div>'


def note(text: str):
    st.markdown(f'<div class="ews-note">{text}</div>', unsafe_allow_html=True)


CHART_TEMPLATE = dict(
    font=dict(family="Inter, sans-serif", color=TEXT_DARK),
    plot_bgcolor=WHITE,
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=50, b=10),
)
