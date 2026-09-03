"""
Official results from Chapter 4 of the thesis, quoted verbatim (not
recomputed here). These describe the full evaluation reported in the
submitted thesis: the complete engineered dataset, 2018-2024 test window,
tuned alert threshold of 0.30. They are intentionally kept separate from
whatever this demo app's own live-retrained model reports on the smaller
sample dataset shipped in data/lagos_features.csv - see src/model.py's
module docstring for why the two differ.

Source: "Design and Development of an Early Warning Framework for Extreme
Weather Conditions Using Ensemble Predictive Analytics in Lagos State,
Nigeria", Chapter 4 (Results), and the accompanying thesis defense deck.
"""

HEAT = {
    "auc_labels": ["Climatology", "Persistence", "XGBoost", "Random Forest", "Ensemble", "Logistic Regression"],
    "auc_values": [0.500, 0.864, 0.986, 0.986, 0.986, 0.987],
    "auc_note": "All three base learners agree closely (AUC 0.986-0.987); the ensemble matches them and clearly beats both naive baselines.",
    "confusion": {"tn": 32044, "fp": 2660, "fn": 67, "tp": 3569},
    "accuracy": "92.9%", "precision": "57.3%", "recall": "98.2%", "f1": "72.4%",
    "metric_note": "9.5% of test days were extreme, above the 5% design rate, a direct sign of the warming trend in the 2018-2024 test window.",
    "skill_gain": "+0.122",
    "skill_gain_note": "heat AUC-ROC gained over the best baseline (Persistence)",
}

FLOOD = {
    "auc_labels": ["Climatology", "Persistence", "Random Forest", "Logistic Regression", "XGBoost", "Ensemble"],
    "auc_values": [0.500, 0.585, 0.830, 0.835, 0.842, 0.846],
    "auc_note": "Persistence is weak here (AUC 0.585): unlike heat, today's rainfall says little about tomorrow's. The ensemble still wins, by a smaller margin.",
    "confusion": {"tn": 25356, "fp": 10750, "fn": 399, "tp": 1835},
    "accuracy": "70.9%", "precision": "14.6%", "recall": "82.1%", "f1": "24.8%",
    "metric_note": "The sensitive threshold trades a high false-alarm rate for a low miss rate, the right trade-off for life safety, but a real limitation.",
    "skill_gain": "+0.261",
    "skill_gain_note": "flood AUC-ROC gained over the best baseline (Persistence)",
}

FEATURE_IMPORTANCE = {
    "heat": {
        "features": [
            "temp_max_roll14 / roll7 / roll3",
            "1, 2, and 3-day temperature lags",
            "Raw daily maximum temperature",
            "Heat Index (T2m + dewpoint)",
        ],
        "note": "Recent temperature history, thermal persistence, is the single strongest signal for tomorrow's heat risk.",
    },
    "flood": {
        "features": [
            "total_precipitation_mm",
            "Antecedent Precipitation Index",
            "Rolling 3, 7, and 14-day precipitation totals",
            "Soil moisture at 1-day lag",
        ],
        "note": "Ground that is already wet floods faster than dry ground given the same new rainfall.",
    },
    "shared_note": "No LGA-identity or land-cover feature appears in either top-15 list, a sign the models learn genuine weather persistence, not just location memorisation.",
}

RISK_TIERS = {
    "labels": ["Low", "Medium", "High"],
    "values": [41.4, 22.9, 35.6],
    "n_lga_days": 20448,
    "period_note": "2018-2024 test period, across the 8 covered LGAs",
}

HEAT_RISK_BY_LGA = [
    {"lga": "Badagry", "tier": "Highest", "detail": "~45% of test days"},
    {"lga": "Ifako-Ijaiye", "tier": "High", "detail": "Large share of High-risk days"},
    {"lga": "Amuwo-Odofin", "tier": "Moderate", "detail": "Close behind the leaders"},
    {"lga": "Lagos Island", "tier": "Moderate", "detail": "Close behind the leaders"},
    {"lga": "Eti-Osa", "tier": "Minimal", "detail": "Almost no High-risk days"},
    {"lga": "Ibeju-Lekki", "tier": "Minimal", "detail": "Almost no High-risk days"},
]

COVERAGE = {
    "covered": 8,
    "total": 20,
    "lgas": ["Amuwo-Odofin", "Badagry", "Epe", "Eti-Osa", "Ibeju-Lekki", "Ifako-Ijaiye", "Ikorodu", "Lagos Island"],
    "note": "The ERA5 download ran in two parts: a western box (complete for 1990-2024) and a planned eastern extension (not yet complete). Every official Chapter 4 result is built on the western-box data only.",
}

TREND = {
    "warming": "+1.3°C",
    "warming_note": "warming, 1990 to 2024",
    "warming_detail": "Average daily maximum temperature rises from ~28.5°C in the early 1990s to ~29.9°C in 2024, a clear, steady warming trend.",
    "rainfall_note": "No clear trend in daily rainfall. Wettest year: 2019 (~6.0 mm/day) | Driest year: 2024 (~3.5 mm/day). Rainfall is far noisier, year to year, than temperature.",
}

CORRELATIONS = [
    {"pair": "Impervious surface ↔ Population", "r": "+0.89", "note": "Built-up areas are, as expected, the most densely populated"},
    {"pair": "Impervious surface ↔ Poverty (MPI)", "r": "-0.14", "note": "Poorer LGAs are not the densest, most built-up ones"},
    {"pair": "Population ↔ Poverty (MPI)", "r": "-0.18", "note": "Poverty and density point in different directions in this data"},
    {"pair": "Temperature ↔ Impervious surface", "r": "+0.27", "note": "A smaller, expected urban-heat-island signal"},
]
