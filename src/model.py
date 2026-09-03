"""
Core modelling module for the Lagos Early Warning Framework demo app.

This reproduces the real methodology described in the thesis (Chapter 3):
  - Labels are shifted forward one day, so the model only ever predicts
    TOMORROW's extreme heat / flood risk using data known up to and
    including TODAY. Nothing about tomorrow's actual weather is used as
    an input.
  - Three base learners (Logistic Regression, Random Forest, XGBoost) are
    trained independently, then combined with a SOFT-VOTING ensemble:
    the final risk score is the plain average of the three predicted
    probabilities:  P_ensemble(x) = (P_LR(x) + P_RF(x) + P_XGB(x)) / 3
  - The train/test split is TIME-based (earliest years train, most recent
    years test) so the model is never evaluated on data "from the past
    relative to its own training set" - this mirrors real early-warning
    deployment.

IMPORTANT - HONESTY NOTE (read this before trusting any number from here):
This app trains on the sample dataset shipped in this repository
(data/lagos_features.csv - 8 of Lagos's 20 LGAs, 1990-1999, ~29k daily
grid-point records), which is the subset of the thesis's full data
pipeline that was available to package with this demo. The metrics this
module reports describe THIS demo model, trained live in your browser
session on that sample - they are NOT the thesis's official Chapter 4
results (which were computed on the full engineered dataset covering the
complete study period and are quoted verbatim, from the thesis text, on
the "Results Dashboard" page). The two are kept clearly separate
throughout this app. Nothing here is invented - every number is either
computed live from real data, or quoted directly from the thesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False
    from sklearn.ensemble import GradientBoostingClassifier

CAT_COLS = ["dominant_lcz", "lga", "senatorial_district"]
NON_PREDICTIVE_COLS = ["date", "lga_match_type"]
TARGETS = ["Y_heat", "Y_flood"]
TARGET_LABEL = {"Y_heat": "Extreme Heat", "Y_flood": "Flash Flood"}
TUNED_THRESHOLD = 0.3
RANDOM_STATE = 42

# The handful of "everyday" fields exposed as sliders in the Custom
# Scenario explorer, with sensible min/max bounds for Lagos.
SCENARIO_FIELDS = {
    "temperature_2m_max_celsius": ("Max temperature today (°C)", 24.0, 42.0, 0.1),
    "relative_humidity_pct": ("Relative humidity today (%)", 30.0, 100.0, 1.0),
    "total_precipitation_mm": ("Rainfall today (mm)", 0.0, 150.0, 1.0),
    "wind_speed_10m_ms": ("Wind speed today (m/s)", 0.0, 15.0, 0.1),
    "soil_water_layer1_m3m3": ("Topsoil moisture today (m³/m³)", 0.0, 0.6, 0.01),
}


def build_tomorrow_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Shift Y_heat/Y_flood forward one day, per grid point, so the model
    is trained to predict TOMORROW using only what is known TODAY."""
    df = df.sort_values(["latitude", "longitude", "date"]).reset_index(drop=True)
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()
    for target in TARGETS:
        vals = df[target].to_numpy(dtype=float)
        shifted = np.full(len(vals), np.nan)
        for i in range(len(vals) - 1):
            if lat[i] == lat[i + 1] and lon[i] == lon[i + 1]:
                shifted[i] = vals[i + 1]
        df[f"{target}_tomorrow"] = shifted
    return df


def get_cat_categories(df: pd.DataFrame) -> dict:
    return {c: sorted(df[c].dropna().unique().tolist()) for c in CAT_COLS if c in df.columns}


def raw_feature_columns(df: pd.DataFrame) -> list:
    tomorrow_cols = [f"{t}_tomorrow" for t in TARGETS]
    exclude = set(NON_PREDICTIVE_COLS) | set(TARGETS) | set(tomorrow_cols)
    return [c for c in df.columns if c not in exclude]

# Prefer if available
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def _encode(df_raw: pd.DataFrame, feat_cols: list, cat_categories: dict) -> pd.DataFrame:
    """Apply consistent one-hot encoding using a fixed category list, so
    a single live inference row lines up with the columns the models
    were trained on."""
    X = df_raw[feat_cols].copy()
    cat_present = [c for c in CAT_COLS if c in X.columns]
    for c in cat_present:
        X[c] = pd.Categorical(X[c], categories=cat_categories.get(c, sorted(X[c].dropna().unique())))
    X = pd.get_dummies(X, columns=cat_present, drop_first=True)
    return X


def _naive_baselines(working: pd.DataFrame, target_col: str, test_mask: np.ndarray) -> dict:
    """Persistence (tomorrow = today) and Climatology (tomorrow = long-run
    historical rate) baselines, computed on the held-out test slice."""
    y_true = working.loc[test_mask, f"{target_col}_tomorrow"].astype(int).to_numpy()
    y_today = working.loc[test_mask, target_col].astype(int).to_numpy()
    out = {}
    if len(set(y_true)) > 1:
        out["Persistence"] = {"roc_auc": roc_auc_score(y_true, y_today)}
    train_rate = working.loc[~test_mask, target_col].mean()
    clim_pred = np.full(len(y_true), train_rate)
    if len(set(y_true)) > 1:
        out["Climatology"] = {"roc_auc": roc_auc_score(y_true, clim_pred)}
    return out


def _metrics_row(y_true, y_pred, y_proba) -> dict:
    row = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    row["roc_auc"] = roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan")
    return row


def train_one_target(df: pd.DataFrame, target_col: str, cat_categories: dict) -> dict:
    working = df.dropna(subset=[f"{target_col}_tomorrow"]).copy()
    feat_cols = raw_feature_columns(working)
    X_all = _encode(working, feat_cols, cat_categories)
    valid = X_all.notna().all(axis=1).to_numpy()
    working = working.loc[valid].reset_index(drop=True)
    X_all = X_all.loc[valid].reset_index(drop=True)
    y_all = working[f"{target_col}_tomorrow"].astype(int).reset_index(drop=True)

    dates = working["date"]
    unique_dates = np.sort(dates.unique())
    cutoff_idx = int(len(unique_dates) * 0.8)
    cutoff_idx = min(max(cutoff_idx, 1), len(unique_dates) - 1)
    cutoff_date = pd.Timestamp(unique_dates[cutoff_idx])
    train_mask = (dates < cutoff_date).to_numpy()
    test_mask = ~train_mask

    X_train, X_test = X_all.loc[train_mask], X_all.loc[test_mask]
    y_train, y_test = y_all.loc[train_mask], y_all.loc[test_mask]
    feature_columns = list(X_all.columns)

    # --- Logistic Regression (scaled) ---
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    lr.fit(X_train_s, y_train)
    proba_lr = lr.predict_proba(X_test_s)[:, 1]

    # --- Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]

    # --- XGBoost (or Gradient Boosting fallback) ---
    n_pos = max(int(y_train.sum()), 1)
    n_neg = max(int(len(y_train) - y_train.sum()), 1)
    if HAS_XGBOOST:
        xgb_model = XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            scale_pos_weight=n_neg / n_pos,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
        )
        xgb_model.fit(X_train, y_train)
        proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    else:  # pragma: no cover
        xgb_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        xgb_model.fit(X_train, y_train)
        proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

    proba_ensemble = (proba_lr + proba_rf + proba_xgb) / 3.0

    metrics = {}
    for name, proba in [("Logistic Regression", proba_lr), ("Random Forest", proba_rf),
                         ("XGBoost", proba_xgb), ("Ensemble", proba_ensemble)]:
        pred_05 = (proba >= 0.5).astype(int)
        pred_tuned = (proba >= TUNED_THRESHOLD).astype(int)
        metrics[name] = {
            "0.5": _metrics_row(y_test, pred_05, proba),
            str(TUNED_THRESHOLD): _metrics_row(y_test, pred_tuned, proba),
        }
    metrics.update(_naive_baselines(working, target_col, test_mask))

    cm = confusion_matrix(y_test, (proba_ensemble >= TUNED_THRESHOLD).astype(int)).tolist()

    importances = pd.Series(rf.feature_importances_, index=feature_columns)
    importances = importances.sort_values(ascending=False).head(12)

    test_df = working.loc[test_mask, ["date", "lga", "latitude", "longitude", target_col]].copy()
    test_df = test_df.reset_index(drop=True)
    test_df["actual_tomorrow"] = y_test.to_numpy()
    test_df["predicted_risk"] = proba_ensemble
    test_df["predicted_tier"] = pd.cut(
        test_df["predicted_risk"], bins=[-0.01, 0.3, 0.5, 1.01],
        labels=["Low", "Medium", "High"],
    )

    return {
        "models": {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb_model},
        "scaler": scaler,
        "feature_columns": feature_columns,
        "raw_feature_cols": feat_cols,
        "metrics": metrics,
        "confusion_matrix": cm,
        "feature_importance": importances,
        "test_df": test_df,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "cutoff_date": cutoff_date,
        "base_rate": float(working[target_col].mean()),
    }


@st.cache_resource(show_spinner="Training the soft-voting ensemble on real Lagos climate data (one-time, ~15s)…")
def get_pipeline(csv_path: str = "data/lagos_features.csv") -> dict:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = build_tomorrow_labels(df)
    cat_categories = get_cat_categories(df)
    targets = {t: train_one_target(df, t, cat_categories) for t in TARGETS}
    lgas = sorted(df["lga"].dropna().unique().tolist())
    date_min, date_max = df["date"].min(), df["date"].max()

    feat_cols = raw_feature_columns(df)
    numeric_cols = [c for c in feat_cols if c not in CAT_COLS]
    lga_medians = df.groupby("lga")[numeric_cols].median()
    lga_modes = df.groupby("lga")[[c for c in CAT_COLS if c != "lga"]].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else None
    )

    return {
        "targets": targets,
        "cat_categories": cat_categories,
        "raw_df": df,
        "lgas": lgas,
        "date_min": date_min,
        "date_max": date_max,
        "lga_medians": lga_medians,
        "lga_modes": lga_modes,
        "year_fixed": int(df["year"].max()),
    }


def heat_index_celsius(temp_c: float, rh_pct: float) -> float:
    """NOAA/Rothfusz heat-index regression (published formula, not
    invented here). Input Celsius/percent, output Celsius."""
    t_f = temp_c * 9 / 5 + 32
    rh = rh_pct
    hi_f = (
        -42.379 + 2.04901523 * t_f + 10.14333127 * rh - 0.22475541 * t_f * rh
        - 0.00683783 * t_f ** 2 - 0.05481717 * rh ** 2
        + 0.00122874 * t_f ** 2 * rh + 0.00085282 * t_f * rh ** 2
        - 0.00000199 * t_f ** 2 * rh ** 2
    )
    return (hi_f - 32) * 5 / 9


def build_scenario_row(pipe: dict, lga: str, month: int, sliders: dict) -> dict:
    """Build a full raw feature row for Custom Scenario mode: start from
    the chosen LGA's real historical median for every field, then
    override the handful of fields the user actually controls. Lag and
    rolling temperature/precipitation columns are set equal to today's
    entered value (a disclosed simplifying assumption: 'assume the last
    few days looked similar to today'), rather than fabricated
    independently."""
    medians = pipe["lga_medians"].loc[lga].to_dict()
    modes = pipe["lga_modes"].loc[lga].to_dict()
    row = dict(medians)
    row.update(modes)
    row["lga"] = lga

    row["temperature_2m_max_celsius"] = sliders["temperature_2m_max_celsius"]
    row["temperature_2m_min_celsius"] = sliders["temperature_2m_max_celsius"] - 6.0
    row["temperature_2m_mean_celsius"] = sliders["temperature_2m_max_celsius"] - 3.0
    row["relative_humidity_pct"] = sliders["relative_humidity_pct"]
    row["total_precipitation_mm"] = sliders["total_precipitation_mm"]
    row["wind_speed_10m_ms"] = sliders["wind_speed_10m_ms"]
    row["soil_water_layer1_m3m3"] = sliders["soil_water_layer1_m3m3"]
    row["soil_water_layer2_m3m3"] = sliders["soil_water_layer1_m3m3"]

    row["heat_index_celsius"] = heat_index_celsius(
        sliders["temperature_2m_max_celsius"], sliders["relative_humidity_pct"]
    )

    for lag_col in ["temp_max_lag1", "temp_max_lag2", "temp_max_lag3",
                    "temp_max_roll3", "temp_max_roll7", "temp_max_roll14"]:
        row[lag_col] = sliders["temperature_2m_max_celsius"]
    for lag_col in ["precip_lag1", "precip_lag2", "precip_lag3",
                    "precip_roll3", "precip_roll7", "precip_roll14"]:
        row[lag_col] = sliders["total_precipitation_mm"]
    row["antecedent_precip_index"] = sliders["total_precipitation_mm"]

    row["month"] = month
    row["day"] = 15
    row["day_of_year"] = int(round((month - 1) * 30.44 + 15))
    row["month_sin"] = float(np.sin(2 * np.pi * month / 12))
    row["month_cos"] = float(np.cos(2 * np.pi * month / 12))
    row["is_rainy_season"] = 1 if month in (4, 5, 6, 7, 8, 9, 10) else 0
    row["year"] = pipe["year_fixed"]

    return row


def predict_row(target_result: dict, cat_categories: dict, row: dict) -> dict:
    """Run all three base learners + soft-voting ensemble on a single
    real (or scenario) feature row (a dict of raw column -> value)."""
    feat_cols = target_result["raw_feature_cols"]
    row_full = {c: row.get(c, np.nan) for c in feat_cols}
    df_row = pd.DataFrame([row_full])
    X = _encode(df_row, feat_cols, cat_categories)
    X = X.reindex(columns=target_result["feature_columns"], fill_value=0)

    lr = target_result["models"]["Logistic Regression"]
    rf = target_result["models"]["Random Forest"]
    xgb_model = target_result["models"]["XGBoost"]
    scaler = target_result["scaler"]

    p_lr = float(lr.predict_proba(scaler.transform(X))[:, 1][0])
    p_rf = float(rf.predict_proba(X)[:, 1][0])
    p_xgb = float(xgb_model.predict_proba(X)[:, 1][0])
    p_ensemble = (p_lr + p_rf + p_xgb) / 3.0

    # Thesis's risk-tier design (Chapter 3.9): Low < 0.30, Medium 0.30-0.50, High >= 0.50
    if p_ensemble >= 0.5:
        tier = "High"
    elif p_ensemble >= 0.3:
        tier = "Medium"
    else:
        tier = "Low"

    return {
        "Logistic Regression": p_lr,
        "Random Forest": p_rf,
        "XGBoost": p_xgb,
        "Ensemble": p_ensemble,
        "tier": tier,
    }
