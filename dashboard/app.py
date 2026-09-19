import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ARTEFACTS_DIR, FEATURE_COLS, TARGET_COL, TEST_SEASON, VAL_SEASON

st.set_page_config(page_title="NBA Winner Predictor", layout="wide")

_BLUE  = "#4C72B0"
_GREY  = "#D0D0D0"
_RED   = "#C0392B"


# Data loading (cached)

@st.cache_resource
def load_model():
    return joblib.load(ARTEFACTS_DIR / "model.pkl")


@st.cache_data
def load_metrics():
    with open(ARTEFACTS_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_processed_df():
    df = pd.read_parquet(ARTEFACTS_DIR / "processed_df.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# Prediction helpers

def _get_team_as_home(df: pd.DataFrame, team: str) -> dict | None:
    rows = df[df["Home"] == team].sort_values("Date")
    if rows.empty:
        return None
    r = rows.iloc[-1]
    return {
        "avg_pts_scored":      r.get("home_avg_pts_scored"),
        "avg_pts_allowed":     r.get("home_avg_pts_allowed"),
        "avg_pts_last5":       r.get("home_avg_pts_last5"),
        "win_pct_last5":       r.get("home_win_pct_last5"),
        "win_pct_last10":      r.get("home_win_pct_last10"),
        "home_win_pct_last10": r.get("home_home_win_pct_last10"),
        "pt_diff_last10":      r.get("home_pt_diff_last10"),
        "SRS":    r.get("home_SRS"),
        "ORtg":   r.get("home_ORtg"),
        "DRtg":   r.get("home_DRtg"),
        "NRtg":   r.get("home_NRtg"),
        "Pace":   r.get("home_Pace"),
        "TS%":    r.get("home_TS%"),
        "eFG%":   r.get("home_eFG%"),
        "TOV%":   r.get("home_TOV%"),
        "ORB%":   r.get("home_ORB%"),
        "FTr":    r.get("home_FTr"),
        "3PAr":   r.get("home_3PAr"),
        "D_eFG%": r.get("home_D_eFG%"),
        "D_TOV%": r.get("home_D_TOV%"),
        "D_DRB%": r.get("home_D_DRB%"),
    }


def _get_team_as_visitor(df: pd.DataFrame, team: str) -> dict | None:
    rows = df[df["Visitor"] == team].sort_values("Date")
    if rows.empty:
        return None
    r = rows.iloc[-1]
    return {
        "avg_pts_scored":      r.get("visitor_avg_pts_scored"),
        "avg_pts_allowed":     r.get("visitor_avg_pts_allowed"),
        "avg_pts_last5":       r.get("visitor_avg_pts_last5"),
        "win_pct_last5":       r.get("visitor_win_pct_last5"),
        "win_pct_last10":      r.get("visitor_win_pct_last10"),
        "away_win_pct_last10": r.get("visitor_away_win_pct_last10"),
        "pt_diff_last10":      r.get("visitor_pt_diff_last10"),
        "SRS":    r.get("visitor_SRS"),
        "ORtg":   r.get("visitor_ORtg"),
        "DRtg":   r.get("visitor_DRtg"),
        "NRtg":   r.get("visitor_NRtg"),
        "Pace":   r.get("visitor_Pace"),
        "TS%":    r.get("visitor_TS%"),
        "eFG%":   r.get("visitor_eFG%"),
        "TOV%":   r.get("visitor_TOV%"),
        "ORB%":   r.get("visitor_ORB%"),
        "FTr":    r.get("visitor_FTr"),
        "3PAr":   r.get("visitor_3PAr"),
        "D_eFG%": r.get("visitor_D_eFG%"),
        "D_TOV%": r.get("visitor_D_TOV%"),
        "D_DRB%": r.get("visitor_D_DRB%"),
    }


def _build_feature_vector(h: dict, v: dict) -> pd.DataFrame:
    home_form    = h["win_pct_last5"] - h["win_pct_last10"]
    visitor_form = v["win_pct_last5"] - v["win_pct_last10"]
    row = {
        "diff_avg_pts_scored":         h["avg_pts_scored"]  - v["avg_pts_scored"],
        "diff_avg_pts_allowed":        h["avg_pts_allowed"] - v["avg_pts_allowed"],
        "diff_avg_pts_last5":          h["avg_pts_last5"]   - v["avg_pts_last5"],
        "diff_win_pct_last5":          h["win_pct_last5"]   - v["win_pct_last5"],
        "diff_win_pct_last10":         h["win_pct_last10"]  - v["win_pct_last10"],
        "diff_pt_diff_last10":         h["pt_diff_last10"]  - v["pt_diff_last10"],
        "diff_days_rest":              0,
        "home_home_win_pct_last10":    h["home_win_pct_last10"],
        "visitor_away_win_pct_last10": v["away_win_pct_last10"],
        "home_b2b":                    0,
        "visitor_b2b":                 0,
        "diff_SRS":    h["SRS"]    - v["SRS"],
        "diff_ORtg":   h["ORtg"]   - v["ORtg"],
        "diff_DRtg":   h["DRtg"]   - v["DRtg"],
        "diff_NRtg":   h["NRtg"]   - v["NRtg"],
        "diff_Pace":   h["Pace"]   - v["Pace"],
        "diff_TS%":    h["TS%"]    - v["TS%"],
        "diff_eFG%":   h["eFG%"]   - v["eFG%"],
        "diff_TOV%":   h["TOV%"]   - v["TOV%"],
        "diff_ORB%":   h["ORB%"]   - v["ORB%"],
        "diff_FTr":    h["FTr"]    - v["FTr"],
        "diff_3PAr":   h["3PAr"]   - v["3PAr"],
        "diff_D_eFG%": h["D_eFG%"] - v["D_eFG%"],
        "diff_D_TOV%": h["D_TOV%"] - v["D_TOV%"],
        "diff_D_DRB%": h["D_DRB%"] - v["D_DRB%"],
        "home_form":    home_form,
        "visitor_form": visitor_form,
        "diff_form":    home_form - visitor_form,
    }
    return pd.DataFrame([row])[FEATURE_COLS]


# Plotly chart helpers

_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(size=13),
    margin=dict(l=10, r=10, t=40, b=10),
)


def _plotly_roc(y_true, y_pred_prob) -> go.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    auc = roc_auc_score(y_true, y_pred_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"XGBoost (AUC = {auc:.3f})",
        line=dict(color=_BLUE, width=2.5),
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random guess",
        line=dict(color=_GREY, dash="dash", width=1.5),
        hoverinfo="skip",
    ))
    fig.update_layout(
        **_LAYOUT,
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
    )
    return fig


def _plotly_precision_recall(y_true, y_pred_prob, threshold: float) -> go.Figure:
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=precision[:-1], mode="lines",
        name="Precision", line=dict(color=_BLUE, width=2.5),
        hovertemplate="Threshold: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=thresholds, y=recall[:-1], mode="lines",
        name="Recall", line=dict(color="#E07B39", width=2.5),
        hovertemplate="Threshold: %{x:.3f}<br>Recall: %{y:.3f}<extra></extra>",
    ))
    fig.add_vline(
        x=threshold, line_dash="dash", line_color=_RED, line_width=1.5,
        annotation_text=f"t = {threshold:.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        **_LAYOUT,
        title="Precision / Recall vs Threshold",
        xaxis_title="Threshold",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1.05]),
        legend=dict(x=0.6, y=0.5),
    )
    return fig


def _plotly_confusion(y_true, y_pred_prob, threshold: float) -> go.Figure:
    y_pred = (y_pred_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Loss", "Win"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=[f"Pred: {l}" for l in labels],
        y=[f"True: {l}" for l in labels],
        colorscale="Blues",
        showscale=False,
        text=cm,
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=22),
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
    ))
    fig.update_layout(**_LAYOUT, title="Confusion Matrix")
    return fig


def _plotly_calibration(calibrated_model, X_val, y_val) -> go.Figure:
    prob_true, prob_pred = calibration_curve(
        y_val, calibrated_model.predict_proba(X_val)[:, 1], n_bins=10
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Perfectly calibrated",
        line=dict(color=_GREY, dash="dash", width=1.5),
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=prob_pred, y=prob_true, mode="lines+markers",
        name="XGBoost",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(size=7),
        hovertemplate="Mean predicted: %{x:.3f}<br>Fraction positive: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        title="Calibration Curve",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        legend=dict(x=0.05, y=0.9),
    )
    return fig


def _plotly_feature_importance(importances: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=importances.values,
        y=importances.index,
        orientation="h",
        marker_color=_BLUE,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        **{**_LAYOUT, "margin": dict(l=180, r=10, t=50, b=10)},
        title="XGBoost Feature Importance (Gain)",
        xaxis_title="Normalised weight",
        yaxis=dict(tickfont=dict(size=11)),
        height=600,
    )
    return fig


def _plotly_prob_bar(prob: float, home: str, visitor: str, threshold: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[prob], y=[""],
        orientation="h",
        marker_color=_BLUE,
        name=home,
        width=0.4,
        hovertemplate=f"{home}: {{x:.1%}}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=[1 - prob], y=[""],
        orientation="h",
        marker_color=_GREY,
        name=visitor,
        width=0.4,
        hovertemplate=f"{visitor}: {{x:.1%}}<extra></extra>",
    ))
    fig.add_vline(
        x=0.5, line_dash="dot", line_color="black", line_width=1,
    )
    fig.update_layout(
        **{**_LAYOUT, "margin": dict(l=10, r=10, t=10, b=40)},
        barmode="stack",
        xaxis=dict(range=[0, 1], tickformat=".0%", showgrid=False),
        yaxis=dict(showticklabels=False),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=-0.4),
        height=120,
        annotations=[
            dict(x=prob / 2, y=0, text=f"<b>{prob:.1%}</b>",
                 showarrow=False, font=dict(color="white", size=15), yref="y"),
            dict(x=prob + (1 - prob) / 2, y=0, text=f"<b>{1 - prob:.1%}</b>",
                 showarrow=False, font=dict(color="#555", size=15), yref="y"),
        ],
    )
    return fig


# Load everything upfront

artefacts_ready = (
    (ARTEFACTS_DIR / "model.pkl").exists()
    and (ARTEFACTS_DIR / "metrics.json").exists()
    and (ARTEFACTS_DIR / "processed_df.parquet").exists()
)

if not artefacts_ready:
    st.error("Artefacts not found. Run `python train.py` from the project root first.")
    st.stop()

model     = load_model()
saved     = load_metrics()
df        = load_processed_df()
metrics   = saved["model"]
baseline  = saved["baseline"]
threshold = metrics["threshold"]

val_df      = df[df["Season"] == VAL_SEASON]
X_val       = val_df[FEATURE_COLS]
y_val       = val_df[TARGET_COL].astype(int)
y_pred_prob = model.predict_proba(X_val)[:, 1]

recent_df = df[df["Season"] == TEST_SEASON]
all_teams = sorted(set(recent_df["Home"]) | set(recent_df["Visitor"]))

xgb_model   = model.calibrated_classifiers_[0].estimator
importances = pd.Series(
    xgb_model.feature_importances_, index=FEATURE_COLS
).sort_values()


# Tabs

tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Model & Metrics", "Evaluation", "Feature Importance"])


# Tab 1 — Predict (home page)

with tab1:
    st.title("NBA Game Winner Predictor")
    st.caption(
        f"stats sourced from Basketball Reference (2020–{TEST_SEASON[:4]})"
    )

    st.divider()

    col_home, col_vs, col_visitor, col_btn = st.columns([5, 1, 5, 2])

    with col_home:
        home_team = st.selectbox("Home team", all_teams, key="home")
    with col_vs:
        st.write("")
        st.write("")
        st.markdown("<div style='text-align:center; font-size:1.3rem; font-weight:600'>vs</div>", unsafe_allow_html=True)
    with col_visitor:
        default_visitor_idx = 1 if all_teams[0] == home_team else 0
        visitor_team = st.selectbox("Visitor team", all_teams, index=default_visitor_idx, key="visitor")
    with col_btn:
        st.write("")
        st.write("")
        predict = st.button("Predict", use_container_width=True, type="primary")

    if predict:
        if home_team == visitor_team:
            st.warning("Home and visitor team must be different.")
        else:
            h = _get_team_as_home(df, home_team)
            v = _get_team_as_visitor(df, visitor_team)

            if h is None:
                st.error(f"No home-game data found for {home_team}.")
            elif v is None:
                st.error(f"No away-game data found for {visitor_team}.")
            else:
                X_pred   = _build_feature_vector(h, v)
                prob     = float(model.predict_proba(X_pred)[0, 1])
                pred_win = prob >= threshold

                st.divider()

                res_left, res_right = st.columns([1, 2])

                with res_left:
                    st.metric("Home win probability", f"{prob:.1%}")
                    if pred_win:
                        st.success(f"**{home_team} wins**  (threshold {threshold:.2f})")
                    else:
                        st.info(f"**{visitor_team} wins**  (threshold {threshold:.2f})")

                with res_right:
                    st.plotly_chart(
                        _plotly_prob_bar(prob, home_team, visitor_team, threshold),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )

    st.divider()

    st.caption(
        f"Team stats are drawn from their most recent games in the {TEST_SEASON} season. "
        "Rest days and back-to-back status are assumed equal for both sides."
    )


# Tab 2 — Model & Metrics

with tab2:
    st.header("Model & Metrics")
    st.caption(f"Validated on the held-out {VAL_SEASON} season")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "Accuracy",
        f"{metrics['accuracy']:.1%}",
        f"{metrics['accuracy'] - baseline['accuracy']:+.1%} vs baseline",
    )
    col2.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    col3.metric(
        "Log Loss",
        f"{metrics['log_loss']:.4f}",
        f"{baseline['log_loss'] - metrics['log_loss']:+.4f} vs baseline",
        delta_color="inverse",
    )
    col4.metric(
        "Brier Score",
        f"{metrics['brier_score']:.4f}",
        f"{baseline['brier_score'] - metrics['brier_score']:+.4f} vs baseline",
        delta_color="inverse",
    )

    st.divider()

    st.subheader("About the model")
    st.markdown("""
        The model predicts whether the **home team wins** a given NBA regular-season game.

        **Data:** Per-game and advanced team statistics from Basketball Reference (2020–2024 seasons),
        combined with match-level results.

        **Pipeline:**
        1. Rolling season-to-date features computed for each team (points, win %, point differential)
        2. Previous-season advanced stats attached as baseline team-strength signals
        3. All features expressed as *home minus visitor* differentials to make the prediction symmetric
        4. XGBoost classifier tuned with Bayesian Optimisation over a time-series cross-validated objective
        5. Probability outputs calibrated with Platt scaling (sigmoid)
        6. Classification threshold selected to target ≥ 65% precision on the validation season
    """)

    st.divider()

    st.subheader("Baseline vs model")
    comparison = pd.DataFrame({
        "Metric":   ["Accuracy", "Log Loss", "Brier Score"],
        "Baseline": [baseline["accuracy"], baseline["log_loss"], baseline["brier_score"]],
        "Model":    [metrics["accuracy"],  metrics["log_loss"],  metrics["brier_score"]],
    })
    st.dataframe(comparison.set_index("Metric"), use_container_width=False)


# Tab 3 — Evaluation

with tab3:
    st.header("Model Evaluation")
    st.caption(f"All results on the held-out {VAL_SEASON} validation season")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    with row1_col1:
        st.plotly_chart(
            _plotly_roc(y_val, y_pred_prob),
            use_container_width=True,
        )

    with row1_col2:
        st.plotly_chart(
            _plotly_precision_recall(y_val, y_pred_prob, threshold),
            use_container_width=True,
        )

    with row2_col1:
        st.plotly_chart(
            _plotly_confusion(y_val, y_pred_prob, threshold),
            use_container_width=True,
        )

    with row2_col2:
        st.plotly_chart(
            _plotly_calibration(model, X_val, y_val),
            use_container_width=True,
        )


# Tab 4 — Feature Importance

with tab4:
    st.header("Feature Importance")
    st.caption("Gain-based importance from the underlying XGBoost classifier")

    st.plotly_chart(
        _plotly_feature_importance(importances),
        use_container_width=True,
    )

    st.markdown("""
**Feature groups:**
- **diff_avg_pts / diff_win_pct / diff_pt_diff** — rolling season-to-date performance gap between home and visitor
- **home_home_win_pct / visitor_away_win_pct** — venue-specific win rates (home teams tend to win more at home)
- **diff_SRS / diff_[O|D|N]Rtg / diff_Pace** — previous-season overall team strength signals
- **diff_TS% / diff_eFG% / ...** — previous-season offensive and defensive efficiency differentials
- **home_form / visitor_form / diff_form** — recent trajectory (last-5 vs last-10 win rate)
    """)
