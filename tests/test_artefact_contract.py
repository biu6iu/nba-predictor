"""
The dashboard reads artefacts written by train.py. These tests check that what training writes
contains every key the dashboard reads, so a rename on either side fails here, not in the browser.

The keys the dashboard reads are extracted from dashboard/app.py itself, so the test follows the
dashboard when it starts reading something new.
"""

import ast
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import train as train_script
from src import model as model_module
from src.config import (
    FEATURE_COLS,
    MODEL_ARTEFACT_KEYS,
    TARGET_COL,
    TEST_SEASON,
    TRAIN_SEASONS,
    VAL_SEASON,
)
from src.evaluate import save_metrics

DASHBOARD = train_script.ARTEFACTS_DIR.parent / "dashboard" / "app.py"

MODEL_NAMES = {"artefact"}
METRIC_NAMES = {"metrics", "baseline", "val_metrics"}
DF_NAMES = {"df", "recent_df", "season_df", "rows"}


def _str_key(node: ast.Subscript) -> str | None:
    key = node.slice
    return key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None


def _dashboard_reads() -> dict[str, set[str]]:
    """String keys the dashboard indexes out of each artefact."""
    tree = ast.parse(DASHBOARD.read_text())
    reads: dict[str, set[str]] = {
        name: set()
        for name in ("model_pkl", "metrics_splits", "metrics_groups", "metric_names", "df_columns")
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and (key := _str_key(node)) is not None:
            base = node.value
            if isinstance(base, ast.Name):
                if base.id in MODEL_NAMES:
                    reads["model_pkl"].add(key)
                elif base.id == "saved":
                    reads["metrics_splits"].add(key)
                elif base.id in METRIC_NAMES:
                    reads["metric_names"].add(key)
                elif base.id in DF_NAMES:
                    reads["df_columns"].add(key)
            elif (isinstance(base, ast.Subscript) and isinstance(base.value, ast.Name)
                  and base.value.id == "saved"):
                reads["metrics_groups"].add(key)  # saved["test"]["model"]
        # r.get("home_SRS") on a row of the processed frame
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
              and node.func.value.id == "r" and node.args
              and isinstance(node.args[0], ast.Constant)):
            reads["df_columns"].add(node.args[0].value)
    return reads


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Run the real train() on synthetic data with a tiny search, writing into a temp dir."""
    out = tmp_path_factory.mktemp("artefacts")
    rng = np.random.default_rng(0)
    seasons = [*TRAIN_SEASONS, VAL_SEASON, TEST_SEASON]
    n = 250
    df = pd.DataFrame(rng.normal(size=(n * len(seasons), len(FEATURE_COLS))), columns=FEATURE_COLS)
    df["Season"] = np.repeat(seasons, n)
    df[TARGET_COL] = (df[FEATURE_COLS[0]] + rng.normal(scale=1.5, size=len(df))) > 0

    patch = pytest.MonkeyPatch()
    patch.setattr(model_module, "ARTEFACTS_DIR", out)
    patch.setattr(model_module, "BAYES_OPT_INIT_POINTS", 2)
    patch.setattr(model_module, "BAYES_OPT_N_ITER", 1)
    try:
        calibrated, threshold, _ = model_module.train(df)
    finally:
        patch.undo()

    val, _, _, _ = train_script.evaluate_split(calibrated, df, VAL_SEASON, threshold)
    test, _, _, _ = train_script.evaluate_split(calibrated, df, TEST_SEASON, threshold)
    metrics_path = out / "metrics.json"
    save_metrics({"validation": val, "test": test}, metrics_path)
    return {"dir": out, "metrics_path": metrics_path}


def test_dashboard_key_extraction_finds_something():
    # guards against the parser silently matching nothing and every contract test passing vacuously
    reads = _dashboard_reads()
    assert {"model", "threshold", "sklearn_version", "xgboost_version"} <= reads["model_pkl"]
    assert {"test", "validation"} <= reads["metrics_splits"]
    assert {"model", "baseline"} <= reads["metrics_groups"]
    assert {"accuracy", "roc_auc", "log_loss", "brier_score"} <= reads["metric_names"]
    assert {"Season", "Home", "Visitor", "home_SRS"} <= reads["df_columns"]


def test_model_pkl_has_every_key_the_dashboard_reads(trained):
    artefact = joblib.load(trained["dir"] / "model.pkl")
    assert isinstance(artefact, dict)
    assert set(artefact) == set(MODEL_ARTEFACT_KEYS)
    assert _dashboard_reads()["model_pkl"] <= set(artefact)


def test_model_pkl_contents_are_usable(trained):
    artefact = joblib.load(trained["dir"] / "model.pkl")
    assert 0 < artefact["threshold"] < 1
    assert artefact["feature_cols"] == FEATURE_COLS
    assert hasattr(artefact["model"], "predict_proba")


def test_metrics_json_has_every_key_the_dashboard_reads(trained):
    saved = json.loads(trained["metrics_path"].read_text())
    reads = _dashboard_reads()
    assert reads["metrics_splits"] <= set(saved)
    for split in reads["metrics_splits"]:
        assert reads["metrics_groups"] <= set(saved[split]), split
    # the dashboard reads the model keys from the model group and the shared ones from baseline
    for split in saved.values():
        assert reads["metric_names"] <= set(split["model"])
        assert {"accuracy", "log_loss", "brier_score"} <= set(split["baseline"])


def test_processed_frame_has_every_column_the_dashboard_reads(features_df):
    # features_df is what train.py writes to processed_df.parquet
    needed = _dashboard_reads()["df_columns"] | {TARGET_COL} | set(FEATURE_COLS)
    assert needed <= set(features_df.columns), sorted(needed - set(features_df.columns))
