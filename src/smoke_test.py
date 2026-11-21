import json
import sys

import joblib
import numpy as np
import pandas as pd

from src.config import ARTEFACTS_DIR, FEATURE_COLS, MODEL_ARTEFACT_KEYS


def check(artefacts_dir=ARTEFACTS_DIR) -> None:
    # explicit raises, not assert, so this still checks under `python -O`
    artefact = joblib.load(artefacts_dir / "model.pkl")
    if not isinstance(artefact, dict) or not set(MODEL_ARTEFACT_KEYS) <= set(artefact):
        raise RuntimeError("model.pkl is missing keys the dashboard reads")
    if artefact["feature_cols"] != FEATURE_COLS:
        raise RuntimeError("model.pkl was trained on different features than src/config.py lists")

    df = pd.read_parquet(artefacts_dir / "processed_df.parquet")
    probs = artefact["model"].predict_proba(df[FEATURE_COLS].head(5))[:, 1]
    if len(probs) == 0 or not np.all((probs >= 0) & (probs <= 1)):
        raise RuntimeError(f"model returned invalid probabilities: {probs}")

    json.loads((artefacts_dir / "metrics.json").read_text())


if __name__ == "__main__":
    try:
        check()
    except Exception as e:
        sys.exit(f"Smoke test failed: {e!r}")
    print("Smoke test passed")
