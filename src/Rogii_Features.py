
"""Shared feature code for the ROGII TVT prediction project.

Core model: horizontal-well data only.

Prediction Start (PS):
    First row where TVT_input is NaN.

Inputs:
    Known TVT_input history before PS.
    MD, X, Y, Z, and GR measurements.

Target:
    TVT - last known TVT, in feet.

Important:
    Hidden TVT is never used as a model input.
    Training-only formation surface columns are ignored.
    Missing GR is imputed using historical GR only.
"""

import numpy as np
import pandas as pd


LOOKBACK_FT = 500

# Fixed feature order: this is the core model input contract.
FEATURES = [
    "dMD",
    "dZ",
    "dX",
    "dY",
    "GR_dev",
    "GR_missing",
    "slope_x_dMD",
]

REQUIRED_COLS = [
    "MD",
    "X",
    "Y",
    "Z",
    "GR",
    "TVT_input",
]


def find_ps(df):
    """Return the first hidden TVT_input row.

    Returns None if all rows are known or all rows are hidden.
    Rejects non-contiguous known/hidden sections.
    """
    known = df["TVT_input"].notna().to_numpy()

    if known.all() or not known.any():
        return None

    ps = int(np.flatnonzero(~known)[0])

    # Known TVT_input must form one continuous prefix.
    if known[ps:].any():
        raise ValueError(
            "TVT_input must contain known values before PS "
            "and NaN values continuously after PS."
        )

    return ps


def build_case(df, max_rows=None):
    """Build core horizontal-only features for one well.

    Returns:
        X: DataFrame with columns FEATURES.
        meta: DataFrame containing MD, last_tvt, slope, and TVT
              when TVT is available.
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    if max_rows is not None:
        if not isinstance(max_rows, (int, np.integer)) or max_rows < 1:
            raise ValueError("max_rows must be a positive integer or None.")

    df = df.copy().reset_index(drop=True)

    # Convert required measurements to numeric values.
    for col in REQUIRED_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coordinates and measured depth are required for prediction.
    for col in ["MD", "X", "Y", "Z"]:
        if not np.isfinite(df[col]).all():
            raise ValueError(f"Column {col} contains missing or non-finite values.")

    ps = find_ps(df)

    if ps is None or ps < 10:
        raise ValueError(
            "Need at least 10 known TVT_input rows followed by hidden rows."
        )

    hist = df.iloc[:ps].copy()
    fut = df.iloc[ps:].copy()

    last = hist.iloc[-1]
    last_tvt = float(last["TVT_input"])

    if not np.isfinite(last_tvt):
        raise ValueError("Last known TVT_input value is invalid.")

    # Use the recent historical window.
    win = hist[
        hist["MD"] >= last["MD"] - LOOKBACK_FT
    ].copy()

    # Estimate recent TVT trend using valid historical observations.
    slope_data = win[["MD", "TVT_input"]].dropna()

    if len(slope_data) >= 5 and slope_data["MD"].nunique() >= 2:
        slope = float(
            np.polyfit(
                slope_data["MD"].to_numpy(),
                slope_data["TVT_input"].to_numpy(),
                1,
            )[0]
        )
    else:
        slope = 0.0

    if not np.isfinite(slope):
        slope = 0.0

    # Historical GR reference only.
    historical_gr = win["GR"].dropna()

    if len(historical_gr) > 0:
        gr_ref = float(historical_gr.median())
    else:
        # Safe fallback when the historical window has no GR.
        gr_ref = 0.0

    # Optional subsampling for development experiments.
    # Use max_rows=None for complete evaluation and final training.
    if max_rows is not None and len(fut) > max_rows:
        positions = np.linspace(
            0,
            len(fut) - 1,
            max_rows,
        ).astype(int)

        fut = fut.iloc[positions].copy()

    future_gr = fut["GR"]

    gr_missing = future_gr.isna().astype(float)

    gr_filled = future_gr.fillna(gr_ref)

    X = pd.DataFrame({
        "dMD": fut["MD"].to_numpy() - float(last["MD"]),
        "dZ": fut["Z"].to_numpy() - float(last["Z"]),
        "dX": fut["X"].to_numpy() - float(last["X"]),
        "dY": fut["Y"].to_numpy() - float(last["Y"]),
        "GR_dev": gr_filled.to_numpy() - gr_ref,
        "GR_missing": gr_missing.to_numpy(),
    })

    X["slope_x_dMD"] = slope * X["dMD"]

    # Enforce exact feature order and check feature validity.
    X = X[FEATURES].astype(float)

    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Generated features contain non-finite values.")

    meta = pd.DataFrame({
        "MD": fut["MD"].to_numpy(),
        "last_tvt": last_tvt,
        "slope": slope,
    })

    # TVT is retained only as an evaluation target, never as a feature.
    if "TVT" in fut.columns:
        meta["TVT"] = pd.to_numeric(
            fut["TVT"], errors="coerce"
        ).to_numpy()

    return X, meta


def predict_tvt(model, df):
    """Predict TVT for every hidden row of one horizontal well.

    Returns:
        MD, TVT_pred, TVT_hold, TVT_linear
    """
    X, meta = build_case(df)

    delta = model.predict(X)

    return pd.DataFrame({
        "MD": meta["MD"].to_numpy(),
        "TVT_pred": meta["last_tvt"].to_numpy() + delta,
        "TVT_hold": meta["last_tvt"].to_numpy(),
        "TVT_linear": (
            meta["last_tvt"].to_numpy()
            + X["slope_x_dMD"].to_numpy()
        ),
    })
