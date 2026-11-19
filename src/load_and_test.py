
"""
Load the trained ROGII horizontal-only core model and
generate TVT predictions for one or multiple wells.

Run from the project root:

    python src/load_and_test.py --well_csv "data/raw/test/EXAMPLE__horizontal_well.csv"

    python src/load_and_test.py --test_dir "data/raw/test"

    python src/load_and_test.py --demo
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Project paths and feature imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(SRC_DIR))

from Rogii_Features import FEATURES, predict_tvt


# --------------------------------------------------
# 2. Synthetic demonstration well
# --------------------------------------------------

def synthetic_well(n=3000, ps=900, seed=1):

    rng = np.random.default_rng(seed)

    md = 10000 + np.arange(n, dtype=float)

    tvt = 5000 + np.cumsum(
        rng.normal(0.004, 0.05, n)
    )

    z = (
        8000
        - 0.9 * (tvt - 5000)
        + rng.normal(0, 0.05, n)
    )

    df = pd.DataFrame({
        "MD": md,
        "X": md * 0.8,
        "Y": md * 0.6,
        "Z": z,
        "GR": 80 + rng.normal(0, 15, n),
        "TVT": tvt,
        "TVT_input": tvt
    })

    df.loc[ps:, "TVT_input"] = np.nan

    return df


# --------------------------------------------------
# 3. Load and verify trained model
# --------------------------------------------------

def load_model(artifact_dir):

    model_path = (
        artifact_dir / "rogii_tvt_core_model.joblib"
    )

    info_path = (
        artifact_dir / "rogii_tvt_core_model_info.json"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run TVT_Model_Training.ipynb first."
        )

    if not info_path.exists():
        raise FileNotFoundError(
            f"Model information not found: {info_path}"
        )

    model = joblib.load(model_path)

    with open(info_path, "r") as f:
        info = json.load(f)

    saved_features = info.get("features")

    if saved_features != list(FEATURES):
        raise ValueError(
            "Saved model features do not match "
            "Rogii_Features.py.\n"
            f"Saved: {saved_features}\n"
            f"Current: {FEATURES}"
        )

    if hasattr(model, "feature_names_in_"):
        if list(model.feature_names_in_) != list(FEATURES):
            raise ValueError(
                "Model feature names do not match "
                "the current feature builder."
            )

    print("MODEL LOADED SUCCESSFULLY")
    print("Model type:", info.get("model_type"))
    print("Features:", saved_features)
    print("Training wells:", info.get("trained_wells"))
    print("Training rows:", info.get("training_rows"))

    return model


# --------------------------------------------------
# 4. Predict one horizontal well
# --------------------------------------------------

def test_one_well(model, csv_path, output_dir):

    df = pd.read_csv(csv_path)

    if "TVT_input" not in df.columns:
        raise ValueError(
            "Missing TVT_input column."
        )

    hidden_mask = df["TVT_input"].isna()

    hidden_count = int(hidden_mask.sum())

    if hidden_count == 0:
        raise ValueError(
            "No hidden TVT_input rows found."
        )

    # Predict using the same feature builder
    # used during model training.
    out = predict_tvt(model, df)

    if len(out) != hidden_count:
        raise ValueError(
            f"Expected {hidden_count} predictions, "
            f"but received {len(out)}."
        )

    if not np.isfinite(
        out["TVT_pred"].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Predictions contain NaN or infinity."
        )

    # Save predictions
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_dir
        / f"{Path(csv_path).stem}_predictions.csv"
    )

    out.to_csv(
        output_path,
        index=False
    )

    print(f"\nWell: {Path(csv_path).name}")
    print("Hidden rows:", hidden_count)

    print(
        out.head(3).round(2).to_string(index=False)
    )

    print("Saved:", output_path)

    # Evaluation only when actual TVT exists.
    metrics = {}

    if "TVT" in df.columns:

        truth = pd.to_numeric(
            df.loc[hidden_mask, "TVT"],
            errors="coerce"
        ).to_numpy(dtype=float)

        valid = np.isfinite(truth)

        if valid.any():

            for col in [
                "TVT_pred",
                "TVT_hold",
                "TVT_linear"
            ]:

                predictions = out[col].to_numpy(
                    dtype=float
                )

                error = (
                    truth[valid]
                    - predictions[valid]
                )

                score = float(
                    np.sqrt(np.mean(error ** 2))
                )

                metrics[f"rmse_{col}"] = score

                print(
                    f"RMSE {col}: {score:.3f} ft"
                )

        else:

            print(
                "Actual hidden TVT values are unavailable."
            )

    else:

        print(
            "No TVT target column. "
            "Predictions generated without RMSE."
        )

    return {
        "file": Path(csv_path).name,
        "hidden_rows": hidden_count,
        "status": "PASSED",
        "prediction_file": str(output_path),
        **metrics
    }


# --------------------------------------------------
# 5. Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--well_csv",
        help="Path to one horizontal-well CSV"
    )

    parser.add_argument(
        "--test_dir",
        help="Directory containing horizontal-well CSVs"
    )

    parser.add_argument(
        "--artifact_dir",
        default=str(
            PROJECT_ROOT / "artifacts"
        )
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a synthetic demonstration well"
    )

    args = parser.parse_args()

    selected_modes = sum([
        bool(args.well_csv),
        bool(args.test_dir),
        bool(args.demo)
    ])

    if selected_modes != 1:
        parser.error(
            "Choose exactly one of "
            "--well_csv, --test_dir, or --demo."
        )

    artifact_dir = Path(args.artifact_dir)

    model = load_model(artifact_dir)

    output_dir = (
        PROJECT_ROOT
        / "results"
        / "test_predictions"
    )

    # Select input files
    if args.well_csv:

        files = [Path(args.well_csv)]

        if not files[0].exists():
            raise FileNotFoundError(
                f"Input file not found: {files[0]}"
            )

    elif args.test_dir:

        test_dir = Path(args.test_dir)

        files = sorted(
            test_dir.rglob("*horizontal_well.csv")
        )

        if not files:
            raise FileNotFoundError(
                f"No horizontal-well CSV files "
                f"found in {test_dir}"
            )

    else:

        print(
            "\nRunning synthetic demonstration."
        )

        df = synthetic_well()

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        demo_path = (
            output_dir / "synthetic_demo.csv"
        )

        df.to_csv(
            demo_path,
            index=False
        )

        files = [demo_path]

    print(
        f"\nFound {len(files)} horizontal-well files."
    )

    results = []

    for csv_path in files:

        try:

            result = test_one_well(
                model,
                csv_path,
                output_dir
            )

            results.append(result)

        except Exception as exc:

            print(
                f"\nFAILED: {csv_path.name}"
            )

            print("Reason:", exc)

            results.append({
                "file": csv_path.name,
                "status": "FAILED",
                "reason": str(exc)
            })

    # Save test summary
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path = (
        output_dir / "test_summary.csv"
    )

    pd.DataFrame(results).to_csv(
        summary_path,
        index=False
    )

    passed = sum(
        r["status"] == "PASSED"
        for r in results
    )

    failed = len(results) - passed

    print("\n========== TEST SUMMARY ==========")

    print("Total files:", len(results))
    print("Passed:", passed)
    print("Failed:", failed)
    print("Summary:", summary_path)

    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print(
            "Some files failed. "
            "Review the error messages."
        )


if __name__ == "__main__":
    main()
