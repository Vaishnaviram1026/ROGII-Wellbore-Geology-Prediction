"""
P1 initial exploration + naive baselines for ROGII Wellbore Geology Prediction.

Usage:
    python src/p1_eda_baseline.py --data_dir /path/to/rogii-wellbore-geology-prediction

What it does (training wells only, where full TVT is known):
  1. Profiles every horizontal well (rows, length, GR missingness, TVT range).
  2. Finds the Prediction Start (PS) = first row where TVT_input is NaN.
  3. Scores two naive baselines on the rows AFTER PS, using the competition metric (RMSE, feet):
        A. "Hold last TVT"      -> predict the last known TVT for every later point
        B. "Linear dip extrap." -> extend the recent TVT-vs-MD slope forward
  4. Saves figures to figures/ and numbers to results/p1_results.json
     (build_p1_pdf.py reads these to fill the report).

ASSUMPTIONS to verify against the real files (printed at runtime):
  * Horizontal files match  *horizontal_well.csv  with columns MD, X, Y, Z, GR, TVT, TVT_input
  * In training files, TVT is complete and TVT_input is NaN from PS onward.
No credentials are needed; the data must be downloaded by you after accepting the Kaggle rules.
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOOKBACK_FT = 500  # window of known TVT used to estimate the dip for baseline B


def find_files(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*horizontal_well.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No *horizontal_well.csv files found under {data_dir}")
    return files


def rmse(err):
    err = np.asarray(err, dtype=float)
    err = err[~np.isnan(err)]
    return float(np.sqrt(np.mean(err ** 2))) if err.size else np.nan


def process_well(path):
    df = pd.read_csv(path)
    need = {"MD", "GR", "TVT", "TVT_input"}
    if not need.issubset(df.columns):
        return None  # e.g. test wells have no TVT
    well = os.path.basename(path).split("__")[0]
    known = df["TVT_input"].notna().to_numpy()
    if known.all() or not known.any() or df["TVT"].isna().all():
        return None
    ps = int(np.argmin(known))  # first False -> first NaN in TVT_input
    hist, fut = df.iloc[:ps], df.iloc[ps:]
    fut = fut[fut["TVT"].notna()]
    if len(hist) < 10 or fut.empty:
        return None

    last_tvt = hist["TVT_input"].iloc[-1]
    last_md = hist["MD"].iloc[-1]

    # Baseline A: hold last known TVT
    err_a = fut["TVT"] - last_tvt

    # Baseline B: linear extrapolation of recent slope (TVT vs MD)
    win = hist[hist["MD"] >= last_md - LOOKBACK_FT]
    slope = np.polyfit(win["MD"], win["TVT_input"], 1)[0] if len(win) >= 5 else 0.0
    pred_b = last_tvt + slope * (fut["MD"] - last_md)
    err_b = fut["TVT"] - pred_b

    return {
        "well": well,
        "rows": int(len(df)),
        "ps_index": ps,
        "n_pred_rows": int(len(fut)),
        "frac_pred": float(len(fut) / len(df)),
        "md_min": float(df["MD"].min()),
        "md_max": float(df["MD"].max()),
        "gr_missing_frac": float(df["GR"].isna().mean()),
        "tvt_range_after_ps": float(fut["TVT"].max() - fut["TVT"].min()),
        "net_tvt_change_after_ps": float(fut["TVT"].iloc[-1] - last_tvt),
        "slope_ft_per_ft": float(slope),
        "rmse_hold": rmse(err_a),
        "rmse_linear": rmse(err_b),
        "sq_err_hold_sum": float(np.nansum(err_a ** 2)),
        "sq_err_linear_sum": float(np.nansum(err_b ** 2)),
        "_df": df,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default=".")
    args = ap.parse_args()

    fig_dir = os.path.join(args.out_dir, "figures")
    res_dir = os.path.join(args.out_dir, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    files = find_files(args.data_dir)
    print(f"Found {len(files)} horizontal-well files")
    wells = [w for w in (process_well(f) for f in files) if w is not None]
    print(f"{len(wells)} wells usable for baseline scoring (have TVT after PS)")
    if not wells:
        raise SystemExit("No scoreable wells: check column names / TVT availability.")

    prof = pd.DataFrame([{k: v for k, v in w.items() if k != "_df"} for w in wells])
    prof.to_csv(os.path.join(res_dir, "well_profile.csv"), index=False)

    n_total = int(prof["n_pred_rows"].sum())
    pooled_hold = float(np.sqrt(prof["sq_err_hold_sum"].sum() / n_total))
    pooled_lin = float(np.sqrt(prof["sq_err_linear_sum"].sum() / n_total))

    summary = {
        "n_files_found": len(files),
        "n_wells_scored": int(len(prof)),
        "total_rows": int(prof["rows"].sum()),
        "total_pred_rows": n_total,
        "median_rows_per_well": float(prof["rows"].median()),
        "median_frac_predicted": float(prof["frac_pred"].median()),
        "median_gr_missing_frac": float(prof["gr_missing_frac"].median()),
        "max_gr_missing_frac": float(prof["gr_missing_frac"].max()),
        "pooled_rmse_hold_last_tvt": pooled_hold,
        "pooled_rmse_linear_dip": pooled_lin,
        "median_well_rmse_hold": float(prof["rmse_hold"].median()),
        "median_well_rmse_linear": float(prof["rmse_linear"].median()),
        "frac_wells_linear_beats_hold": float((prof["rmse_linear"] < prof["rmse_hold"]).mean()),
        "lookback_ft": LOOKBACK_FT,
    }
    with open(os.path.join(res_dir, "p1_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # ---- Figure 1: example wells, known vs future TVT with baselines ----
    order = prof.sort_values("rmse_hold")
    picks = [order.iloc[0]["well"], order.iloc[len(order) // 2]["well"], order.iloc[-1]["well"]]
    labels = ["lowest hold-RMSE", "median hold-RMSE", "highest hold-RMSE"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    for ax, wid, lab in zip(axes, picks, labels):
        w = next(x for x in wells if x["well"] == wid)
        df, ps = w["_df"], w["ps_index"]
        last_tvt = df["TVT_input"].iloc[ps - 1]
        last_md = df["MD"].iloc[ps - 1]
        ax.plot(df["MD"], df["TVT"], color="#1f4e79", lw=1.6, label="True TVT")
        ax.plot(df["MD"].iloc[:ps], df["TVT_input"].iloc[:ps], color="#2e8b57", lw=2.4, label="Known (before PS)")
        ax.hlines(last_tvt, last_md, df["MD"].iloc[-1], color="#c0392b", ls="--", label="Baseline A: hold")
        ax.plot([last_md, df["MD"].iloc[-1]],
                [last_tvt, last_tvt + w["slope_ft_per_ft"] * (df["MD"].iloc[-1] - last_md)],
                color="#e67e22", ls=":", lw=2, label="Baseline B: linear")
        ax.axvline(last_md, color="k", lw=0.8)
        ax.set_title(f"{wid} ({lab})\nRMSE hold {w['rmse_hold']:.1f} ft | linear {w['rmse_linear']:.1f} ft", fontsize=9)
        ax.set_xlabel("Measured depth, MD (ft)")
        ax.set_ylabel("TVT (ft)")
    axes[0].legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_baseline_examples.png"), dpi=200)
    plt.close()

    # ---- Figure 2: distribution of per-well baseline error ----
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    bins = np.linspace(0, max(prof["rmse_hold"].quantile(0.98), prof["rmse_linear"].quantile(0.98)), 30)
    ax.hist(prof["rmse_hold"].clip(upper=bins[-1]), bins=bins, alpha=0.65, color="#c0392b", label="A: hold last TVT")
    ax.hist(prof["rmse_linear"].clip(upper=bins[-1]), bins=bins, alpha=0.65, color="#e67e22", label="B: linear extrapolation")
    ax.set_xlabel("Per-well RMSE after Prediction Start (ft)")
    ax.set_ylabel("Number of wells")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_baseline_rmse_hist.png"), dpi=200)
    plt.close()

    # ---- Figure 3: how much of each well must be predicted + GR missingness ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].hist(prof["frac_pred"], bins=25, color="#1f4e79")
    axes[0].set_xlabel("Fraction of each lateral that must be predicted")
    axes[0].set_ylabel("Number of wells")
    axes[1].hist(prof["gr_missing_frac"] * 100, bins=25, color="#2e8b57")
    axes[1].set_xlabel("Gamma-ray values missing (% of rows)")
    axes[1].set_ylabel("Number of wells")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig_profile_hists.png"), dpi=200)
    plt.close()
    print("Saved figures to", fig_dir, "and results to", res_dir)


if __name__ == "__main__":
    main()
