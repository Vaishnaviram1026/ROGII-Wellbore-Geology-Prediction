# ROGII Wellbore Geology Prediction: MSBA 265 Final Project (P1)

## 1. Project Overview

This project uses the **ROGII Wellbore Geology Prediction** dataset to predict **True Vertical Thickness (TVT)** along horizontal wellbores.

### Problem and Objective

During horizontal drilling, geologists and geosteering specialists use well trajectories and gamma-ray (GR) measurements to understand the wellbore's position relative to geological layers.

The objective is to improve TVT predictions beyond the **Prediction Start (PS)** point, where actual TVT values are hidden. More accurate predictions could support better-informed geosteering decisions and potentially reduce avoidable drilling time and costs.

### Methodology

The project uses **linear regression** with engineered features derived from well trajectory, gamma-ray measurements, and historical TVT information.

Model performance is compared against two baseline methods:

1. **Hold Last TVT:** Use the last known TVT value for all subsequent predictions.
2. **Linear Dip Extrapolation:** Extend the recent TVT trend along measured depth.

Evaluation uses **five-fold GroupKFold cross-validation by well**, with Root Mean Squared Error (RMSE) as the primary metric.

### Project Outputs

The repository includes exploratory data analysis, feature-engineering code, a model-training notebook, validation results, visualizations, a saved model, and a script for generating predictions on test wells.

The model is intended to support geologists and geosteering specialists in decision-making, not to automate drilling decisions.

**Data source.** Kaggle competition *ROGII - Wellbore Geology Prediction*: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction

---

## 2. Repository Structure

```text
ROGII-Wellbore-Geology-Prediction/
├── README.md
├── requirements.txt
├── src/
│   ├── p1_eda_baseline.py
│   ├── Rogii_Features.py
│   ├── TVT_Model_Training.ipynb
│   └── load_and_test.py
├── data/
│   └── raw/
│       ├── train/            # Kaggle training files (local only)
│       └── test/             # Kaggle test files (local only)
├── artifacts/
│   ├── rogii_tvt_core_model.joblib
│   └── rogii_tvt_core_model_info.json
├── results/                  # Evaluation and prediction CSVs
└── figures/                  # Analysis plots
```

## 3. Environment Setup

**Requirements:** Python 3.10 or newer (developed using Python 3.12).

### Step 1: Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Vaishnaviram1026/ROGII-Wellbore-Geology-Prediction.git
cd ROGII-Wellbore-Geology-Prediction
```

### Step 2: Create and Activate a Virtual Environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

With the virtual environment activated, run:

```bash
python -m pip install -r requirements.txt
```

### Step 4: Create a Jupyter Kernel

Register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name rogii-wellbore --display-name "Python (ROGII Wellbore)"
```

When opening `src/TVT_Model_Training.ipynb` in VS Code, click **Select Kernel** in the upper-right corner and choose **Python (ROGII Wellbore)**.

### Data Availability

The raw dataset files are stored locally on my computer and are not included in this GitHub repository because of their large size and the Kaggle competition's data-use requirements. The repository contains the code, figures, and results generated from the analysis. Please follow the download instructions below to obtain the dataset.

## 4. Getting the data from Kaggle (required; data is not in this repo)

The competition data cannot be redistributed, so each person downloads it with their **own** Kaggle account.

### Step 1: Account and rules
1. Create or sign in to a Kaggle account.
2. Open the competition page: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
3. Click **Join / I Understand and Accept** on the Data tab. Without this, downloads fail with **403 Forbidden**.

### Step 2: Create your own API token
1. Kaggle: profile picture, then **Settings**, then **API**, then **Create New Token**.
2. Copy the token when it is shown (it appears only once).
3. **Never paste your token into chat, a README, a screenshot or a Git commit.** If you expose it by accident, revoke it in the same Settings page and create a new one.

### Step 3: Save the token (outside the repo)

**Windows (PowerShell)**
```powershell
mkdir $HOME\.kaggle -Force
Set-Content $HOME\.kaggle\access_token "YOUR_TOKEN"
```

**macOS / Linux**
```bash
mkdir -p ~/.kaggle
echo YOUR_TOKEN > ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```
For one session only you can instead set the environment variable `KAGGLE_API_TOKEN`.

### Step 4: Download and unzip

```bash
pip install kaggle
kaggle competitions download -c rogii-wellbore-geology-prediction -p data/raw
```

Unzip into `data/raw`:
- **Windows:** `Expand-Archive data\raw\rogii-wellbore-geology-prediction.zip -DestinationPath data\raw`
- **macOS / Linux:** `unzip data/raw/rogii-wellbore-geology-prediction.zip -d data/raw`

Check you have `data/raw/train/` and `data/raw/test/`.

### Data description (per the competition host's task brief)
Each well has two CSV files:
- **Horizontal well file** (`*horizontal_well.csv`): one row per point along the lateral. Columns include `MD` (measured depth), `X`, `Y`, `Z` (position), `GR` (gamma ray, some missing), `TVT` (target, training only) and `TVT_input` (TVT known up to PS; blank after). Training files also contain six formation-top depth columns (training only).
- **Typewell file** (`*typewell*.csv`): the reference vertical well with `TVT`, `GR`, `Geology`.

All units are feet. The files use anonymised IDs in their names.

---

## 4b. Troubleshooting downloads

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'six'` (or another module) | `pip install six`, or `pip install --upgrade --force-reinstall kaggle` |
| `403 Forbidden` | Accept the competition rules in the browser (Step 1) |
| `401` / "could not find credentials" | Check `~/.kaggle/access_token` exists and holds a **new, valid** token |
| `No *horizontal_well.csv files found` | Check `--data_dir` points at the folder that holds the files (section 5) |
| `can't open file 'src\...'` | Run commands from the project root and confirm `src/p1_eda_baseline.py` exists |

---

## 5. Run the Project

Run the following steps in order after completing the environment setup and downloading the Kaggle dataset.

All terminal commands should be run from the **project root directory** with the virtual environment activated.

### Step 1: Run Exploratory Data Analysis and Baselines

Run:

```bash
python src/p1_eda_baseline.py --data_dir data/raw/train
```

This script:

- Profiles the training wells and checks gamma ray (GR) missingness.
- Identifies the Prediction Start (PS) point.
- Evaluates two baseline methods: holding the last known TVT and linear dip extrapolation.
- Saves baseline results and figures.

**Expected outputs:**

```text
results/p1_results.json
results/well_profile.csv

figures/fig_baseline_examples.png
figures/fig_baseline_rmse_hist.png
figures/fig_profile_hists.png
```

### Step 2: Feature Engineering

Feature engineering is implemented in:

```text
src/Rogii_Features.py
```

This module prepares the input features used by the regression model, including trajectory changes, gamma ray information, and the historical TVT trend.

**No separate terminal command is required for this step.** The training notebook uses the feature-engineering code.

### Step 3: Train and Evaluate the Model

Open the following notebook in VS Code:

```text
src/TVT_Model_Training.ipynb
```

1. Select the **Python (ROGII Wellbore)** Jupyter kernel created during environment setup.
2. Run the notebook cells from top to bottom.
3. Wait for model training and validation to finish.
4. Confirm that the model and evaluation outputs have been saved.

The notebook trains a linear regression model and evaluates it using five-fold cross-validation grouped by well.

**Expected outputs:**

```text
artifacts/rogii_tvt_core_model.joblib
artifacts/rogii_tvt_core_model_info.json

results/training_summary.json
results/fold_validation_metrics.csv
results/per_well_validation_metrics.csv
results/distance_band_metrics.csv
results/oof_predictions.csv

figures/oof_actual_vs_predicted.png
figures/rmse_by_distance.png
```
**Note on generated files:** The files `results/oof_predictions.csv` and `results/well_profile.csv` are excluded from GitHub to keep the repository lightweight. Both files are generated locally when running the analysis and training code. The smaller validation summary files, figures, and test predictions are included in the repository.


### Step 4: Load the Saved Model and Generate Test Predictions

After the training notebook has created the saved model, return to the project root directory and run:

```bash
python src/load_and_test.py --test_dir "data/raw/test"
```

This script loads the saved model and generates TVT predictions for the test wells.

**Expected outputs:**

```text
results/test_predictions/
    test_summary.csv
    00bbac68__horizontal_well_predictions.csv
    000d7d20__horizontal_well_predictions.csv
    00e12e8b__horizontal_well_predictions.csv
```

The test wells do not contain the hidden TVT target values, so their prediction accuracy cannot be evaluated locally.

### Step 5: Verify the Results

Check that all expected files have been created.

The training validation results should be approximately:

| Method | RMSE (ft) | MAE (ft) |
|---|---:|---:|
| Hold last TVT baseline | 15.61 | 10.95 |
| Linear dip extrapolation | 149.31 | 57.18 |
| Linear regression model | 14.84 | 10.47 |

The linear regression model improves RMSE by approximately **4.90%** compared with the hold-last-TVT baseline.

These results are based on grouped cross-validation using training wells. The test wells have hidden TVT values, so test-set prediction accuracy cannot be calculated locally.

## 6. Limitations

- **Model performance:** The linear regression model improves RMSE by approximately 4.90% over the hold-last-TVT baseline, but the improvement is modest.
- **Long-distance predictions:** Model accuracy decreases farther from Prediction Start, and the model performs worse than the hold baseline beyond 5,000 ft.
- **Limited features:** The model uses trajectory, gamma ray, and historical TVT features. Typewell information and more advanced models are not included.
- **Test evaluation:** Actual TVT values are hidden for the test wells, so test-set prediction accuracy cannot be measured locally.
- **Business impact:** Potential drilling cost savings are hypothetical and have not been validated using operational data.


## 7. References

- Kaggle / ROGII. *ROGII - Wellbore Geology Prediction* (2026). https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction

