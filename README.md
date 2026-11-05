# ROGII Wellbore Geology Prediction: MSBA 265 Final Project (P1)

## 1. Project summary

**Business problem.** A geosteering team drills a horizontal well through thin layers of rock and must keep the bit in the target layer. They estimate the bit's position in the rock column (called **TVT**, true vertical thickness) from gamma-ray (GR) readings and a nearby reference vertical well (the **typewell**).

**Question.** Using data available at the **Prediction Start (PS)** point, how accurately can a supervised model predict TVT for the rest of each lateral, compared with simple rules a geologist could apply by hand?

**Topic / method.** Supervised machine learning (regression). Planned model: gradient-boosted trees predicting the change in TVT from the last known value.

**Metric.** RMSE of TVT error in feet over all points after PS (the competition's metric).

**Status at P1.** Data profile and two naive baselines are done. **No trained model yet** (that is P2).

**Data source.** Kaggle competition *ROGII - Wellbore Geology Prediction*: https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction

---

## 2. Repository structure

```
ROGII-Wellbore-Geology-Prediction/
|-- README.md
|-- requirements.txt
|-- src/
|   `-- p1_eda_baseline.py        # data profile + two baselines 
|-- data/
|   `-- raw/            # (local only, excluded from GitHub)            
|       |-- train/                # training wells (horizontal + typewell CSVs)
|       |-- test/                 # test wells (no TVT target)
|       `-- (competition zip and host task slides)
|-- figures/                      # created by the script
|   |-- fig_baseline_examples.png
|   |-- fig_baseline_rmse_hist.png
|   `-- fig_profile_hists.png
`-- results/                      # created by the script
    |-- p1_results.json
    `-- well_profile.csv

```

### Data Availability

The raw dataset files are stored locally on my computer and are not included in this GitHub repository because of their large size and the Kaggle competition's data-use requirements. The repository contains the code, figures, and results generated from the analysis. Please follow the download instructions above to obtain the dataset.
---

## 3. Environment setup

### 1. Clone the Repository

```bash
git clone https://github.com/Vaishnaviram1026/ROGII-Wellbore-Geology-Prediction
cd ROGII-Wellbore-Geology-Prediction
```

Requires **Python 3.10 or newer** (developed with Python 3.12). Run these from the project root.

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` first.

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

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

## 5. Run the analysis

From the project root, with the environment active:

```bash
python src/p1_eda_baseline.py --data_dir data/raw/train
```
(Windows: `python src\p1_eda_baseline.py --data_dir data\raw\train`)

Use `data/raw/train` so only training wells (which have the TVT answer) are scored. The script searches subfolders for `*horizontal_well.csv`, skips any file without usable TVT after PS, and prints a summary.

**What it does**
1. Profiles every horizontal well: rows, share after PS, GR missingness.
2. Finds PS as the first row where `TVT_input` is blank.
3. Scores two baselines on points after PS (no model is fitted, so scoring on training wells is not leakage):
   - **A. Hold last TVT:** predict the last known TVT for every later point.
   - **B. Linear dip extrapolation:** extend the TVT-vs-MD slope from the last 500 ft.
4. Writes figures to `figures/` and numbers to `results/`.

**Runtime:** about a minute or two for ~770 wells on a normal laptop.

---

## 6. Expected outputs

Files created: `results/p1_results.json`, `results/well_profile.csv`, `figures/fig_baseline_examples.png`, `figures/fig_baseline_rmse_hist.png`, `figures/fig_profile_hists.png`.

My run produced the following in `results/p1_results.json` (a correct reproduction should match; small differences mean different input files):

| Quantity | Expected value |
|---|---|
| Horizontal-well files found / scored | 776 / 773 (if you point at `data/raw` instead of `data/raw/train`) |
| Total rows / rows to predict | 5,092,255 / 3,783,989 |
| Median rows per well | 6,576 |
| Median share of well to predict | 74.0% |
| Median / max GR missing per well | 27.7% / 80.1% |
| **Baseline A (hold) pooled / median per-well RMSE** | **15.91 ft / 10.67 ft** |
| **Baseline B (linear) pooled / median per-well RMSE** | **160.33 ft / 38.73 ft** |
| Share of wells where B beats A | 12.5% |

> If your "found / scored" counts differ from 776 / 773, check which folder you passed to `--data_dir`.
> I have not yet confirmed why 3 of 776 files were not scored; the likely cause is files without usable TVT after PS (for example test wells). Check `results/well_profile.csv` (one row per scored well) against the list of files in the folder.

---

## 7. Key P1 findings

- Holding the last known TVT is the stronger naive rule (15.9 ft pooled RMSE). Linear dip extrapolation is far worse and wins in only 12.5% of wells.
- Errors are concentrated in a minority of wells (pooled RMSE exceeds the median).
- Gamma ray is missing for a median of 27.7% of rows per well, so missing-data handling is a core preparation step.
- Working target for P2: about 25% below 15.9 ft, or roughly 11.9 ft (my own target, to be revisited).

---

## 9. Known limitations

- No predictive model yet; baselines only.
- Baselines are scored on training wells; competition leaderboard scores use a hidden test set and are not comparable.
- Findings come from one competition's wells and may not transfer to other basins.
- The data are subject to the Kaggle competition rules; do not share them.
- The task brief is the host's description; verify details against the competition Data tab.

## 11. References

- Kaggle / ROGII. *ROGII - Wellbore Geology Prediction* (2026). https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction
- Full reference list is in the P1 report.
