# Advanced Finance Seminar — PEAD Analysis

> **Course**: Advanced Finance Seminar, 8th semester  
> **University**: University of Copenhagen  
> **Topic**: Post-Earnings Announcement Drift (PEAD)

---

## Hypothesis

A larger disagreement gap (Management Guidance EPS − Analyst Consensus EPS) creates informational complexity, causing an initial underreaction on the announcement day and a stronger PEAD over the following 60 trading days.

**Main regression:**
```
CAR_2_60 = α + β1·Std_Surprise_EPS + β2·Std_Disagreement_Gap
         + β3·(Std_Surprise_EPS × Std_Disagreement_Gap) + Controls + ε
```

---

## Project structure

```
advanced_finance_seminar/
├── data/
│   └── raw/
│       └── testdata_clean.xlsx   ← PUT THE EXCEL FILE HERE
├── results/                ← CSV output from the regression (auto-created)
├── analysis.py             ← MAIN SCRIPT — run this
├── requirements.txt
└── README.md
```

---

## Setup (first time only)

### 1. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies (already done, but in case of a fresh clone)

```powershell
pip install -r requirements.txt
```

---

## How to run

1. Place `Testdata.xlsx` in `data/raw/`.
2. Open `analysis.py` in VS Code.
3. Run it — either press **F5** or in the terminal:

```powershell
python analysis.py
```

The script will print the parsed CARs, constructed variables, descriptive
statistics, and the full regression summary to the console.  
The coefficient table is also saved to `results/regression_results.csv`.

---

## Data file location

Both partners should place the Excel file at:
```
data/raw/Testdata.xlsx
```
This path is **relative to the project root**, so it works on any machine
as long as the folder structure is intact.  
The `data/raw/` folder is listed in `.gitignore` — data files are **not**
committed to Git (to keep the repo lightweight and avoid sharing large files
via version control).

---

## Key dependencies

| Package | Purpose |
|---|---|
| `pandas` | Data loading and wrangling |
| `numpy` | Numerical operations |
| `statsmodels` | OLS regression with robust standard errors |

---

## TODO — later iterations

- Scale `Surprise_EPS` and `Disagreement_Gap` by the pre-announcement stock price
- Add control variables (e.g., `log(Market_Cap)`, book-to-market, momentum)
- Extend to the full dataset (same Excel structure, no code changes needed)
