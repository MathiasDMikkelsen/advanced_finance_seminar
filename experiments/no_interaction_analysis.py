"""
no_interaction_analysis.py
===========
Post-Earnings Announcement Drift (PEAD) ï¿½ Main Analysis Script (No Interaction Term)

Hypothesis:
    A larger disagreement gap (Management Guidance EPS - Analyst Consensus EPS) 
    creates informational complexity, causing an initial underreaction on the   
    announcement day and a stronger PEAD over the following 60 trading days.    

Main regression:
    CAR_2_60 = alpha + b1*Z_Surprise + b2*Z_Gap
             + b3*Std_Log_Market_Cap + Sector_Fixed_Effects + e

Data files: data/raw/Data til Advanced Finance seminar v*.xlsx (sheet: "Main Hard Copy")
            All key variables (CARs, Std_Surprise_EPS, Std_Disagreement_Gap) are
            pre-calculated in the Excel file via Bloomberg formulas.

How to run:
    python no_interaction_analysis.py

Output:
    Console  -- descriptive statistics and full regression summary
    results/regression_results.csv  -- coefficient table
"""

# ----------------------------------------------------------------------------- 
# Imports & paths
# ----------------------------------------------------------------------------- 

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT        = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data_files = list(RAW_DATA_DIR.glob("Data til Advanced Finance seminar v*.xlsx"))

if not data_files:
    sys.exit(
        f"\nERROR: No data files found in:\n  {RAW_DATA_DIR}\n\n"
        "Please place the 'Data til Advanced Finance seminar vX.xlsx' files in the data/raw/ folder and re-run.\n"
    )


# ============================================================================= 
# TASK 1 -- LOAD DATA
# ============================================================================= 

print("\n" + "=" * 65)
print("TASK 1 -- Loading data")
print("=" * 65)

dfs = []
for file_path in sorted(data_files):
    print(f"  Loading {file_path.name}...")
    temp_df = pd.read_excel(file_path, sheet_name="Main Hard Copy")
    dfs.append(temp_df)

df = pd.concat(dfs, ignore_index=True)

print(f"\n  Loaded {len(df)} events - {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}\n")


# ============================================================================= 
# TASK 2 -- VARIABLE CONSTRUCTION
# ============================================================================= 

print("=" * 65)
print("TASK 2 -- Constructing variables")
print("=" * 65)

# -- Std_Surprise_EPS  (Standardized Surprise) --------------------------------
if "Std_Surprise_EPS" not in df.columns or df["Std_Surprise_EPS"].isna().all(): 
    print("  'Std_Surprise_EPS' not found -- falling back to raw computation.") 
    df["Std_Surprise_EPS"] = df["Actual_EPS"] - df["Analyst_Consensus_EPS"]     
else:
    print("  'Std_Surprise_EPS' loaded from Excel.")

# -- Std_Disagreement_Gap  (Standardized Gap) --------------------------------- 
if "Std_Disagreement_Gap" not in df.columns or df["Std_Disagreement_Gap"].isna().all():
    print("  'Std_Disagreement_Gap' not found -- falling back to raw computation.")
    df["Std_Disagreement_Gap"] = df["Mgmt_Guidance_EPS"] - df["Analyst_Consensus_EPS"]
else:
    print("  'Std_Disagreement_Gap' loaded from Excel.")

# -- Continuous Variable Centering and Standardization --------------------------
df["Z_Surprise"] = (df["Std_Surprise_EPS"] - df["Std_Surprise_EPS"].mean()) / df["Std_Surprise_EPS"].std()
df["Z_Gap"] = (df["Std_Disagreement_Gap"] - df["Std_Disagreement_Gap"].mean()) / df["Std_Disagreement_Gap"].std()

# -- Control Variables: Market Cap and Sector --------------------------------- 
# We use the natural log of Market Cap to account for size effects, and standardize it.
if "Market_Cap" in df.columns:
    log_mc = np.log(df["Market_Cap"].replace(0, np.nan))
    df["Std_Log_Market_Cap"] = (log_mc - log_mc.mean()) / log_mc.std()
    print("  'Std_Log_Market_Cap' computed.")
else:
    df["Std_Log_Market_Cap"] = np.nan

# We create dummy variables for the Sector column.
sector_cols = []
if "Sector" in df.columns:
    sector_dummies = pd.get_dummies(df["Sector"], prefix="Sector", drop_first=True, dtype=int)
    df = pd.concat([df, sector_dummies], axis=1)
    sector_cols = sector_dummies.columns.tolist()
    print(f"  {len(sector_cols)} Sector dummy variables created.")

# -- Print variable overview -------------------------------------------------- 
display_cols = [
    "Ticker", "Event_ID",
    "Actual_EPS", "Analyst_Consensus_EPS", "Mgmt_Guidance_EPS",
    "Z_Surprise", "Z_Gap",
    "Std_Log_Market_Cap", "Sector",
    "CAR_0_1", "CAR_2_60",
]
display_cols = [c for c in display_cols if c in df.columns]

print("\nPreview of key variables (first 10 rows):")
print(df[display_cols].head(10).to_string(index=False))

key_vars = ["Z_Surprise", "Z_Gap",
            "Std_Log_Market_Cap", "CAR_0_1", "CAR_2_60"]
key_vars = [c for c in key_vars if c in df.columns]

print("\nDescriptive statistics:")
print(df[key_vars].describe().round(4).to_string())


# ============================================================================= 
# TASK 3 -- OLS REGRESSION
# ============================================================================= 

print("\n" + "=" * 65)
print("TASK 3 -- OLS Regression")
print("  Dependent variable : CAR_2_60 (post-announcement drift, pp)")
print("  Regressors         : Z_Surprise")
print("                       Z_Gap")
print("                       Std_Log_Market_Cap")
print("                       Sector Fixed Effects")
print("  Standard errors    : Clustered by Firm (Ticker)")
print("=" * 65)

print("\n" + "-" * 65)
print("MODEL 1 -- Baseline PEAD")
print("  Regressors         : Z_Surprise")
print("-" * 65)

DEPENDENT  = "CAR_2_60"
REGRESSORS = ["Z_Surprise"]

reg_df = df[["Ticker", DEPENDENT] + REGRESSORS].dropna()
n_obs  = len(reg_df)

if n_obs >= len(REGRESSORS) + 2:
    y = reg_df[DEPENDENT]
    X = sm.add_constant(reg_df[REGRESSORS])
    result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": reg_df["Ticker"]})
    print("\n")
    print(result.summary())

print("\n" + "-" * 65)
print("MODEL 2 -- Add Disagreement")
print("  Regressors         : Z_Surprise")
print("                       Z_Gap")
print("-" * 65)

REGRESSORS = ["Z_Surprise", "Z_Gap"]
reg_df = df[["Ticker", DEPENDENT] + REGRESSORS].dropna()
n_obs  = len(reg_df)

if n_obs >= len(REGRESSORS) + 2:
    y = reg_df[DEPENDENT]
    X = sm.add_constant(reg_df[REGRESSORS])
    result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": reg_df["Ticker"]})
    print("\n")
    print(result.summary())

print("\n" + "-" * 65)
print("MODEL 3 -- Full Specification (No Interaction)")
print("  Regressors         : Z_Surprise")
print("                       Z_Gap")
print("                       Std_Log_Market_Cap")
print("                       Sector Fixed Effects")
print("-" * 65)

REGRESSORS = ["Z_Surprise", "Z_Gap", "Std_Log_Market_Cap"] + sector_cols
reg_df = df[["Ticker", DEPENDENT] + REGRESSORS].dropna()
n_obs  = len(reg_df)

if n_obs >= len(REGRESSORS) + 2:
    y = reg_df[DEPENDENT]
    X = sm.add_constant(reg_df[REGRESSORS])
    result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": reg_df["Ticker"]})
    print("\n")
    print(result.summary())

    coef_table = pd.DataFrame({
        "Coefficient" : result.params,
        "Std_Error"   : result.bse,
        "t_stat"      : result.tvalues,
        "p_value"     : result.pvalues,
        "CI_lower_95" : result.conf_int()[0],
        "CI_upper_95" : result.conf_int()[1],
    })
    out_path = RESULTS_DIR / "no_interaction_regression_results.csv"
    coef_table.to_csv(out_path)
    print(f"\n  Coefficient table saved to: {out_path.relative_to(ROOT)}")      
