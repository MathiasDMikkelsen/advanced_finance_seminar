"""
analysis.py
===========
Post-Earnings Announcement Drift (PEAD) — Main Analysis Script

Hypothesis:
    A larger disagreement gap (Management Guidance EPS - Analyst Consensus EPS)
    creates informational complexity, causing an initial underreaction on the
    announcement day and a stronger PEAD over the following 60 trading days.

Main regression:
    CAR_2_60 = alpha + b1*Surprise_EPS + b2*Disagreement_Gap
             + b3*(Surprise_EPS x Disagreement_Gap) + e

Data file:  data/raw/testdata_clean.xlsx  (sheet: "Main Hard Copy")
            All key variables (CARs, Surprise_EPS, Disagreement_Gap) are
            pre-calculated in the Excel file via Bloomberg formulas.

How to run:
    python analysis.py

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

ROOT        = Path(__file__).parent
DATA_FILE   = ROOT / "data" / "raw" / "testdata_clean.xlsx"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    sys.exit(
        f"\nERROR: Data file not found at:\n  {DATA_FILE}\n\n"
        "Please place testdata_clean.xlsx in the data/raw/ folder and re-run.\n"
    )


# =============================================================================
# TASK 1 -- LOAD DATA
# =============================================================================

print("\n" + "=" * 65)
print("TASK 1 -- Loading data")
print("=" * 65)

df = pd.read_excel(DATA_FILE, sheet_name="Main Hard Copy")

print(f"\n  Loaded {len(df)} events - {df.shape[1]} columns")
print(f"  Columns: {df.columns.tolist()}\n")


# =============================================================================
# TASK 2 -- VARIABLE CONSTRUCTION
# =============================================================================

print("=" * 65)
print("TASK 2 -- Constructing variables")
print("=" * 65)

# The Excel file already contains Surprise_EPS and Disagreement_Gap as
# pre-calculated columns from Bloomberg. We verify they are present and
# construct only the interaction term needed for the regression.
#
# If either column is missing (e.g. in a future dataset where they are not
# pre-calculated), we fall back to computing them from the raw EPS columns.

# -- Surprise_EPS  (Actual EPS - Analyst Consensus EPS) -----------------------
# Positive = beat, Negative = miss.
# TODO (later): scale by pre-announcement stock price for cross-firm comparability.
if "Surprise_EPS" not in df.columns or df["Surprise_EPS"].isna().all():
    print("  'Surprise_EPS' not found -- computing from raw EPS columns.")
    df["Surprise_EPS"] = df["Actual_EPS"] - df["Analyst_Consensus_EPS"]
else:
    print("  'Surprise_EPS' loaded from Excel.")

# -- Disagreement_Gap  (Mgmt Guidance EPS - Analyst Consensus EPS) ------------
# Positive = management more optimistic than analysts.
# Negative = management more pessimistic than analysts.
# Large absolute value = high informational complexity -> expected stronger drift.
# TODO (later): scale by pre-announcement stock price.
if "Disagreement_Gap" not in df.columns or df["Disagreement_Gap"].isna().all():
    print("  'Disagreement_Gap' not found -- computing from raw EPS columns.")
    df["Disagreement_Gap"] = df["Mgmt_Guidance_EPS"] - df["Analyst_Consensus_EPS"]
else:
    print("  'Disagreement_Gap' loaded from Excel.")

# -- Interaction term: Surprise_EPS x Disagreement_Gap ------------------------
# b3 on this term tests the core hypothesis: does a wider disagreement gap
# amplify the relationship between the earnings surprise and subsequent drift?
df["Surprise_x_Gap"] = df["Surprise_EPS"] * df["Disagreement_Gap"]
print("  'Surprise_x_Gap' (interaction term) computed.")

# -- Print variable overview --------------------------------------------------
display_cols = [
    "Ticker", "Event_ID",
    "Actual_EPS", "Analyst_Consensus_EPS", "Mgmt_Guidance_EPS",
    "Surprise_EPS", "Disagreement_Gap", "Surprise_x_Gap",
    "CAR_0_1", "CAR_2_60",
]
display_cols = [c for c in display_cols if c in df.columns]

print("\nAll events -- key variables:")
print(df[display_cols].to_string(index=False))

key_vars = ["Surprise_EPS", "Disagreement_Gap", "Surprise_x_Gap",
            "CAR_0_1", "CAR_2_60"]
key_vars = [c for c in key_vars if c in df.columns]

print("\nDescriptive statistics:")
print(df[key_vars].describe().round(4).to_string())


# =============================================================================
# TASK 3 -- OLS REGRESSION
# =============================================================================

print("\n" + "=" * 65)
print("TASK 3 -- OLS Regression")
print("  Dependent variable : CAR_2_60 (post-announcement drift, pp)")
print("  Regressors         : Surprise_EPS")
print("                       Disagreement_Gap")
print("                       Surprise_EPS x Disagreement_Gap  (interaction)")
print("  Standard errors    : HC3 (heteroskedasticity-robust)")
print("=" * 65)

DEPENDENT  = "CAR_2_60"
REGRESSORS = ["Surprise_EPS", "Disagreement_Gap", "Surprise_x_Gap"]

# Drop rows where any required variable is missing (listwise deletion)
reg_df = df[[DEPENDENT] + REGRESSORS].dropna()
n_obs  = len(reg_df)

print(f"\n  Full sample  : {len(df)} events")
print(f"  Regression N : {n_obs} observations (after removing rows with missing values)")

if n_obs < len(REGRESSORS) + 2:
    print(
        "\n  WARNING: Not enough complete observations to run the regression.\n"
        "  Please check that all required columns contain data.\n"
    )
else:
    y = reg_df[DEPENDENT]

    # sm.add_constant() prepends a column of 1s (the intercept alpha).
    X = sm.add_constant(reg_df[REGRESSORS])

    # HC3 robust standard errors -- recommended for cross-sectional financial
    # data, especially in small samples. Inflates SEs for high-leverage obs,
    # making inference more conservative.
    result = sm.OLS(y, X).fit(cov_type="HC3")

    # -- Full statsmodels summary ----------------------------------------------
    print("\n")
    print(result.summary())

    # -- Save coefficient table ------------------------------------------------
    coef_table = pd.DataFrame({
        "Coefficient" : result.params,
        "Std_Error"   : result.bse,
        "t_stat"      : result.tvalues,
        "p_value"     : result.pvalues,
        "CI_lower_95" : result.conf_int()[0],
        "CI_upper_95" : result.conf_int()[1],
    })
    out_path = RESULTS_DIR / "regression_results.csv"
    coef_table.to_csv(out_path)
    print(f"\n  Coefficient table saved to: {out_path.relative_to(ROOT)}")

    # -- Quick-read summary ----------------------------------------------------
    print()
    print("  --- Key results ---------------------------------------------------")
    for var in ["const"] + REGRESSORS:
        coef  = result.params[var]
        pval  = result.pvalues[var]
        se    = result.bse[var]
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(
            f"  {var:<32}  b = {coef:+.4f}  SE = {se:.4f}"
            f"  p = {pval:.3f}  {stars}"
        )
    print(f"\n  R2 = {result.rsquared:.4f}   Adj. R2 = {result.rsquared_adj:.4f}"
          f"   N = {int(result.nobs)}")
    print("  -------------------------------------------------------------------")
    print()
    print("  Significance: *** p<0.01  ** p<0.05  * p<0.10")
    print()
    print("  NOTE: Results are from the TEST dataset -- interpret with caution")
    print("        until the full dataset is in place.")
    print()
