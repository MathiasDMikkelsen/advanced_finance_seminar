"""
analysis.py
===========
Post-Earnings Announcement Drift (PEAD) � Main Analysis Script

Hypothesis:
    A larger disagreement gap (Management Guidance EPS - Analyst Consensus EPS)
    creates informational complexity, causing an initial underreaction on the
    announcement day and a stronger PEAD over the following 60 trading days.

Main regression:
    CAR_2_60 = alpha + b1*Z_Surprise + b2*Z_Gap
             + b3*Z_Interaction
             + b4*Std_Log_Market_Cap + Sector_Fixed_Effects + e

Data file:  data/raw/phase2.xlsx  (sheet: "Main Hard Copy")
            All key variables (CARs, Std_Surprise_EPS, Std_Disagreement_Gap) are
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
DATA_FILE   = ROOT / "data" / "raw" / "phase2.xlsx"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    sys.exit(
        f"\nERROR: Data file not found at:\n  {DATA_FILE}\n\n"
        "Please place phase2.xlsx in the data/raw/ folder and re-run.\n"
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

# The Excel file already contains Std_Surprise_EPS and Std_Disagreement_Gap as
# pre-calculated, standardized columns from Bloomberg. We verify they are present
# and construct the standardized interaction term needed for the regression.
#
# If either column is missing, the script falls back to computing raw differences.
# (Note: for a true standardized fallback later, you'd divide by stock price).

# -- Std_Surprise_EPS  (Standardized Surprise) --------------------------------
# Positive = beat, Negative = miss. Comparable across firms.
if "Std_Surprise_EPS" not in df.columns or df["Std_Surprise_EPS"].isna().all():
    print("  'Std_Surprise_EPS' not found -- falling back to raw computation.")
    df["Std_Surprise_EPS"] = df["Actual_EPS"] - df["Analyst_Consensus_EPS"]
else:
    print("  'Std_Surprise_EPS' loaded from Excel.")

# -- Std_Disagreement_Gap  (Standardized Gap) ---------------------------------
# Positive = management more optimistic than analysts.
# Negative = management more pessimistic than analysts.
# Comparable across firms. Large absolute value = high informational complexity -> stronger drift.
if "Std_Disagreement_Gap" not in df.columns or df["Std_Disagreement_Gap"].isna().all():
    print("  'Std_Disagreement_Gap' not found -- falling back to raw computation.")
    df["Std_Disagreement_Gap"] = df["Mgmt_Guidance_EPS"] - df["Analyst_Consensus_EPS"]
else:
    print("  'Std_Disagreement_Gap' loaded from Excel.")

# -- Continuous Variable Centering and Standardization --------------------------
# Convert Surprise and Gap to Z-scores to reduce structural multicollinearity 
# and explicit fix scaling issues before creating the interaction term.
df["Z_Surprise"] = (df["Std_Surprise_EPS"] - df["Std_Surprise_EPS"].mean()) / df["Std_Surprise_EPS"].std()
df["Z_Gap"] = (df["Std_Disagreement_Gap"] - df["Std_Disagreement_Gap"].mean()) / df["Std_Disagreement_Gap"].std()

# -- Interaction term: Z_Surprise x Z_Gap -------------------------------------
df["Z_Interaction"] = df["Z_Surprise"] * df["Z_Gap"]
print("  'Z_Interaction' (interaction term) computed.")

# -- Control Variables: Market Cap and Sector ---------------------------------
# We use the natural log of Market Cap to account for size effects, and standardize it.
if "Market_Cap" in df.columns:
    log_mc = np.log(df["Market_Cap"].replace(0, np.nan))
    df["Std_Log_Market_Cap"] = (log_mc - log_mc.mean()) / log_mc.std()
    print("  'Std_Log_Market_Cap' computed.")
else:
    df["Std_Log_Market_Cap"] = np.nan

# We create dummy variables for the Sector column.
# drop_first=True avoids the dummy variable trap (perfect multicollinearity).
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
    "Z_Surprise", "Z_Gap", "Z_Interaction",
    "Std_Log_Market_Cap", "Sector",
    "CAR_0_1", "CAR_2_60",
]
display_cols = [c for c in display_cols if c in df.columns]

print("\nAll events -- key variables:")
print(df[display_cols].to_string(index=False))

key_vars = ["Z_Surprise", "Z_Gap", "Z_Interaction",
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
print("                       Z_Interaction (int.)")
print("                       Std_Log_Market_Cap")
print("                       Sector Fixed Effects")
print("  Standard errors    : Clustered by Firm (Ticker)")
print("=" * 65)

DEPENDENT  = "CAR_2_60"
REGRESSORS = ["Z_Surprise", "Z_Gap", "Z_Interaction", "Std_Log_Market_Cap"] + sector_cols

# Drop rows where any required variable (or our clustering variable) is missing (listwise deletion)
reg_df = df[["Ticker", DEPENDENT] + REGRESSORS].dropna()
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

    # Clustered standard errors at the firm level to account for the fact
    # that some observations belong to the same company across different events.
    result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": reg_df["Ticker"]})

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
