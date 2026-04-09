"""
analysis_extended.py
====================
Post-Earnings Announcement Drift (PEAD) -- Extended Analysis Script

Purpose:
    Extends the baseline PEAD regressions with additional specifications to test
    whether disagreement between management guidance and analyst consensus
    affects post-announcement drift more strongly in nonlinear, conditional, or
    alternative return specifications.

Main extensions:
    1. Baseline regressions on CAR_2_60
    2. Alternative dependent variable: BHAR_2_60
    3. Immediate reaction test: CAR_0_1
    4. Absolute disagreement measure
    5. High-vs-low disagreement interaction
    6. Subsample regressions by disagreement level
    7. Winsorized specifications to reduce outlier influence

Data files:
    data/raw/Data til Advanced Finance seminar v*.xlsx
    sheet: "Main Hard Copy"

How to run:
    python analysis_extended.py

Outputs:
    Console -- descriptive statistics and regression summaries
    results/extended_regression_results.csv -- combined coefficient table
"""

# -----------------------------------------------------------------------------
# Imports & paths
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT         = Path(__file__).parent
RAW_DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR  = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data_files = list(RAW_DATA_DIR.glob("Data til Advanced Finance seminar v*.xlsx"))

if not data_files:
    sys.exit(
        f"\nERROR: No data files found in:\n  {RAW_DATA_DIR}\n\n"
        "Please place the 'Data til Advanced Finance seminar vX.xlsx' files in the data/raw/ folder and re-run.\n"
    )


# -----------------------------------------------------------------------------
# Run controls
# -----------------------------------------------------------------------------

RUN_MAIN_MODELS = True
RUN_WINSORIZED  = False
RUN_BHAR        = False
RUN_IMMEDIATE   = False
RUN_SUBSAMPLES  = False

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

# -- Std_Surprise_EPS ---------------------------------------------------------
if "Std_Surprise_EPS" not in df.columns or df["Std_Surprise_EPS"].isna().all():
    print("  'Std_Surprise_EPS' not found -- falling back to raw computation.")
    df["Std_Surprise_EPS"] = df["Actual_EPS"] - df["Analyst_Consensus_EPS"]
else:
    print("  'Std_Surprise_EPS' loaded from Excel.")

# -- Std_Disagreement_Gap -----------------------------------------------------
if "Std_Disagreement_Gap" not in df.columns or df["Std_Disagreement_Gap"].isna().all():
    print("  'Std_Disagreement_Gap' not found -- falling back to raw computation.")
    df["Std_Disagreement_Gap"] = df["Mgmt_Guidance_EPS"] - df["Analyst_Consensus_EPS"]
else:
    print("  'Std_Disagreement_Gap' loaded from Excel.")

# -- Z-score variables --------------------------------------------------------
df["Z_Surprise"] = (
    (df["Std_Surprise_EPS"] - df["Std_Surprise_EPS"].mean())
    / df["Std_Surprise_EPS"].std()
)
df["Z_Gap"] = (
    (df["Std_Disagreement_Gap"] - df["Std_Disagreement_Gap"].mean())
    / df["Std_Disagreement_Gap"].std()
)

# -- Main interaction ---------------------------------------------------------
df["Z_Interaction"] = df["Z_Surprise"] * df["Z_Gap"]
print("  'Z_Interaction' (signed interaction) computed.")

# -- Absolute disagreement ----------------------------------------------------
df["Abs_Z_Gap"] = df["Z_Gap"].abs()
df["Z_Int_AbsGap"] = df["Z_Surprise"] * df["Abs_Z_Gap"]
print("  'Abs_Z_Gap' and 'Z_Int_AbsGap' computed.")

# -- High disagreement dummy --------------------------------------------------
median_abs_gap = df["Abs_Z_Gap"].median()
df["High_Abs_Gap"] = (df["Abs_Z_Gap"] > median_abs_gap).astype(int)
df["Z_Int_HighGap"] = df["Z_Surprise"] * df["High_Abs_Gap"]
print("  'High_Abs_Gap' and 'Z_Int_HighGap' computed.")

# -- Winsorized variables -----------------------------------------------------
def winsorize_series(s: pd.Series, lower_q: float = 0.01, upper_q: float = 0.99) -> pd.Series:
    lower = s.quantile(lower_q)
    upper = s.quantile(upper_q)
    return s.clip(lower, upper)

df["W_Z_Surprise"] = winsorize_series(df["Z_Surprise"])
df["W_Z_Gap"] = winsorize_series(df["Z_Gap"])
df["W_Z_Interaction"] = df["W_Z_Surprise"] * df["W_Z_Gap"]
df["W_Abs_Z_Gap"] = df["W_Z_Gap"].abs()
df["W_Z_Int_AbsGap"] = df["W_Z_Surprise"] * df["W_Abs_Z_Gap"]
print("  Winsorized variables computed (1st and 99th percentiles).")

# -- Market cap control -------------------------------------------------------
if "Market_Cap" in df.columns:
    log_mc = np.log(df["Market_Cap"].replace(0, np.nan))
    df["Std_Log_Market_Cap"] = (log_mc - log_mc.mean()) / log_mc.std()
    print("  'Std_Log_Market_Cap' computed.")
else:
    df["Std_Log_Market_Cap"] = np.nan

# -- Sector fixed effects -----------------------------------------------------
sector_cols = []
if "Sector" in df.columns:
    sector_dummies = pd.get_dummies(df["Sector"], prefix="Sector", drop_first=True, dtype=int)
    df = pd.concat([df, sector_dummies], axis=1)
    sector_cols = sector_dummies.columns.tolist()
    print(f"  {len(sector_cols)} Sector dummy variables created.")

# -- Preview ------------------------------------------------------------------
display_cols = [
    "Ticker", "Event_ID",
    "Actual_EPS", "Analyst_Consensus_EPS", "Mgmt_Guidance_EPS",
    "Z_Surprise", "Z_Gap", "Abs_Z_Gap", "Z_Interaction", "Z_Int_AbsGap",
    "Std_Log_Market_Cap", "Sector", "CAR_0_1", "CAR_2_60", "BHAR_2_60"
]
display_cols = [c for c in display_cols if c in df.columns]

print("\nPreview of key variables (first 10 rows):")
print(df[display_cols].head(10).to_string(index=False))

key_vars = [
    "Z_Surprise", "Z_Gap", "Abs_Z_Gap", "Z_Interaction", "Z_Int_AbsGap",
    "W_Z_Surprise", "W_Z_Gap", "W_Z_Interaction",
    "Std_Log_Market_Cap", "CAR_0_1", "CAR_2_60", "BHAR_2_60"
]
key_vars = [c for c in key_vars if c in df.columns]

print("\nDescriptive statistics:")
print(df[key_vars].describe().round(4).to_string())


# =============================================================================
# TASK 3 -- REGRESSION HELPER
# =============================================================================

def run_regression(
    data: pd.DataFrame,
    dependent: str,
    regressors: list,
    model_name: str,
    sample_name: str = "Full sample"
):
    """
    Run OLS regression with firm-clustered standard errors and print results.
    Returns:
        result, coef_table
    """
    print("\n" + "-" * 65)
    print(model_name)
    print(f"  Sample             : {sample_name}")
    print(f"  Dependent variable : {dependent}")
    print("  Regressors         :")
    for reg in regressors:
        print(f"                       {reg}")
    print("-" * 65)

    needed_cols = ["Ticker", dependent] + regressors
    reg_df = data[needed_cols].dropna()
    n_obs = len(reg_df)

    print(f"\n  Regression N : {n_obs} observations (after removing rows with missing values)")

    if n_obs < len(regressors) + 2:
        print(
            "\n  WARNING: Not enough complete observations to run the regression.\n"
            "  Please check that all required columns contain data.\n"
        )
        return None, None

    y = reg_df[dependent]
    X = sm.add_constant(reg_df[regressors])

    result = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": reg_df["Ticker"]})

    print("\n")
    print(result.summary())

    print()
    print("  --- Key results ---------------------------------------------------")
    for var in result.params.index:
        coef  = result.params[var]
        pval  = result.pvalues[var]
        se    = result.bse[var]
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
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

    coef_table = pd.DataFrame({
        "Model"       : model_name,
        "Sample"      : sample_name,
        "Dependent"   : dependent,
        "Variable"    : result.params.index,
        "Coefficient" : result.params.values,
        "Std_Error"   : result.bse.values,
        "t_stat"      : result.tvalues.values,
        "p_value"     : result.pvalues.values,
        "CI_lower_95" : result.conf_int()[0].values,
        "CI_upper_95" : result.conf_int()[1].values,
        "N"           : int(result.nobs),
        "R2"          : result.rsquared,
        "Adj_R2"      : result.rsquared_adj,
    })

    return result, coef_table


# =============================================================================
# TASK 4 -- MAIN REGRESSIONS: CAR(2,60)
# =============================================================================

all_results = []

if RUN_MAIN_MODELS:
    print("\n" + "=" * 65)
    print("TASK 4 -- Main Regressions: CAR_2_60")
    print("  Standard errors    : Clustered by Firm (Ticker)")
    print("=" * 65)

    main_specs = [
        ("Model 1 -- Baseline PEAD", "CAR_2_60", ["Z_Surprise"]),
        ("Model 2 -- Add Signed Gap", "CAR_2_60", ["Z_Surprise", "Z_Gap"]),
        ("Model 3 -- Full Signed Model", "CAR_2_60",
         ["Z_Surprise", "Z_Gap", "Z_Interaction", "Std_Log_Market_Cap"] + sector_cols),
        ("Model 4 -- Absolute Gap", "CAR_2_60", ["Z_Surprise", "Abs_Z_Gap"]),
        ("Model 5 -- Absolute Gap Interaction", "CAR_2_60",
         ["Z_Surprise", "Abs_Z_Gap", "Z_Int_AbsGap", "Std_Log_Market_Cap"] + sector_cols),
        ("Model 6 -- High Disagreement Interaction", "CAR_2_60",
         ["Z_Surprise", "High_Abs_Gap", "Z_Int_HighGap", "Std_Log_Market_Cap"] + sector_cols),
    ]

    for model_name, dep, regs in main_specs:
        _, coef_table = run_regression(df, dep, regs, model_name)
        if coef_table is not None:
            all_results.append(coef_table)


# =============================================================================
# TASK 5 -- ROBUSTNESS: WINSORIZED VARIABLES
# =============================================================================

if RUN_WINSORIZED:
    print("\n" + "=" * 65)
    print("TASK 5 -- Robustness Regressions: Winsorized Variables")
    print("=" * 65)

    winsor_specs = [
        ("Winsorized Model 1 -- Signed Interaction", "CAR_2_60",
         ["W_Z_Surprise", "W_Z_Gap", "W_Z_Interaction", "Std_Log_Market_Cap"] + sector_cols),
        ("Winsorized Model 2 -- Absolute Interaction", "CAR_2_60",
         ["W_Z_Surprise", "W_Abs_Z_Gap", "W_Z_Int_AbsGap", "Std_Log_Market_Cap"] + sector_cols),
    ]

    for model_name, dep, regs in winsor_specs:
        _, coef_table = run_regression(df, dep, regs, model_name)
        if coef_table is not None:
            all_results.append(coef_table)


# =============================================================================
# TASK 6 -- ALTERNATIVE DEPENDENT VARIABLE: BHAR(2,60)
# =============================================================================

if RUN_BHAR and "BHAR_2_60" in df.columns:
    print("\n" + "=" * 65)
    print("TASK 6 -- Alternative Dependent Variable: BHAR_2_60")
    print("=" * 65)

    bhar_specs = [
        ("BHAR Model 1 -- Baseline PEAD", "BHAR_2_60", ["Z_Surprise"]),
        ("BHAR Model 2 -- Signed Interaction", "BHAR_2_60",
         ["Z_Surprise", "Z_Gap", "Z_Interaction", "Std_Log_Market_Cap"] + sector_cols),
        ("BHAR Model 3 -- Absolute Interaction", "BHAR_2_60",
         ["Z_Surprise", "Abs_Z_Gap", "Z_Int_AbsGap", "Std_Log_Market_Cap"] + sector_cols),
    ]

    for model_name, dep, regs in bhar_specs:
        _, coef_table = run_regression(df, dep, regs, model_name)
        if coef_table is not None:
            all_results.append(coef_table)


# =============================================================================
# TASK 7 -- IMMEDIATE REACTION TEST: CAR(0,1)
# =============================================================================

if RUN_IMMEDIATE and "CAR_0_1" in df.columns:
    print("\n" + "=" * 65)
    print("TASK 7 -- Immediate Reaction Test: CAR_0_1")
    print("=" * 65)

    immediate_specs = [
        ("Immediate Reaction Model 1 -- Baseline", "CAR_0_1", ["Z_Surprise"]),
        ("Immediate Reaction Model 2 -- Signed Interaction", "CAR_0_1",
         ["Z_Surprise", "Z_Gap", "Z_Interaction", "Std_Log_Market_Cap"] + sector_cols),
        ("Immediate Reaction Model 3 -- Absolute Interaction", "CAR_0_1",
         ["Z_Surprise", "Abs_Z_Gap", "Z_Int_AbsGap", "Std_Log_Market_Cap"] + sector_cols),
    ]

    for model_name, dep, regs in immediate_specs:
        _, coef_table = run_regression(df, dep, regs, model_name)
        if coef_table is not None:
            all_results.append(coef_table)


# =============================================================================
# TASK 8 -- SUBSAMPLE REGRESSIONS: HIGH VS LOW DISAGREEMENT
# =============================================================================

if RUN_SUBSAMPLES:
    print("\n" + "=" * 65)
    print("TASK 8 -- Subsample Regressions: High vs Low Disagreement")
    print("=" * 65)

    high_gap_df = df[df["High_Abs_Gap"] == 1].copy()
    low_gap_df  = df[df["High_Abs_Gap"] == 0].copy()

    subsample_specs = [
        ("Subsample Model -- Baseline PEAD", "CAR_2_60", ["Z_Surprise"]),
        ("Subsample Model -- Full Signed", "CAR_2_60",
         ["Z_Surprise", "Z_Gap", "Z_Interaction", "Std_Log_Market_Cap"] + sector_cols),
    ]

    for model_name, dep, regs in subsample_specs:
        _, coef_table = run_regression(high_gap_df, dep, regs, model_name, sample_name="High disagreement")
        if coef_table is not None:
            all_results.append(coef_table)

        _, coef_table = run_regression(low_gap_df, dep, regs, model_name, sample_name="Low disagreement")
        if coef_table is not None:
            all_results.append(coef_table)


# =============================================================================
# TASK 9 -- SAVE ALL RESULTS
# =============================================================================

print("\n" + "=" * 65)
print("TASK 9 -- Saving Results")
print("=" * 65)

if all_results:
    final_results = pd.concat(all_results, ignore_index=True)
    out_path = RESULTS_DIR / "extended_regression_results.csv"
    final_results.to_csv(out_path, index=False)
    print(f"\n  Combined coefficient table saved to: {out_path.relative_to(ROOT)}")
else:
    print("\n  No regression results were generated.")

print("\n  NOTE: Extended analysis completed.")
print()