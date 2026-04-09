"""
low_market_cap_analysis.py
==========================
Post-Earnings Announcement Drift (PEAD)  Subgroup Analysis Script

Hypothesis:
    A larger disagreement gap (Management Guidance EPS - Analyst Consensus EPS)
    creates informational complexity, causing an initial underreaction on the
    announcement day and a stronger PEAD over the following 60 trading days.
    This analysis restricts the sample to a specific subgroup of companies 
    with relatively low market capitalizations.

How to run:
    python experiments/low_market_cap_analysis.py

Output:
    Console  -- descriptive statistics and full regression summary
    experiments/results/low_cap_regression_results.csv  -- coefficient table
"""

# -----------------------------------------------------------------------------
# Imports & paths
# -----------------------------------------------------------------------------

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Root is one level up from 'experiments' folder
ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = ROOT / "data" / "raw"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

data_files = list(RAW_DATA_DIR.glob("Data til Advanced Finance seminar v*.xlsx"))

if not data_files:
    sys.exit(
        f"\nERROR: No data files found in:\n  {RAW_DATA_DIR}\n\n"
        "Please place the 'Data til Advanced Finance seminar vX.xlsx' files in the data/raw/ folder and re-run.\n"
    )

# List of lower market cap companies to filter by
LOW_CAP_TICKERS = [
    "SLB", "EOG", "GM", "CL", "FCX", "WELL", "PNC", "NOC", "GD", "WM", "NSC", 
    "CMG", "ABNB", "MDLZ", "KMB", "OXY", "MPC", "EXC", "CHTR", "SHW", "NEM", 
    "SPG", "USB", "ICE", "MMC", "ADI", "LRCX", "SNPS", "ITW", "CSX", "PH", 
    "RCL", "HLT", "DHI", "GIS", "STZ", "DVN", "HES", "AEP", "SRE", "APD", 
    "ECL", "O", "MCK", "DXCM", "IQV", "TFC", "MET", "FTNT", "DELL", "MMM", 
    "EMR", "CTAS", "YUM", "EBAY", "PHM", "SYY", "KHC", "HAL", "BKR", "D", 
    "EA", "DOW", "PPG", "PSA", "CBRE", "HUM", "BDX", "BIIB", "EW", "PRU", 
    "AFL", "ALL", "NXPI", "HPQ", "ROK", "CARR", "OTIS", "AZO", "ORLY", "LVS", 
    "HSY", "ADM", "FANG", "PCG", "ES", "FOXA", "NUE", "VMC", "EQR", "IDXX", 
    "A", "STT", "BK", "CTSH", "ROP", "GWW", "PCAR", "ROST", "TSCO", "LEN", 
    "KR", "MNST", "KMI", "OKE", "WEC", "ETR", "WBD", "LYV", "LYB", "CF", 
    "DLR", "RMD", "HOLX", "AIG", "HIG", "WDAY", "MPWR", "GLW", "IR", "FAST", 
    "DRI", "CCL", "EXPE", "EL", "CHD", "CTRA", "XEL", "AWK", "OMC", "ALB", 
    "IP", "VTR", "CCI"
]


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

# Important: Filter the dataset using our subset of companies
# The tickers in the dataset have suffixes (e.g., "LEN US"). We extract just the base ticker.
original_len = len(df)
base_tickers = df['Ticker'].astype(str).str.split().str[0]
df = df[base_tickers.isin(LOW_CAP_TICKERS)].copy()

print(f"\n  Loaded {original_len} total events.")
print(f"  Filtered down to {len(df)} events for selected low market cap companies.")
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

# -- Interaction term: Z_Surprise x Z_Gap -------------------------------------
df["Z_Interaction"] = df["Z_Surprise"] * df["Z_Gap"]
print("  'Z_Interaction' (interaction term) computed.")

# -- Control Variables: Market Cap and Sector ---------------------------------
if "Market_Cap" in df.columns:
    log_mc = np.log(df["Market_Cap"].replace(0, np.nan))
    df["Std_Log_Market_Cap"] = (log_mc - log_mc.mean()) / log_mc.std()
    print("  'Std_Log_Market_Cap' computed.")
else:
    df["Std_Log_Market_Cap"] = np.nan

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

reg_df = df[["Ticker", DEPENDENT] + REGRESSORS].dropna()
n_obs  = len(reg_df)

print(f"\n  Filtered sample : {len(df)} events")
print(f"  Regression N    : {n_obs} observations (after removing rows with missing values)")

if n_obs < len(REGRESSORS) + 2:
    print(
        "\n  WARNING: Not enough complete observations to run the regression.\n"
        "  Please check that all required columns contain data.\n"
    )
else:
    y = reg_df[DEPENDENT]
    X = sm.add_constant(reg_df[REGRESSORS])

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
    out_path = RESULTS_DIR / "low_cap_regression_results.csv"
    coef_table.to_csv(out_path)
    print(f"\n  Coefficient table saved to: experiments/results/low_cap_regression_results.csv")

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
    print("  NOTE: Results are based on the Low Market Cap Subgroup.")
    print()