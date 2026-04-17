# v2/analysis_v2.py

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).parent.parent
COMBINED_DATA_FILE = ROOT / "data" / "combined" / "all_relevant_data.csv"
OUT_DIR = ROOT / "v2" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 65)
    print("v2 Analysis - Guidance-Based Surprise Methodology")
    print("=" * 65)

    if not COMBINED_DATA_FILE.exists():
        sys.exit(f"Data file not found: {COMBINED_DATA_FILE}")
        
    df = pd.read_csv(COMBINED_DATA_FILE)
    print(f"Loaded {len(df)} observations.")

    all_main_results = []

    # 1. Variable Construction
    # The user specifies that Guidance Surprise must be scaled by the stock price 5 days 
    # before the announcement to prevent the variable from being dominated by high-priced stocks.
    # We recover that historical Price_t-5 using the pre-scaled Bloomberg columns.
    df["Stock_Price_t5"] = df["Actual_EPS"] / df["Std_Actual_EPS"]
    
    # Construct explicitly: (Actual EPS - Mgmt Guidance EPS) / Price_t-5
    df["Guidance_Surprise_Raw"] = (df["Actual_EPS"] - df["Mgmt_Guidance_EPS"]) / df["Stock_Price_t5"]

    # 2. Winsorization at 1st and 99th percentiles
    def winsorize_col(series, limits=(0.01, 0.99)):
        lower = series.quantile(limits[0])
        upper = series.quantile(limits[1])
        return series.clip(lower=lower, upper=upper)

    df["Guidance_Surprise_Win"] = winsorize_col(df["Guidance_Surprise_Raw"])
    df["Std_Disagreement_Gap_Win"] = winsorize_col(df["Std_Disagreement_Gap"])
    df["Std_Surprise_EPS_Win"] = winsorize_col(df["Std_Surprise_EPS"])

    # 3. Z-score standardizations
    df["Z_Guidance_Surprise"] = (df["Guidance_Surprise_Win"] - df["Guidance_Surprise_Win"].mean()) / df["Guidance_Surprise_Win"].std()
    df["Z_Disagreement_Gap"] = (df["Std_Disagreement_Gap_Win"] - df["Std_Disagreement_Gap_Win"].mean()) / df["Std_Disagreement_Gap_Win"].std()

    # Log Market Cap
    df["Log_Market_Cap"] = np.log(df["Market_Cap"].replace(0, np.nan))
    df["Std_Log_Market_Cap"] = (df["Log_Market_Cap"] - df["Log_Market_Cap"].mean()) / df["Log_Market_Cap"].std()

    # Interaction
    df["Z_Interaction"] = df["Z_Guidance_Surprise"] * df["Z_Disagreement_Gap"]

    # Asymmetry & Conditional variables
    df["Negative_Gap_Dummy"] = (df["Std_Disagreement_Gap_Win"] < 0).astype(int)
    df["Int_Gap_Neg"] = df["Z_Disagreement_Gap"] * df["Negative_Gap_Dummy"]
    df["Beat_Dummy"] = (df["EPS_Beat_Miss"] == "Beat").astype(int)

    # Sector Fixed Effects
    sector_dummies = pd.get_dummies(df["Sector"], prefix="Sector", drop_first=True, dtype=int)
    df = pd.concat([df, sector_dummies], axis=1)
    sector_cols = sector_dummies.columns.tolist()

    # Base controls
    controls = ["Std_Log_Market_Cap"] + sector_cols
    
    # =========================================================================
    # Descriptive Statistics
    # =========================================================================
    desc_vars = [
        "Guidance_Surprise_Raw", "Std_Disagreement_Gap", "Log_Market_Cap", 
        "Negative_Gap_Dummy", "CAR_0_1", "CAR_2_15"
    ]
    print("\n" + "=" * 65 + "\nDESCRIPTIVE STATISTICS\n" + "=" * 65)
    desc_stats = df[desc_vars].describe().round(4)
    print(desc_stats.to_string())
    print()
    desc_stats.to_csv(OUT_DIR / "descriptive_statistics.csv")
    print(f">> Saved descriptive_statistics.csv to {OUT_DIR}")
    
    def run_regression(dep_var, indep_vars, df_reg, model_name=""):
        req_cols = ["Ticker", dep_var] + indep_vars
        df_clean = df_reg[req_cols].dropna()
        
        y = df_clean[dep_var]
        X = sm.add_constant(df_clean[indep_vars])
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df_clean["Ticker"]})
        
        # Store key statistics for printing and returning
        res_info = {
            "Model": model_name,
            "Dep_Var": dep_var,
            "N_Obs": int(res.nobs),
            "R2": res.rsquared,
            "Adj_R2": res.rsquared_adj,
            "Result": res
        }
        return res_info

    def print_and_save_block(models, out_file):
        summary_data = [] # For CSV output
        
        for m in models:
            print(f"\n--- {m['Model']} (Dep: {m['Dep_Var']}) ---")
            m_res = m['Result']
            
            # Print core metrics
            for var in ["const"] + list(m_res.params.index)[1:]:
                # Only print the coefficients that aren't sector fixed effects for brevity in console
                if var.startswith("Sector_"): continue
                
                coef = m_res.params[var]
                se = m_res.bse[var]
                pval = m_res.pvalues[var]
                stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                
                print(f"{var:<32} b={coef:+.4f} (SE={se:.4f})  p={pval:.3f} {stars}")
                
            print(f"R2 = {m['R2']:.4f}  Adj.R2 = {m['Adj_R2']:.4f}  N = {m['N_Obs']}")
            
            # Build structured data for CSV
            for var in m_res.params.index:
                summary_data.append({
                    "Model": m["Model"],
                    "Dep_Var": m["Dep_Var"],
                    "Variable": var,
                    "Coefficient": m_res.params[var],
                    "Std_Error": m_res.bse[var],
                    "t_stat": m_res.tvalues[var],
                    "p_value": m_res.pvalues[var],
                    "Significance": "***" if m_res.pvalues[var] < 0.01 else "**" if m_res.pvalues[var] < 0.05 else "*" if m_res.pvalues[var] < 0.10 else "",
                    "R_Squared": m["R2"],
                    "Adj_R_Squared": m["Adj_R2"],
                    "Observations": m["N_Obs"]
                })
                
        for item in summary_data:
            item["Block/File"] = out_file
            all_main_results.append(item)
        pass

    # =========================================================================
    # BLOCK 1: PEAD Regressions (Short Windows)
    # =========================================================================
    print("\n" + "=" * 65 + "\nBLOCK 1: Small Windows (CAR_2_3, CAR_2_5, CAR_2_7, CAR_2_10, CAR_2_15)" + "\n" + "=" * 65)
    short_models = [
        run_regression("CAR_2_3", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df, "Model: Key Spec with CAR_2_3"),
        run_regression("CAR_2_5", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df, "Model: Key Spec with CAR_2_5"),
        run_regression("CAR_2_7", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df, "Model: Key Spec with CAR_2_7"),
        run_regression("CAR_2_10", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df, "Model: Key Spec with CAR_2_10"),
        run_regression("CAR_2_15", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df, "Model: Key Spec with CAR_2_15")
    ]
    print_and_save_block(short_models, "short_window_pead_results.csv")
    
    # Empty out pead_models since we are not running 2_60
    pead_models = []

    # =========================================================================
    # BLOCK 2: Immediate Reaction (CAR_0_1)
    # =========================================================================
    print("\n" + "=" * 65 + "\nBLOCK 2: Immediate Reaction (CAR_0_1)" + "\n" + "=" * 65)
    imm_models = [
        run_regression("CAR_0_1", ["Z_Guidance_Surprise"] + controls, df, "Model 1: Guidance Surprise Only"),
        run_regression("CAR_0_1", ["Z_Disagreement_Gap"] + controls, df, "Model 2: Pure Predictive Gap"),
        run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df, "Model 3: Key Specification"),
        run_regression("CAR_0_1", ["Std_Surprise_EPS_Win", "Z_Disagreement_Gap"] + controls, df, "Model 4: Diagnostic (Old Surprise)"),
        run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap", "Z_Interaction"] + controls, df, "Model 5: Interaction")
    ]
    print_and_save_block(imm_models, "immediate_reaction_regression_results.csv")

    # =========================================================================
    # BLOCK 3: Quartile Analysis on Std_Disagreement_Gap
    # =========================================================================
    print("\n" + "=" * 65 + "\nBLOCK 3: Quartile Analysis (CAR_0_1 over Std_Disagreement_Gap)" + "\n" + "=" * 65)
    # Remove NaNs first
    q_df = df[["CAR_0_1", "Z_Disagreement_Gap"]].dropna().copy()
    q_df["Gap_Q"] = pd.qcut(q_df["Z_Disagreement_Gap"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    
    q_stats = q_df.groupby("Gap_Q")["CAR_0_1"].agg(['mean', 'median', 'count']).reset_index()
    q_stats.rename(columns={"mean": "Mean_CAR_0_1", "median": "Median_CAR_0_1", "count": "N"}, inplace=True)
    
    q1_vals = q_df[q_df["Gap_Q"] == "Q1"]["CAR_0_1"]
    q4_vals = q_df[q_df["Gap_Q"] == "Q4"]["CAR_0_1"]
    
    t_stat, p_val_t = stats.ttest_ind(q4_vals, q1_vals)
    spread = q4_vals.mean() - q1_vals.mean()
    
    f_stat, p_val_f = stats.f_oneway(
        q_df[q_df["Gap_Q"] == "Q1"]["CAR_0_1"],
        q_df[q_df["Gap_Q"] == "Q2"]["CAR_0_1"],
        q_df[q_df["Gap_Q"] == "Q3"]["CAR_0_1"],
        q_df[q_df["Gap_Q"] == "Q4"]["CAR_0_1"]
    )
    
    print("\nQuartile Means and Medians:")
    print(q_stats)
    
    print(f"\nSpread (Q4 - Q1): {spread:+.4f}")
    stars_t = "***" if p_val_t < 0.01 else "**" if p_val_t < 0.05 else "*" if p_val_t < 0.10 else ""
    print(f"T-test (Q4 vs Q1): t = {t_stat:.4f}, p = {p_val_t:.4f} {stars_t}")
    
    stars_f = "***" if p_val_f < 0.01 else "**" if p_val_f < 0.05 else "*" if p_val_f < 0.10 else ""
    print(f"One-way ANOVA (all quartiles): F = {f_stat:.4f}, p = {p_val_f:.4f} {stars_f}")
    
    # Save to CSV
    with open(OUT_DIR / "quartile_analysis_results.csv", "w") as f:
        f.write("Quartile,Mean_CAR_0_1,Median_CAR_0_1,N\n")
        for _, row in q_stats.iterrows():
            f.write(f"{row['Gap_Q']},{row['Mean_CAR_0_1']},{row['Median_CAR_0_1']},{row['N']}\n")
        f.write("\n")
        f.write(f"Spread (Q4 - Q1),{spread}\n")
        f.write(f"T-test (Q4 vs Q1) t-stat,{t_stat}\n")
        f.write(f"T-test (Q4 vs Q1) p-value,{p_val_t}\n")
        f.write(f"ANOVA F-stat,{f_stat}\n")
        f.write(f"ANOVA p-value,{p_val_f}\n")
    print(f">> Saved quartile_analysis_results.csv to {OUT_DIR}")

    # =========================================================================
    # BLOCK 4: Robustness (BHAR_2_60)
    # =========================================================================
    print("\n" + "=" * 65 + "\nBLOCK 4: Robustness (BHAR_2_60)" + "\n" + "=" * 65)
    robust_models = [
        run_regression("BHAR_2_60", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df, "Model: Key Specification with BHAR_2_60")
    ]
    print_and_save_block(robust_models, "bhar_robustness_results.csv")


    # =========================================================================
    # BLOCK 5: Asymmetry Analysis (Negative vs Positive Gaps)
    # =========================================================================
    print("\n" + "=" * 65 + "\nBLOCK 5: Asymmetry Analysis (Negative vs Positive Gaps)\n" + "=" * 65)
    asymmetry_models = [
        run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap", "Negative_Gap_Dummy", "Int_Gap_Neg"] + controls, df, "Model 1: Asymmetry (No Beat Dummy)"),
        run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap", "Negative_Gap_Dummy", "Int_Gap_Neg", "Beat_Dummy"] + controls, df, "Model 2: Asymmetry (With Beat Dummy)")
    ]
    print_and_save_block(asymmetry_models, "asymmetry_results.csv")

    # =========================================================================
    # BLOCK 6: Conditional Analysis (Beats vs Misses)
    # =========================================================================
    print("\n" + "=" * 65 + "\nBLOCK 6: Conditional Analysis (Beats vs Misses)\n" + "=" * 65)
    print("\n  [!] WARNING: EPS_Beat_Miss is revealed SIMULTANEOUSLY with CAR_0_1.")
    print("      This test should be interpreted as a decomposition of the reaction, NOT a predictive test.\n")
    
    df_beats = df[df["EPS_Beat_Miss"] == "Beat"].copy()
    df_misses = df[df["EPS_Beat_Miss"] == "Miss"].copy()
    
    beats_model = run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df_beats, "Model: BEATS Only")
    misses_model = run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df_misses, "Model: MISSES Only")
    
    conditional_models = [beats_model, misses_model]
    print_and_save_block(conditional_models, "beat_miss_conditional_results.csv")
    
    print("\n  --- Side-by-Side Comparison (Z_Disagreement_Gap) ---")
    b_res = beats_model["Result"]
    m_res = misses_model["Result"]
    b_coef, b_pval = b_res.params["Z_Disagreement_Gap"], b_res.pvalues["Z_Disagreement_Gap"]
    m_coef, m_pval = m_res.params["Z_Disagreement_Gap"], m_res.pvalues["Z_Disagreement_Gap"]
    b_stars = "***" if b_pval < 0.01 else "**" if b_pval < 0.05 else "*" if b_pval < 0.10 else ""
    m_stars = "***" if m_pval < 0.01 else "**" if m_pval < 0.05 else "*" if m_pval < 0.10 else ""
    
    print(f"  BEATS Subsample : b = {b_coef:+.4f}  (p={b_pval:.3f}) {b_stars}  [N={beats_model['N_Obs']}]")
    print(f"  MISSES Subsample: b = {m_coef:+.4f}  (p={m_pval:.3f}) {m_stars}  [N={misses_model['N_Obs']}]")
    print("  ----------------------------------------------------------\n")

    # =========================================================================
    # Final Model Summary
    # =========================================================================
    print("\n" + "=" * 65 + "\nSUMMARY OF SIGNIFICANCE\n" + "=" * 65)
    
    all_models = short_models + pead_models + imm_models + robust_models + asymmetry_models + conditional_models
    for m in all_models:
        res = m["Result"]
        print(f"{m['Model']} (Dep: {m['Dep_Var']}):")
        
        gap_sig, surp_sig = "-", "-"
        if "Z_Disagreement_Gap" in res.params.index:
            pval = res.pvalues["Z_Disagreement_Gap"]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else "NS"
            gap_sig = f"{'✅ Sig' if pval < 0.10 else '❌ NS'} ({stars})"
        
        if "Z_Guidance_Surprise" in res.params.index:
            pval = res.pvalues["Z_Guidance_Surprise"]
            stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else "NS"
            surp_sig = f"{'✅ Sig' if pval < 0.10 else '❌ NS'} ({stars})"
            
        print(f"  Z_Guidance_Surprise   : {surp_sig}")
        print(f"  Std_Disagreement_Gap  : {gap_sig}")
    print()


    pd.DataFrame(all_main_results).to_csv(OUT_DIR / "all_analysis_results.csv", index=False)
    print(f"\n>> Successfully combined all results into all_analysis_results.csv")

if __name__ == "__main__":
    main()
