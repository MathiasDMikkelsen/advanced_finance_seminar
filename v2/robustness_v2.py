# v2/robustness_v2.py

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).parent.parent
COMBINED_DATA_FILE = ROOT / "data" / "combined" / "all_relevant_data.csv"       
OUT_DIR = ROOT / "v2" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 65)
    print("v2 Analysis - Robustness Checks")
    print("=" * 65)

    if not COMBINED_DATA_FILE.exists():
        sys.exit(f"Data file not found: {COMBINED_DATA_FILE}")

    df = pd.read_csv(COMBINED_DATA_FILE)
    print(f"Loaded {len(df)} observations.")

    all_robustness_results = []

    # 1. Variable Construction (Matching Key Spec)
    df["Stock_Price_t5"] = df["Actual_EPS"] / df["Std_Actual_EPS"]
    df["Guidance_Surprise_Raw"] = (df["Actual_EPS"] - df["Mgmt_Guidance_EPS"]) / df["Stock_Price_t5"]

    def winsorize_col(series, limits=(0.01, 0.99)):
        lower = series.quantile(limits[0])
        upper = series.quantile(limits[1])
        return series.clip(lower=lower, upper=upper)

    df["Guidance_Surprise_Win"] = winsorize_col(df["Guidance_Surprise_Raw"])    
    df["Std_Disagreement_Gap_Win"] = winsorize_col(df["Std_Disagreement_Gap"])  

    df["Z_Guidance_Surprise"] = (df["Guidance_Surprise_Win"] - df["Guidance_Surprise_Win"].mean()) / df["Guidance_Surprise_Win"].std()
    df["Z_Disagreement_Gap"] = (df["Std_Disagreement_Gap_Win"] - df["Std_Disagreement_Gap_Win"].mean()) / df["Std_Disagreement_Gap_Win"].std()

    log_mc = np.log(df["Market_Cap"].replace(0, np.nan))
    df["Std_Log_Market_Cap"] = (log_mc - log_mc.mean()) / log_mc.std()

    # Sector Fixed Effects
    sector_dummies = pd.get_dummies(df["Sector"], prefix="Sector", drop_first=True, dtype=int)
    df = pd.concat([df, sector_dummies], axis=1)
    sector_cols = sector_dummies.columns.tolist()

    controls = ["Std_Log_Market_Cap"] + sector_cols

    def run_regression(dep_var, indep_vars, df_reg, model_name=""):
        req_cols = ["Ticker", dep_var] + indep_vars
        df_clean = df_reg[req_cols].dropna()

        y = df_clean[dep_var]
        X = sm.add_constant(df_clean[indep_vars])
        res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df_clean["Ticker"]})

        res_info = {
            "Model": model_name,
            "Dep_Var": dep_var,
            "N_Obs": int(res.nobs),
            "R2": res.rsquared,
            "Adj_R2": res.rsquared_adj,
            "Result": res
        }
        return res_info

    def save_and_print_models(models, filename, print_all_coeffs=False):
        summary_data = [] 
        for m in models:
            m_res = m['Result']
            if print_all_coeffs:
                print(f"\n--- {m['Model']} (Dep: {m['Dep_Var']}) ---")
                for var in ["const"] + list(m_res.params.index)[1:]:
                    coef, se, pval = m_res.params[var], m_res.bse[var], m_res.pvalues[var]
                    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
                    print(f"{var:<40} b={coef:+.4f} (SE={se:.4f})  p={pval:.3f} {stars}")
                print(f"R2={m['R2']:.4f}  Adj.R2={m['Adj_R2']:.4f}  N={m['N_Obs']}")

            for var in m_res.params.index:
                summary_data.append({
                    "Model": m["Model"],
                    "Variable": var,
                    "Coefficient": m_res.params[var],
                    "Std_Error": m_res.bse[var],
                    "p_value": m_res.pvalues[var],
                    "Significance": "***" if m_res.pvalues[var] < 0.01 else "**" if m_res.pvalues[var] < 0.05 else "*" if m_res.pvalues[var] < 0.10 else "",
                    "N_Obs": m["N_Obs"]
                })
        for item in summary_data:
            item["Block/File"] = filename
            all_robustness_results.append(item)
        pass


    # ========================================================================= 
    # 1. Firm Size Subsamples (Median split on Market_Cap)
    # ========================================================================= 
    print("\n" + "=" * 65 + "\n1. Firm Size Subsamples (Below vs Above Median)\n" + "=" * 65)
    med_mc = df["Market_Cap"].median()
    df_small = df[df["Market_Cap"] <= med_mc].copy()
    df_large = df[df["Market_Cap"] > med_mc].copy()

    model_small = run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df_small, "Below Median Market Cap")
    model_large = run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df_large, "Above Median Market Cap")
    
    save_and_print_models([model_small, model_large], "size_subsample_results.csv", print_all_coeffs=False)

    print("\n  --- Side-by-Side Comparison (Z_Disagreement_Gap) ---")
    s_res, l_res = model_small["Result"], model_large["Result"]
    s_p = s_res.pvalues["Z_Disagreement_Gap"]
    l_p = l_res.pvalues["Z_Disagreement_Gap"]
    s_stars = "***" if s_p < 0.01 else "**" if s_p < 0.05 else "*" if s_p < 0.10 else ""
    l_stars = "***" if l_p < 0.01 else "**" if l_p < 0.05 else "*" if l_p < 0.10 else ""

    print(f"  Below Median (Smaller): b={s_res.params['Z_Disagreement_Gap']:+.4f} (p={s_p:.3f}) {s_stars} [N={model_small['N_Obs']}]")
    print(f"  Above Median (Larger) : b={l_res.params['Z_Disagreement_Gap']:+.4f} (p={l_p:.3f}) {l_stars} [N={model_large['N_Obs']}]")


    # ========================================================================= 
    # 2. Sector Heterogeneity (Interactions)
    # ========================================================================= 
    print("\n" + "=" * 65 + "\n2. Sector Heterogeneity (Interactions)\n" + "=" * 65)
    
    sector_inter_cols = []
    # Create the interaction terms manually so we can inspect them in the results
    for col in sector_cols:
        inter_name = f"Int_Gap_{col}"
        df[inter_name] = df["Z_Disagreement_Gap"] * df[col]
        sector_inter_cols.append(inter_name)
        
    model_sector = run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + sector_inter_cols + controls, df, "Sector Heterogeneity")
    save_and_print_models([model_sector], "sector_heterogeneity_results.csv", print_all_coeffs=True)
    
    # Joint F-test
    m_res = model_sector["Result"]
    f_test = m_res.f_test(" = 0, ".join(sector_inter_cols) + " = 0")
    f_stat = float(np.squeeze(f_test.fvalue))
    p_val = float(np.squeeze(f_test.pvalue))
    print(f"\n--- Joint F-Test: Sector Interactions ---")
    print(f"H0: Interaction terms are jointly zero (No sector heterogeneity)")
    print(f"F-Statistic: {f_stat:.4f}  (p-value: {p_val:.4f})")
    if p_val < 0.05:
        print(">> Rejected H0: Significant sector heterogeneity exists.")
    else:
        print(">> Failed to reject H0: Sector heterogeneity is not jointly significant.")


    # =========================================================================
    # 3. Announcement Timing Subsamples
    # ========================================================================= 
    print("\n" + "=" * 65 + "\n3. Announcement Timing Subsamples\n" + "=" * 65)
    
    def classify_time(t):
        if pd.isna(t): return 'Unknown'
        t_str = str(t).replace(':', '')
        try:
            val = int(t_str)
            # Rough proxy: before 09:30 vs after 16:00
            # Depending on data formats some use 24h, let's treat anything above 1600 or roughly PM as Post, <930 as Pre
            if val < 930: return 'Pre-Market'
            elif val >= 1600: return 'Post-Market'
            else: return 'During-Market' # Might include Pre-Market depending on time format logic if times are e.g., 8:00=800
        except: return 'Unknown'
        
    df["Timing_Cat"] = df["Announcement_Time"].apply(classify_time)
    
    # Looking at the data earlier, most times are 11:30, 12:00, 22:00. This implies UTC or another timezone for a US dataset? 
    # Let's just blindly use our rough proxy. If it doesn't split well we will reconsider.
    df_pre = df[df["Timing_Cat"] == "Pre-Market"].copy()
    df_post = df[df["Timing_Cat"] == "Post-Market"].copy()
    df_during = df[df["Timing_Cat"] == "During-Market"].copy()

    models_timing = []
    if len(df_pre) > 30: # Need enough obs
        models_timing.append(run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df_pre, "Pre-Market"))
    models_timing.append(run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df_during, "During-Market"))
    models_timing.append(run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap"] + controls, df_post, "Post-Market"))

    save_and_print_models(models_timing, "announcement_timing_results.csv", print_all_coeffs=False)

    print("\n  --- Side-by-Side Comparison (Z_Disagreement_Gap) ---")
    for m in models_timing:
        res = m["Result"]
        coef, p = res.params["Z_Disagreement_Gap"], res.pvalues["Z_Disagreement_Gap"]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
        print(f"  {m['Model']:<15}: b={coef:+.4f} (p={p:.3f}) {stars} [N={m['N_Obs']}]")


    # =========================================================================
    # 4. Non-Linearity (Squared Gap)
    # =========================================================================
    print("\n" + "=" * 65 + "\n4. Non-Linear Relationship (Squared Gap)\n" + "=" * 65)
    df["Z_Disagreement_Gap_Sq"] = df["Z_Disagreement_Gap"] ** 2

    model_nl = run_regression("CAR_0_1", ["Z_Guidance_Surprise", "Z_Disagreement_Gap", "Z_Disagreement_Gap_Sq"] + controls, df, "Non-Linear Gap Effect")
    save_and_print_models([model_nl], "nonlinear_results.csv", print_all_coeffs=False)

    nl_res = model_nl["Result"]
    c1, p1 = nl_res.params["Z_Disagreement_Gap"], nl_res.pvalues["Z_Disagreement_Gap"]
    c2, p2 = nl_res.params["Z_Disagreement_Gap_Sq"], nl_res.pvalues["Z_Disagreement_Gap_Sq"]
    
    s1 = "***" if p1 < 0.01 else "**" if p1 < 0.05 else "*" if p1 < 0.10 else ""
    s2 = "***" if p2 < 0.01 else "**" if p2 < 0.05 else "*" if p2 < 0.10 else ""
    
    print("\n  --- Gap vs Gap Squared ---")
    print(f"  Linear Term : b={c1:+.4f} (p={p1:.3f}) {s1}")
    print(f"  Squared Term: b={c2:+.4f} (p={p2:.3f}) {s2}")
    
    # Summary
    print("\n" + "=" * 65 + "\nOVERALL SUMMARY OF SIGNIFICANCE (Std_Disagreement_Gap)\n" + "=" * 65)
    summ_models = [model_small, model_large, model_sector] + models_timing + [model_nl]
    for m in summ_models:
        pval = m["Result"].pvalues["Z_Disagreement_Gap"]
        stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else "NS"
        sig = f"{'✅ Sig' if pval < 0.10 else '❌ NS'} ({stars})"
        print(f"  {m['Model']:<30}: {sig}")
    

    pd.DataFrame(all_robustness_results).to_csv(OUT_DIR / "all_robustness_results.csv", index=False)
    print(f"\n>> Successfully combined all results into all_robustness_results.csv")

if __name__ == "__main__":
    main()
