import re

def update_file(filepath, global_var_name, out_filename):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    # Add the global list definition at the start of main
    text = text.replace('df = pd.read_csv(COMBINED_DATA_FILE)\n    print(f"Loaded {len(df)} observations.")',
                        f'df = pd.read_csv(COMBINED_DATA_FILE)\n    print(f"Loaded {{len(df)}} observations.")\n\n    {global_var_name} = []')

    # Modify print_and_save_block in analysis_v2 or save_and_print_models in robustness_v2
    # Find the line where it writes to csv
    if 'print_and_save_block(models, out_file):' in text:
        text = text.replace('def print_and_save_block(models, out_file):', 'def print_and_save_block(models, out_file):')
        text = text.replace('pd.DataFrame(summary_data).to_csv(OUT_DIR / out_file, index=False)',
                            f'for item in summary_data:\n            item["Block/File"] = out_file\n            {global_var_name}.append(item)')
        text = text.replace('print(f"\\n>> Saved {out_file} to {OUT_DIR}")', 'pass')

    if 'def save_and_print_models(models, filename, print_all_coeffs=False):' in text:
        text = text.replace('pd.DataFrame(summary_data).to_csv(OUT_DIR / filename, index=False)',
                            f'for item in summary_data:\n            item["Block/File"] = filename\n            {global_var_name}.append(item)')
        text = text.replace('print(f">> Saved {filename}")', 'pass')


    # Special case for quartile analysis list in analysis_v2
    if 'q_summary.append({' in text:
        # Instead of 'pd.DataFrame(q_summary).to_csv(OUT_DIR / "quartile_analysis_results.csv", index=False)'
        # Let's just find that export and replace it.
        text = text.replace('pd.DataFrame(q_summary).to_csv(OUT_DIR / "quartile_analysis_results.csv", index=False)\n    print(f"\\n>> Saved quartile_analysis_results.csv to {OUT_DIR}")',
                            f'for item in q_summary:\n        item["Block/File"] = "quartile_analysis_results.csv"\n        {global_var_name}.append(item)')

    # Add the final export at the end of the script before if __name__ == "__main__":
    export_code = f'\n    pd.DataFrame({global_var_name}).to_csv(OUT_DIR / "{out_filename}", index=False)\n    print(f"\\n>> Successfully combined all results into {out_filename}")\n\nif __name__ == "__main__":'
    text = text.replace('if __name__ == "__main__":', export_code)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(text)

update_file('v2/analysis_v2.py', 'all_main_results', 'all_analysis_results.csv')
update_file('v2/robustness_v2.py', 'all_robustness_results', 'all_robustness_results.csv')
print("Updater complete.")