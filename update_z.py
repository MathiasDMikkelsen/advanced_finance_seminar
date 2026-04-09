import re
import os

files_to_update = ['v2/analysis_v2.py', 'v2/robustness_v2.py']

for file_path in files_to_update:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Replace in array list and outputs
    lines = text.split('\n')
    for i, line in enumerate(lines):
        if 'Std_Disagreement_Gap_Win' in line:
            # Catch the remaining outputs like SUMMARY OF SIGNIFICANCE and pval checks
            if 'pvalues' in line or 'SUMMARY OF SIGNIFICANCE' in line or 'OVERALL SUMMARY' in line:
                lines[i] = line.replace('Std_Disagreement_Gap_Win', 'Z_Disagreement_Gap')
                
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

print("Done updating pass 2.")