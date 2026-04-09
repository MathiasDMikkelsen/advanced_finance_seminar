# Advanced Finance Seminar

This repository contains our empirical analysis for the Advanced Finance Seminar at the University of Copenhagen.

## Purpose
We are investigating how the market reacts to the "Disagreement Gap": the divergence between Management Earnings Guidance and Analyst Consensus estimates. Our research examines both the immediate market reaction (CAR 0 to 1) and the short-window Post-Earnings Announcement Drift (PEAD, looking at windows from 2 to 15 days). We utilize standardized Z-scores to evaluate how this informational complexity impacts price discovery, including tests for asymmetry and various cross-sectional robustness checks.

## How to Run
1. Activate the virtual environment:
   `powershell
   .\.venv\Scripts\Activate.ps1
   `
2. Ensure dependencies are up to date (pip install -r requirements.txt).
3. Execute the analysis scripts from the root of the project:
   `powershell
   python v2/analysis_v2.py
   python v2/robustness_v2.py
   `
   