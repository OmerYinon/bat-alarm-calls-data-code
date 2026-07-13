README
======
Code and Data for:
"First evidence consistent with context-dependent alarm calls in bats"

Authors: Omer Yinon, Ksenia Krivoruchko, Ofri Eitan, Ismaeel Janoo, National Parks and Conservation Service, Ryszard Oleksy, Yossi Yovel
Affiliation: Yossi Yovel Lab, Tel Aviv University
Contact: omeryinon98@gmail.com

----------------------------------------------------------------------
OVERVIEW
----------------------------------------------------------------------
This repository contains all analysis scripts and processed acoustic data
required to reproduce the figures and statistics reported in the manuscript.
Acoustic features were extracted from calls of the Mauritian flying fox
(Pteropus niger) recorded during alarm (net-capture) and social contexts
across two field seasons (2022 and 2023).

----------------------------------------------------------------------
FILES
----------------------------------------------------------------------
all-vocalizations.xlsx
    Master data file. One sheet per vocal category:
      Alarm (2022): day-net(22), day-no-net(22)
      Alarm (2023): day-net(23), day-no-net(23), night-net(23), night-no-net(23)
      Social:       food, territorial, reproductive, unknown
      Social (2022): territorial(22), unknown(22)
    Each row is one segmented call. Columns are acoustic features extracted
    in Avisoft SASLab Pro (48 features x 3 time points = up to 60 columns).
    Note: food(22) calls were reclassified as unknown during annotation and
    are included in the unknown(22) sheet (n=345 total).

tables/reduced48_potential9_assignments.csv
    PCA coordinates and acoustic cluster assignments for 54 calls recorded
    in reduced-social trials (48 reduced + 9 potential alarm calls).
    Used by script 04 only.

01_PCA_analysis.py
    Principal Component Analysis of alarm and social calls.
    Produces Figure 1B.
    Output: outputs/pca/fig1B_pca_alarm_vs_social.png/.svg

02_RF_2022_3features.py
    Random Forest classifier on the 2022 dataset using 3 SHAP-identified
    features (min frequency at call start, bandwidth at call end, maximum
    fundamental frequency). Tests cross-year generalization.
    Produces Figure 1C (cross-year panel) and associated statistics.
    Output: outputs/rf_2022/

03_ttest_callrate_and_acoustic_features.py
    Welch's t-tests for call-rate comparisons (Net vs No-Net, Day vs Night)
    and trial-level acoustic feature contrasts (alarm vs social).
    Produces Figures 2, 3A, 3B, and Supplementary Figure 1.
    Output: outputs/ttests/

04_supp_fig3_tiny_colony_pca.py
    PCA and acoustic clustering of calls from reduced-social trials.
    Produces Supplementary Figure 3.
    Output: outputs/supp_fig3/
    Input: tables/reduced48_potential9_assignments.csv

05_RF_2023_alarm_vs_social.py
    Five Random Forest classifiers:
      RF#1 - Alarm (2023) vs Social -> Figure 1C (main panel)
      RF#2 - Net vs No-Net, Season 2022 (text result)
      RF#3 - Net vs No-Net, Season 2023 (text result)
      RF#4 - Net vs No-Net, Pooled 2022+2023 (text result)
      RF#5 - Season 2022 Net vs All Other Alarm (text result)
    All include permutation tests (n=1000) for empirical chance estimation.
    Output: outputs/rf_2023/

compile_results.py
    Run after all five scripts to compile all key statistics into a single
    human-readable file: results_summary.txt

results_summary.txt
    Auto-generated summary of all key statistics. All numbers match the
    manuscript text exactly.

----------------------------------------------------------------------
HOW TO RUN
----------------------------------------------------------------------
1. Install dependencies (Python 3.9+):
   pip install numpy pandas matplotlib scikit-learn shap scipy seaborn openpyxl

2. Place all-vocalizations.xlsx in the same folder as the scripts.

3. Run scripts in order:
   python 01_PCA_analysis.py
   python 02_RF_2022_3features.py        # ~5 min (permutation test n=1000)
   python 03_ttest_callrate_and_acoustic_features.py
   python 04_supp_fig3_tiny_colony_pca.py
   python 05_RF_2023_alarm_vs_social.py  # ~15 min (5 x permutation test n=1000)

4. Compile results:
   python compile_results.py
   -> results_summary.txt

All outputs are saved to subfolders under outputs/.
Scripts use Path(__file__).resolve().parent to locate input files, so they
can be run from any working directory.

----------------------------------------------------------------------
SOFTWARE VERSIONS
----------------------------------------------------------------------
Python        3.9+
numpy         1.24+
pandas        1.5+
matplotlib    3.6+
scikit-learn  1.2+
shap          0.41+
scipy         1.10+
seaborn       0.12+
openpyxl      3.1+

----------------------------------------------------------------------
DATA COLLECTION
----------------------------------------------------------------------
Acoustic recordings were made at two field sites:
  - 2022 colony: Mauritius (small colony, day trials only)
  - 2023 colony: Mauritius (larger colony, day and night trials)
Calls were segmented and acoustic features extracted using
Avisoft SASLab Pro. See manuscript Methods for full details.

----------------------------------------------------------------------
LICENSE
----------------------------------------------------------------------
Code: MIT License
Data: CC BY 4.0

----------------------------------------------------------------------
CITATION
----------------------------------------------------------------------
If you use this code or data, please cite:
Yinon O., Krivoruchko K., Eitan O., Janoo I., NPCS, Oleksy R., Yovel Y. (2026). First evidence consistent with
context-dependent alarm calls in bats. Proceedings of the Royal Society B.
[DOI to be added upon publication]
