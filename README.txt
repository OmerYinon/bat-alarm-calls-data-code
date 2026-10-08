README
======
Code and Data for:
"Context-dependent alarm calls in the Mauritian flying fox, an island-endemic fruit bat"
(Scientific Reports, revised version)

Authors: Omer Yinon, Ksenia Krivoruchko, Ofri Eitan, Ismaeel Janoo, National Parks and Conservation Service, Ryszard Oleksy, Yossi Yovel
Affiliation: Yossi Yovel Lab, Tel Aviv University
Contact: omeryinon98@gmail.com

----------------------------------------------------------------------
OVERVIEW
----------------------------------------------------------------------
This repository contains all analysis scripts and processed acoustic data
required to reproduce the figures and statistics reported in the manuscript.
Acoustic features were extracted from calls of the Mauritian flying fox
(Pteropus niger) recorded during human-entry trials ("Net" / "No-Net";
potential alarm calls) and spontaneous social interactions across two
field seasons (2022 and 2023).

----------------------------------------------------------------------
REVISION NOTE (October 2026)
----------------------------------------------------------------------
In each sheet of all-vocalizations.xlsx, the last row of every trial holds
only the recording filename (a trial-separator row, with no acoustic
measurements). In the original version of scripts 01, 02 and 05 these rows
were not removed and were mean-imputed as if they were calls. The revised
scripts remove them (function drop_trial_separator_rows) before any
analysis. All numbers in the revised manuscript were produced with the
revised scripts. Scripts 03 and 04 were not affected.

----------------------------------------------------------------------
FILES
----------------------------------------------------------------------
all-vocalizations.xlsx
    Master data file. One sheet per vocal category:
      Alarm (2022): day-net(22), day-no-net(22)
      Alarm (2023): day-net(23), day-no-net(23), night-net(23), night-no-net(23)
      Social (2023): food, territorial, reproductive, unknown
      Social (2022): territorial(22), unknown(22)
    Each row is one segmented call; the last row of each trial is a
    separator row containing only the recording filename (see above).
    Columns are acoustic features exported from Avisoft SASLab Pro
    (20 core features measured at 3 time points: start, end and point of
    maximum energy). Amplitude features are excluded in the analyses,
    leaving 48 acoustic features (Supplementary Table 2).
    Note: food(22) calls were reclassified as unknown during annotation and
    are included in the unknown(22) sheet (331 calls).
    Totals: 2,614 potential alarm calls (69 trials) and 2,150 social calls
    (122 interactions).

reduced48_potential9_assignments.csv
    PCA coordinates and acoustic cluster assignments for the 54 calls
    recorded in the small-colony experiment (48 acoustic features).
    Used by script 04 only; place it in a folder named tables/ next to
    the script before running.

01_PCA_analysis.py
    Principal Component Analysis of potential alarm and social calls.
    Produces Figure 1B.
    Output: outputs/pca/

02_RF_2022_3features.py
    Random Forest classifier on the independent 2022 dataset using the 3
    focal features (minimum frequency at call start, bandwidth at call end,
    maximum fundamental frequency); tests cross-year
    generalization, including 100 down-sampled repetitions.
    Reported in the Results text (balanced accuracy 92.6%).
    Output: outputs/rf_2022/

03_ttest_callrate_and_acoustic_features.py
    Welch's t-tests for call-rate comparisons (Net vs No-Net, Day vs Night)
    and trial-level acoustic feature contrasts (alarm vs social).
    Produces Figures 2, 3A, 3B and Supplementary Figure 1.
    Output: outputs/ttests/

04_supp_fig3_tiny_colony_pca.py
    PCA plot of calls from the small-colony experiment.
    Produces Supplementary Figure 2 (the file name keeps its original
    numbering).
    Input: tables/reduced48_potential9_assignments.csv (see above)
    Output: outputs/supp_fig3/

05_RF_2023_alarm_vs_social.py
    Five Random Forest classifiers:
      RF#1 - Alarm (2023) vs Social -> Figure 1C
      RF#2 - Net vs No-Net, Season 2022 (text result)
      RF#3 - Net vs No-Net, Season 2023 (text result)
      RF#4 - Net vs No-Net, Pooled 2022+2023
      RF#5 - Season 2022 Net vs All Other Alarm (text result)
    All include permutation tests (n = 1,000) for empirical chance estimation.
    Output: outputs/rf_2023/

bat_alarm_signature_pipeline.py
subcluster_within_primary_clusters.py
reduced_48_subcluster_analysis.py
reduced_feature_reanalysis.py
    Code for the small-colony acoustic-structure analysis (Methods 5.6):
    call segmentation and clustering pipeline (hierarchical clustering,
    Gaussian mixture models, HDBSCAN, silhouette permutation tests,
    bootstrap stability), subclustering within the two primary clusters,
    and the 48-feature re-analysis.

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

All outputs are saved to subfolders under outputs/.

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
Recordings were made in Mauritius from two captive colonies:
  - 2022 colony: 18 adults, daytime trials only
  - 2023 colony: 17 adults, day and night trials, plus the small-colony
    (reduced-group) experiment
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
Yinon, O. et al. Context-dependent alarm calls in the Mauritian flying fox,
an island-endemic fruit bat. Scientific Reports (in review).
[DOI to be added upon publication]
