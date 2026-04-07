# fNIRS Hemodynamic Comparative Analysis

This repository contains a Python-based pipeline for analyzing **Functional Near-Infrared Spectroscopy (fNIRS)** data. The script automates the extraction, statistical testing, and visualization of hemoglobin concentration changes across two experimental conditions: **Indoor** and **Outdoor** environments.

---

## ## Overview

The analysis focuses on two primary chromophores:
1.  **HbO** (Oxygenated Hemoglobin)
2.  **HbR** (Deoxygenated Hemoglobin)

By processing raw `.snirf` files, the script evaluates whether environmental transitions significantly impact cortical hemodynamic responses using a paired-sample experimental design.

---

## ## Key Features

* **SNIRF Integration**: Utilizes `mne-nirs` for high-level handling of Near-Infrared Spectroscopy Data Format (SNIRF).
* **Dual Statistical Testing**:
    * **Paired T-Test**: For parametric evaluation of mean differences.
    * **Wilcoxon Signed-Rank Test**: A non-parametric alternative for robust analysis against outliers or non-normal distributions.
* **Automated Batch Processing**: Iterates through subject folders to calculate per-subject means for both conditions.
* **Visual Analytics**: Generates comparative boxplots to illustrate data spread, medians, and variance.

---

## ## Requirements

Ensure you have the following dependencies installed:

```bash
pip install mne mne-nirs numpy scipy matplotlib
```

---

## ## Data Structure

The script expects the following directory structure (default paths are configured for Google Colab/Drive):

```text
/Rehan_Dataset/
├── Indoor/
│   ├── subject_01.snirf
│   ├── subject_02.snirf ...
└── Outdoor/
    ├── subject_01.snirf
    ├── subject_02.snirf ...
```

> [!IMPORTANT]  
> The script uses `sorted()` on file names. Ensure that files in both folders are named consistently so that `subject_01` in the Indoor folder matches `subject_01` in the Outdoor folder for accurate paired testing.

---

## ## Usage

1.  Clone this repository.
2.  Update the `indoor_folder` and `outdoor_folder` variables in the script to point to your data.
3.  Run the script:
    ```bash
    python nirs_analysis.py
    ```

---

## ## Output Metrics

For each chromophore (HbO/HbR), the script provides:
* **Descriptive Stats**: Mean and Standard Deviation ($\sigma$) for both conditions.
* **Statistical Significance**: $t$-statistics, $W$-statistics, and $p$-values.
* **Distribution Plots**: Boxplots showing the interquartile range (IQR) and potential outliers.
