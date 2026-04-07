# fNIRS Hemodynamic Analysis: Indoor vs. Outdoor Environments

This repository contains a Python pipeline for processing and analyzing **Functional Near-Infrared Spectroscopy (fNIRS)** data. The script compares cortical hemodynamic responses—specifically oxygenated hemoglobin (**HbO**) and deoxygenated hemoglobin (**HbR**)—between two experimental conditions: **Indoor** and **Outdoor**.

---

### Analysis Overview

The script automates the transition from raw data to statistical results. It handles data loading via `mne-nirs`, extracts mean chromophore concentrations for each subject, and performs comparative statistics to identify significant physiological differences across environments.

---

### Data Structure

The pipeline is designed to navigate a nested directory structure where each subject has their own folder within the condition directory:

* **Indoor Folder:** `/Indoor/sub1/file.snirf`, `/Indoor/sub2/file.snirf`...
* **Outdoor Folder:** `/Outdoor/sub1/file.snirf`, `/Outdoor/sub2/file.snirf`...

The script uses alphabetical sorting to ensure that subject data is correctly paired between the two conditions for the statistical tests.

---

### Key Features

* **Nested Directory Traversal:** Automatically searches through subject subfolders to locate `.snirf` files.
* **Hemodynamic Extraction:** Filters and extracts mean values for both **HbO** and **HbR** channels using MNE-Python.
* **Dual Statistical Testing:**
    * **Paired T-test:** For parametric analysis of mean differences.
    * **Wilcoxon Signed-Rank Test:** For non-parametric analysis, providing robustness against non-normal distributions or outliers.
* **Data Visualization:** Generates boxplots for each chromophore to visualize medians, quartiles, and variance across conditions.

---

### Requirements

The following dependencies are required to run the analysis:

* **mne** & **mne-nirs**: For SNIRF file handling and NIRS processing.
* **numpy**: For numerical operations and array management.
* **scipy**: For performing paired statistical tests.
* **matplotlib**: For generating distribution boxplots.

Install them via pip:
```bash
pip install mne mne-nirs numpy scipy matplotlib
```

---

### Usage

1. **Configure Paths:** Update the `indoor_folder` and `outdoor_folder` variables in the script to match your local or Drive directory.
2. **Execute:** Run the script. It will print the Mean ± Standard Deviation for both conditions, output the $p$-values for both statistical tests, and display the distribution plots.
3. **Interpret Results:** A $p < 0.05$ typically indicates a significant difference in hemodynamic activity between the Indoor and Outdoor environments.

---

### Statistical Metrics Provided

* **Descriptive Stats:** Mean and Standard Deviation for both environments.
* **T-Statistic & P-Value:** Results from the parametric Paired T-test.
* **W-Statistic & P-Value:** Results from the non-parametric Wilcoxon Signed-Rank test.
