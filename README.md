fNIRS Signal Quality and Preprocessing Pipeline

This repository contains the code used for preprocessing, quality assessment, and validation of indoor and outdoor fNIRS recordings in SNIRF format.

The implementation was developed to provide a transparent and reproducible preprocessing workflow for evaluating:

* HbO and HbR signal validity
* raw signal quality
* motion artifacts
* global/systemic signal contamination
* preprocessing effects on signal distributions

Implemented Processing Pipeline

Raw Intensity
→ Optical Density Conversion
→ Notch Filtering
→ Bandpass Filtering
→ Motion Correction using TDDR
→ Beer–Lambert Law Conversion
→ Downsampling to 10 Hz
→ Baseline Correction
→ Signal Validation and Quality Assessment

Implemented Features

* Recursive SNIRF file loading
* Automatic indoor/outdoor file separation
* Optical density conversion
* Notch filtering
* Bandpass filtering
* Motion artifact correction
* Beer–Lambert conversion
* Baseline correction
* Downsampling
* HbO/HbR validation
* Raw intensity SNR computation
* Coefficient of variation analysis
* Motion artifact percentage estimation
* Scalp coupling index computation
* Global signal correlation analysis
* Min-max scaling analysis
* Statistical distribution analysis
* Automatic CSV export
* Automatic visualization generation

Computed Metrics

Raw Intensity Metrics

* Signal-to-noise ratio (SNR)
* Coefficient of variation (CV)
* Motion artifact percentage
* Signal distribution statistics

HbO/HbR Validation Metrics

* HbO mean
* HbR mean
* HbO standard deviation
* HbR standard deviation
* HbO–HbR correlation
* Global signal correlation

Distribution Analysis
For raw, cleaned, and scaled signals:

* mean
* standard deviation
* skewness
* kurtosis

Generated Outputs

* fnirs_quality_metrics_all.csv
* paper_summary.txt
* cleaned HbO/HbR plots

Dependencies

* Python 3.9+
* NumPy
* Pandas
* SciPy
* Scikit-learn
* Matplotlib
* MNE
* MNE-NIRS
* tqdm

Installation

pip install numpy pandas scipy scikit-learn matplotlib mne mne-nirs tqdm

Usage

Place SNIRF files inside the data directory and run:

python main.py

Dataset Structure

data/
├── subject_01/
│   ├── indoor/
│   │   └── recording.snirf
│   └── outdoor/
│       └── recording.snirf

The code automatically separates files using:

* "indoor" in path
* "outdoor" in path

Notes

* Signal quality metrics are computed at the raw intensity stage before hemoglobin conversion.
* HbO and HbR analyses are performed after Beer–Lambert conversion.
* Visualizations are generated using cleaned hemoglobin signals only.
* The implementation uses the MNE-fNIRS framework for preprocessing and validation.


Provided for academic and research purposes.
