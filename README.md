fNIRS Indoor and Outdoor Signal Quality Analysis

This repository contains the preprocessing and quality assessment pipeline used for evaluating indoor and outdoor fNIRS recordings stored in SNIRF format.

The implementation focuses on:

* preprocessing transparency
* HbO and HbR validation
* raw intensity signal quality assessment
* motion artifact estimation
* global signal contamination analysis
* indoor versus outdoor comparison

The pipeline was developed in response to reviewer concerns regarding:

* validity of HbR signals
* preprocessing correctness
* potential systemic/global contamination
* interpretation of signal quality metrics

Features

* Automatic recursive SNIRF file loading
* Indoor and outdoor file separation using file paths
* Optical density conversion
* Notch filtering
* Bandpass filtering
* Motion artifact correction using TDDR
* Beer–Lambert conversion
* Baseline correction
* Downsampling to 10 Hz
* HbO/HbR relationship validation
* Raw intensity SNR computation
* Coefficient of variation computation
* Motion artifact percentage estimation
* Global signal correlation analysis
* CSV export of quality metrics
* Paper-ready summary generation

Processing Pipeline

Raw Intensity
→ Optical Density Conversion
→ Notch Filtering
→ Bandpass Filtering
→ Motion Correction (TDDR)
→ Beer–Lambert Law Conversion
→ Baseline Correction
→ Downsampling to 10 Hz
→ Quality Assessment and Validation

Signal Quality Metrics

The following metrics are computed at the raw intensity level before hemoglobin conversion:

* Signal-to-Noise Ratio (SNR)
* Coefficient of Variation (CV)
* Motion Artifact Percentage

Additional validation metrics include:

* HbO mean and standard deviation
* HbR mean and standard deviation
* HbO–HbR correlation
* Global signal correlation

Directory Structure

Example dataset structure:

data/
├── subject_01/
│   ├── indoor/
│   │   └── recording.snirf
│   └── outdoor/
│       └── recording.snirf
├── subject_02/
│   ├── indoor/
│   └── outdoor/

Files are automatically categorized using:

* "indoor" in path
* "outdoor" in path

Installation

Create a Python environment and install dependencies:

pip install numpy pandas scipy tqdm mne mne-nirs

Required Libraries

* Python 3.9+
* numpy
* pandas
* scipy
* tqdm
* mne
* mne-nirs

Usage

Place all SNIRF files inside the data directory.

Run:

python main.py

Outputs

The pipeline generates:

* indoor_quality_metrics.csv
* outdoor_quality_metrics.csv
* paper_summary.txt

Example Output

Raw Intensity SNR (dB): 27.68 ± 1.72
Coefficient of Variation: 0.0455 ± 0.0096
Motion Artifact Percentage: 0.02% ± 0.02%
HbO-HbR Correlation: -0.021
Global Signal Correlation: 0.344

Interpretation

* Higher SNR indicates better signal quality
* Lower coefficient of variation indicates more stable recordings
* Lower motion artifact percentage indicates fewer transient artifacts
* Moderate negative or near-zero HbO–HbR correlations are more physiologically plausible than strong positive correlations
* High global signal correlation may indicate systemic physiological contamination

Notes

* Signal quality metrics are computed on raw intensity data prior to optical density and hemoglobin conversion.
* The implementation uses the MNE-fNIRS framework for preprocessing.
* The code was designed for reproducibility and reviewer transparency.

Citation

If this repository is used in research, please cite the associated publication.

License

This project is provided for academic and research purposes.
