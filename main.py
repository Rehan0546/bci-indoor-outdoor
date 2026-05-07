import os
import glob
import warnings

import numpy as np
import pandas as pd
import mne

from tqdm import tqdm

from mne.io import read_raw_snirf
from mne.preprocessing.nirs import (
    optical_density,
    beer_lambert_law,
    scalp_coupling_index,
    temporal_derivative_distribution_repair,
)


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================

ROOT_DIR = r"data"

LOW_CUT = 0.01
HIGH_CUT = 0.2

# physiological/systemic frequencies
NOTCH_FREQ = (0.6, 2.5)

FINAL_SAMPLING_RATE = 10

BASELINE_DURATION = 10

SCI_THRESHOLD = 0.7
CV_THRESHOLD = 0.15
SNR_THRESHOLD_DB = 20

PPF = 0.1


# =============================================================================
# FILE COLLECTION
# =============================================================================

def collect_snirf_files(root_dir):

    print("\nCollecting SNIRF files...")

    all_files = glob.glob(
        os.path.join(root_dir, "**", "*.snirf"),
        recursive=True
    )

    indoor_files = []
    outdoor_files = []

    for file in tqdm(all_files, desc="Scanning Files"):

        lower_path = file.lower()

        if "indoor" in lower_path:
            indoor_files.append(file)

        elif "outdoor" in lower_path:
            outdoor_files.append(file)

    print(f"\nIndoor Files Found : {len(indoor_files)}")
    print(f"Outdoor Files Found: {len(outdoor_files)}")

    return indoor_files, outdoor_files


# =============================================================================
# QUALITY METRICS
# =============================================================================

def compute_raw_intensity_snr(raw):

    data = raw.get_data()

    signal_power = np.mean(data ** 2, axis=1)

    noise_power = np.var(
        data - np.mean(data, axis=1, keepdims=True),
        axis=1
    )

    snr = 10 * np.log10(signal_power / (noise_power + 1e-12))

    return np.mean(snr), snr


def compute_coefficient_variation(raw):

    data = raw.get_data()

    cv = np.std(data, axis=1) / (
        np.mean(np.abs(data), axis=1) + 1e-12
    )

    return np.mean(cv), cv


def compute_scalp_coupling(raw):

    try:

        sci = scalp_coupling_index(raw)

        return np.mean(sci), sci

    except:

        return np.nan, None


def estimate_motion_artifact_percentage(raw):

    data = raw.get_data()

    diff_signal = np.diff(data, axis=1)

    threshold = (
        np.mean(np.abs(diff_signal), axis=1, keepdims=True)
        + 8 * np.std(np.abs(diff_signal), axis=1, keepdims=True)
    )

    motion_mask = np.abs(diff_signal) > threshold

    motion_percentage = (
        np.sum(motion_mask) / motion_mask.size
    ) * 100

    return motion_percentage


# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================

def preprocess_fnirs(raw):

    # -------------------------------------------------------------------------
    # OPTICAL DENSITY
    # -------------------------------------------------------------------------

    raw = optical_density(raw)

    # -------------------------------------------------------------------------
    # NOTCH FILTER
    # -------------------------------------------------------------------------

    raw.notch_filter(
        freqs=NOTCH_FREQ,
        verbose=False
    )

    # -------------------------------------------------------------------------
    # BANDPASS FILTER
    # -------------------------------------------------------------------------

    raw.filter(
        l_freq=LOW_CUT,
        h_freq=HIGH_CUT,
        method="fir",
        verbose=False
    )

    # -------------------------------------------------------------------------
    # MOTION CORRECTION
    # -------------------------------------------------------------------------

    raw = temporal_derivative_distribution_repair(raw)

    # -------------------------------------------------------------------------
    # BEER LAMBERT LAW
    # -------------------------------------------------------------------------

    raw = beer_lambert_law(
        raw,
        ppf=PPF
    )


    # -------------------------------------------------------------------------
    # DOWNSAMPLE
    # -------------------------------------------------------------------------

    raw.resample(
        FINAL_SAMPLING_RATE,
        npad="auto",
        verbose=False
    )

    # -------------------------------------------------------------------------
    # BASELINE CORRECTION
    # -------------------------------------------------------------------------

    data = raw.get_data()

    baseline_samples = int(
        BASELINE_DURATION * raw.info["sfreq"]
    )

    baseline = np.mean(
        data[:, :baseline_samples],
        axis=1,
        keepdims=True
    )

    raw._data = data - baseline

    return raw


# =============================================================================
# GLOBAL SIGNAL CORRELATION
# =============================================================================

def compute_global_correlation(raw):

    data = raw.get_data()

    global_signal = np.mean(data, axis=0)

    correlations = []

    for ch in range(data.shape[0]):

        corr = np.corrcoef(
            data[ch],
            global_signal
        )[0, 1]

        correlations.append(corr)

    return np.mean(correlations)


# =============================================================================
# HbO HbR VALIDATION
# =============================================================================

def validate_hbo_hbr_relationship(raw):

    data = raw.get_data()

    ch_names = raw.ch_names

    hbo_indices = []
    hbr_indices = []

    for idx, name in enumerate(ch_names):

        lower_name = name.lower()

        if "hbo" in lower_name:
            hbo_indices.append(idx)

        elif "hbr" in lower_name:
            hbr_indices.append(idx)

    if len(hbo_indices) == 0 or len(hbr_indices) == 0:

        return {}

    hbo = data[hbo_indices]
    hbr = data[hbr_indices]

    correlations = []

    min_channels = min(
        len(hbo_indices),
        len(hbr_indices)
    )

    for i in tqdm(
        range(min_channels),
        desc="Validating HbO/HbR",
        leave=False
    ):

        corr = np.corrcoef(
            hbo[i],
            hbr[i]
        )[0, 1]

        correlations.append(corr)

    return {
        "HbO_Mean": np.mean(hbo),
        "HbR_Mean": np.mean(hbr),
        "HbO_STD": np.std(hbo),
        "HbR_STD": np.std(hbr),
        "HbO_HbR_Correlation_Mean": np.mean(correlations),
    }


# =============================================================================
# PROCESS SINGLE FILE
# =============================================================================

def process_single_file(file_path):

    print("\n=================================================")
    print(f"Processing File:")
    print(file_path)
    print("=================================================")

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    print("\nReading SNIRF file...")

    raw = read_raw_snirf(
        file_path,
        preload=True,
        verbose=False
    )

    # -------------------------------------------------------------------------
    # RAW QUALITY METRICS
    # -------------------------------------------------------------------------

    print("Computing Raw Quality Metrics...")

    raw_snr_mean, _ = compute_raw_intensity_snr(raw)

    raw_cv_mean, _ = compute_coefficient_variation(raw)

    raw_sci_mean, _ = compute_scalp_coupling(raw)

    motion_percentage = estimate_motion_artifact_percentage(raw)

    # -------------------------------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------------------------------

    print("Running Preprocessing Pipeline...")

    raw = preprocess_fnirs(raw)

    # -------------------------------------------------------------------------
    # HbO HbR VALIDATION
    # -------------------------------------------------------------------------

    print("Validating HbO and HbR Relationship...")

    validation_results = validate_hbo_hbr_relationship(raw)

    # -------------------------------------------------------------------------
    # GLOBAL SIGNAL
    # -------------------------------------------------------------------------

    print("Computing Global Signal Correlation...")

    global_corr = compute_global_correlation(raw)

    # -------------------------------------------------------------------------
    # QUALITY FLAGS
    # -------------------------------------------------------------------------

    snr_pass = raw_snr_mean >= SNR_THRESHOLD_DB

    cv_pass = raw_cv_mean <= CV_THRESHOLD

    sci_pass = (
        raw_sci_mean >= SCI_THRESHOLD
        if not np.isnan(raw_sci_mean)
        else False
    )

    # -------------------------------------------------------------------------
    # OUTPUT
    # -------------------------------------------------------------------------

    output = {
        "File": file_path,

        "Raw_SNR_dB": raw_snr_mean,
        "Raw_CV": raw_cv_mean,
        "SCI": raw_sci_mean,
        "Motion_Artifact_Percentage": motion_percentage,

        "Global_Correlation": global_corr,

        "SNR_Pass": snr_pass,
        "CV_Pass": cv_pass,
        "SCI_Pass": sci_pass,
    }

    output.update(validation_results)

    return output


# =============================================================================
# PROCESS GROUP
# =============================================================================

def process_group(file_list, group_name):

    print("\n=================================================")
    print(f"PROCESSING GROUP: {group_name}")
    print("=================================================")

    all_results = []

    for file_path in tqdm(
        file_list,
        desc=f"{group_name} Files"
    ):

        try:

            result = process_single_file(file_path)

            all_results.append(result)

        except Exception as e:

            print(f"\nFailed Processing:")
            print(file_path)
            print(e)

    df = pd.DataFrame(all_results)

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------

    print("\n=================================================")
    print(f"{group_name} SUMMARY")
    print("=================================================")

    if len(df) > 0:

        print(df.describe())

        total_subjects = len(df)

        snr_pass_count = np.sum(df["SNR_Pass"])

        cv_pass_count = np.sum(df["CV_Pass"])

        sci_pass_count = np.sum(df["SCI_Pass"])

        print("\nQuality Summary")
        print("----------------------------")

        print(f"Total Subjects : {total_subjects}")

        print(
            f"SNR Pass        : "
            f"{snr_pass_count}/{total_subjects}"
        )

        print(
            f"CV Pass         : "
            f"{cv_pass_count}/{total_subjects}"
        )

        print(
            f"SCI Pass        : "
            f"{sci_pass_count}/{total_subjects}"
        )

    return df


# =============================================================================
# PAPER SUMMARY
# =============================================================================

def generate_paper_summary(df, group_name):

    if len(df) == 0:

        return f"\nNo valid files found for {group_name}"

    text = []

    text.append("\n=================================================")
    text.append(f"PAPER SUMMARY : {group_name}")
    text.append("=================================================")

    text.append(
        f"Raw Intensity SNR (dB): "
        f"{df['Raw_SNR_dB'].mean():.2f} ± "
        f"{df['Raw_SNR_dB'].std():.2f}"
    )

    text.append(
        f"Coefficient of Variation: "
        f"{df['Raw_CV'].mean():.4f} ± "
        f"{df['Raw_CV'].std():.4f}"
    )

    text.append(
        f"Motion Artifact Percentage: "
        f"{df['Motion_Artifact_Percentage'].mean():.2f}% ± "
        f"{df['Motion_Artifact_Percentage'].std():.2f}%"
    )

    text.append(
        f"Scalp Coupling Index: "
        f"{df['SCI'].mean():.3f} ± "
        f"{df['SCI'].std():.3f}"
    )

    text.append(
        f"HbO Mean: "
        f"{df['HbO_Mean'].mean():.6f}"
    )

    text.append(
        f"HbR Mean: "
        f"{df['HbR_Mean'].mean():.6f}"
    )

    text.append(
        f"HbO STD: "
        f"{df['HbO_STD'].mean():.6f}"
    )

    text.append(
        f"HbR STD: "
        f"{df['HbR_STD'].mean():.6f}"
    )

    text.append(
        f"HbO-HbR Correlation: "
        f"{df['HbO_HbR_Correlation_Mean'].mean():.3f}"
    )

    text.append(
        f"Global Signal Correlation: "
        f"{df['Global_Correlation'].mean():.3f}"
    )

    return "\n".join(text)


# =============================================================================
# MAIN
# =============================================================================

def main():

    indoor_files, outdoor_files = collect_snirf_files(ROOT_DIR)

    # -------------------------------------------------------------------------
    # INDOOR
    # -------------------------------------------------------------------------

    indoor_df = process_group(
        indoor_files,
        "INDOOR"
    )

    indoor_df.to_csv(
        "indoor_quality_metrics.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # OUTDOOR
    # -------------------------------------------------------------------------

    outdoor_df = process_group(
        outdoor_files,
        "OUTDOOR"
    )

    outdoor_df.to_csv(
        "outdoor_quality_metrics.csv",
        index=False
    )

    # -------------------------------------------------------------------------
    # FINAL PAPER SUMMARIES
    # -------------------------------------------------------------------------

    indoor_summary = generate_paper_summary(
        indoor_df,
        "INDOOR"
    )

    outdoor_summary = generate_paper_summary(
        outdoor_df,
        "OUTDOOR"
    )

    print(indoor_summary)

    print(outdoor_summary)

    # -------------------------------------------------------------------------
    # SAVE PAPER SUMMARY
    # -------------------------------------------------------------------------

    with open("paper_summary.txt", "w") as f:

        f.write(indoor_summary)
        f.write("\n\n")
        f.write(outdoor_summary)

    print("\nSaved:")
    print("1. indoor_quality_metrics.csv")
    print("2. outdoor_quality_metrics.csv")
    print("3. paper_summary.txt")


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()
