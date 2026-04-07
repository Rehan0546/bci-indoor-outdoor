import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon

# Paths to your data folders
indoor_folder = "/content/drive/MyDrive/Rehan_Dataset/Indoor"
outdoor_folder = "/content/drive/MyDrive/Rehan_Dataset/Outdoor"

chromophores = ['hbo', 'hbr']  # Analyze both HbO and HbR

def load_chromophore_mean(file_path, chromo):
    """
    Load SNIRF file using mne-nirs and extract mean value of specified chromophore.
    """
    raw = mne.io.read_raw_snirf(file_path, preload=True)
    ch_names = [ch for ch in raw.ch_names if chromo in ch.lower()]
    data = raw.copy().pick_channels(ch_names).get_data()
    return np.mean(data)

def compute_condition_means(folder, chromo):
    """
    Compute mean value per subject for a folder (indoor/outdoor).
    """
    all_files = sorted(os.listdir(folder))
    means = []
    for file in all_files:
        path = os.path.join(folder, file)
        try:
            means.append(load_chromophore_mean(path, chromo))
        except Exception as e:
            print(f"Error loading {file} for {chromo}: {e}")
    return np.array(means)

results = {}

for chromo in chromophores:
    # Compute averages
    indoor_avg = compute_condition_means(indoor_folder, chromo)
    outdoor_avg = compute_condition_means(outdoor_folder, chromo)

    results[chromo] = {'Indoor': indoor_avg, 'Outdoor': outdoor_avg}

    # Paired statistical tests
    t_stat, t_p = ttest_rel(indoor_avg, outdoor_avg)
    w_stat, w_p = wilcoxon(indoor_avg, outdoor_avg)

    print(f"\n{chromo.upper()} Analysis:")
    print(f"Indoor mean ± SD: {np.mean(indoor_avg):.3f} ± {np.std(indoor_avg):.3f}")
    print(f"Outdoor mean ± SD: {np.mean(outdoor_avg):.3f} ± {np.std(outdoor_avg):.3f}")
    print(f"Paired t-test: t={t_stat:.3f}, p={t_p:.4f}")
    print(f"Wilcoxon signed-rank: W={w_stat:.3f}, p={w_p:.4f}")

    # Boxplot
    plt.figure(figsize=(6,5))
    plt.boxplot([indoor_avg, outdoor_avg], tick_labels=['Indoor', 'Outdoor'])
    plt.ylabel(f"{chromo.upper()} Mean")
    plt.title(f"{chromo.upper()} Distribution Across Conditions")
    plt.show()
