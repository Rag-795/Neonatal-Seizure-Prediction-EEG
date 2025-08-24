import numpy as np
import pandas as pd
from scipy import signal
import glob
import os

# Define channel names as provided
CHANNELS = [
    "EEG Fp1-REF", "EEG Fp2-REF", "EEG F3-REF", "EEG F4-REF", "EEG C3-REF",
    "EEG C4-REF", "EEG P3-REF", "EEG P4-REF", "EEG O1-REF", "EEG O2-REF",
    "EEG F7-REF", "EEG F8-REF", "EEG T3-REF", "EEG T4-REF", "EEG T5-REF",
    "EEG T6-REF", "EEG Fz-REF", "EEG Cz-REF", "EEG Pz-REF", "ECG EKG-REF",
    "Resp Effort-REF"
]

# Function to load and compute statistics from multiple seizure CSV files
def compute_channel_stats(seizure_files, channels):
    # Initialize stats dictionary
    stats = {ch: {'mean': 0, 'std': 1} for ch in channels}
    
    # Collect statistics across all files
    all_means = {ch: [] for ch in channels}
    all_stds = {ch: [] for ch in channels}
    for file_path in seizure_files:
        df = pd.read_csv(file_path)
        for channel in channels:
            if channel in df.columns:
                data = df[channel].dropna()
                if len(data) > 0:
                    all_means[channel].append(data.mean())
                    all_stds[channel].append(data.std())
    
    # Aggregate stats
    for channel in channels:
        if all_means[channel]:  # If data exists for the channel
            stats[channel] = {
                'mean': np.mean(all_means[channel]),
                'std': np.mean(all_stds[channel])
            }
    
    return stats

# Function to generate synthetic EEG data for one sample
def generate_synthetic_data(stats, num_samples=7680, fs=256):
    np.random.seed()  # Reset seed for each sample to ensure variability
    data = {}
    time = np.arange(num_samples) / fs  # Time vector (30 seconds at 256 Hz)

    for channel in CHANNELS:
        mean = stats[channel]['mean']
        std = stats[channel]['std'] * 1.5  # Increase amplitude for seizure
        noise = np.random.normal(mean, std, num_samples)
        # Apply low-pass filter to mimic EEG smoothness
        b, a = signal.butter(4, 40 / (fs / 2), btype='low')
        noise = signal.filtfilt(b, a, noise)
        data[channel] = noise
    
    data['time'] = time
    return pd.DataFrame(data)

# Function to generate synthetic annotations for all samples
def generate_synthetic_annotations(num_samples=100, seizure_prob=0.1):
    annotations = np.zeros((num_samples, len(CHANNELS)))
    # Simulate sparse seizure events
    for i in range(num_samples):
        if np.random.random() < seizure_prob:
            # Randomly select some channels to mark as seizure
            seizure_channels = np.random.choice(range(len(CHANNELS)), size=5, replace=False)
            annotations[i, seizure_channels] = 1
    return pd.DataFrame(annotations, columns=[f"Ch{i+1}" for i in range(len(CHANNELS))])

# Main execution
if __name__ == "__main__":
    # Define input and output directories
    seizure_folder = "seizure_samples"
    output_folder = "synthetic_data"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all seizure files
    seizure_files = glob.glob(os.path.join(seizure_folder, "eeg*_seizure_sample.csv"))
    if not seizure_files:
        raise FileNotFoundError("No seizure sample files found in seizure_samples folder")
    
    # Load statistics from seizure files
    seizure_stats = compute_channel_stats(seizure_files, CHANNELS)

    # Generate 100 synthetic seizure samples
    for i in range(1, 101):  # Generate samples 1 to 100
        synthetic_seizure = generate_synthetic_data(seizure_stats, num_samples=7680)
        output_file = os.path.join(output_folder, f"synthetic_seizure_eeg_{i}.csv")
        synthetic_seizure.to_csv(output_file, index=False)
        print(f"Generated: {output_file}")

    # Generate and save synthetic annotations
    synthetic_annotations = generate_synthetic_annotations(num_samples=100)
    synthetic_annotations.to_csv(os.path.join(output_folder, "synthetic_annotations.csv"), index=False)

    print("\nSynthetic seizure data and annotations generated and saved in synthetic_data folder:")
    print("- synthetic_seizure_eeg_1.csv to synthetic_seizure_eeg_100.csv")
    print("- synthetic_annotations.csv")