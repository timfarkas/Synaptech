import torch
from pathlib import Path
import numpy as np

def analyze_wavelet_shards(dataset_path):
    # Initialize stats containers for each band
    eeg_stats = {band: {'values': []} for band in ['alpha', 'beta', 'gamma']}
    mag_stats = {band: {'values': []} for band in ['alpha', 'beta', 'gamma']}
    
    # Process each mode (train/val/test)
    for mode in ['train', 'val', 'test']:
        mode_path = Path(dataset_path) / mode
        if not mode_path.exists():
            print(f"Skipping non-existent mode folder {mode_path}")
            continue
            
        # Process each subject folder
        for subj_folder in mode_path.iterdir():
            if not subj_folder.is_dir():
                continue
                
            # Process EEG wavelet shards
            eeg_wavelet_path = subj_folder / "EEG_WAVELET_shards"
            if eeg_wavelet_path.exists():
                for shard in eeg_wavelet_path.glob("*.pt"):
                    data = torch.load(shard)
                    # For single channel (13), shape should be [3, time, windows]
                    for band_idx, band in enumerate(['alpha', 'beta', 'gamma']):
                        band_data = data[band_idx].numpy().flatten()
                        eeg_stats[band]['values'].append(band_data)
                        
            # Process MAG wavelet shards
            mag_wavelet_path = subj_folder / "MAG_WAVELET_shards"
            if mag_wavelet_path.exists():
                for shard in mag_wavelet_path.glob("*.pt"):
                    data = torch.load(shard)
                    # For single channel (21), shape should be [3, time, windows]
                    for band_idx, band in enumerate(['alpha', 'beta', 'gamma']):
                        band_data = data[band_idx].numpy().flatten()
                        mag_stats[band]['values'].append(band_data)

    # Calculate and print statistics
    print("\nEEG Channel 13 Statistics:")
    print("-" * 50)
    for band in ['alpha', 'beta', 'gamma']:
        if eeg_stats[band]['values']:
            values = np.concatenate(eeg_stats[band]['values'])
            print(f"\n{band.upper()} band:")
            print(f"Min: {values.min():.10f}")
            print(f"Max: {values.max():.10f}")
            print(f"Mean: {values.mean():.10f}")
            print(f"Std: {values.std():.10f}")

    print("\nMAG Channel 21 Statistics:")
    print("-" * 50)
    for band in ['alpha', 'beta', 'gamma']:
        if mag_stats[band]['values']:
            values = np.concatenate(mag_stats[band]['values'])
            print(f"\n{band.upper()} band:")
            print(f"Min: {values.min():.10f}")
            print(f"Max: {values.max():.10f}")
            print(f"Mean: {values.mean():.10f}")
            print(f"Std: {values.std():.10f}")

if __name__ == "__main__":
    dataset_path = "/srv/openfmri/"
    analyze_wavelet_shards(dataset_path)