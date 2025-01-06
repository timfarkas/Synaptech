import os
import shutil
from pathlib import Path

def clear_wavelet_shards(dataset_path):
    """
    Removes:
    - EEG_WAVELET_shards, MAG_WAVELET_shards folders
    - .waveleted marker file
    from the dataset.

    Args:
        dataset_path (str): Path to the root of the dataset directory.
    """
    # Remove wavelet marker file
    wavelet_marker = Path(dataset_path) / '.waveleted'
    if wavelet_marker.exists():
        print(f"Removing marker file: {wavelet_marker}")
        wavelet_marker.unlink()
    else:
        print(f"Marker file {wavelet_marker} does not exist, skipping.")

    # Remove wavelet shard folders
    for mode in ["train", "val", "test"]:
        mode_path = Path(dataset_path) / mode
        if not mode_path.exists():
            print(f"Skipping non-existent folder: {mode_path}")
            continue

        for subject in sorted(mode_path.iterdir()):
            if not subject.is_dir():
                continue

            # Remove only wavelet shard folders
            for shard_folder in ["EEG_WAVELET_shards", "MAG_WAVELET_shards"]:
                shard_path = subject / shard_folder
                if shard_path.exists():
                    print(f"Removing {shard_path}")
                    shutil.rmtree(shard_path)
                else:
                    print(f"{shard_path} does not exist, skipping.")

if __name__ == "__main__":
    dataset_path = "/srv/openfmri"  # Update this path to your dataset's root directory
    clear_wavelet_shards(dataset_path)