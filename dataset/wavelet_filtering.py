import os
import torch
from pathlib import Path
import glob
from tqdm.notebook import tqdm
from dotenv import load_dotenv
import warnings
import numpy as np
import pywt

load_dotenv()
DATASET_PATH = os.getenv('DATASET_PATH')


# Load and concatenate the data
dataset_path = DATASET_PATH  # Adjust this path as needed

class Wavelet_Transformer():
    def __init__(self):
        self.startWaveletFiltering() 

    def startWaveletFiltering(self):
        eeg_tensor, mag_tensor = self._load_and_concat_shards(dataset_path) 
        wavelet_bands_eeg, wavelet_bands_mag = self._filter_wavelet_bands(eeg_tensor, mag_tensor) # Output shape [3, 275, 25706]

    ######################################################################## 
    # ADJUST THE FUNCTION TO WORK ON BOTH TRAIN, TEST & VAL SET
    ########################################################################
    def _load_and_concat_shards(self, dataset_path, mode='train'): 
        """
        Loads and concatenates all EEG and MAG shards from the specified dataset path and mode.

        Args:
            dataset_path (str): Path to the dataset root directory
            mode (str): One of 'train', 'val', or 'test'

        Returns:
            tuple: (concatenated_eeg, concatenated_mag) tensors
        """

        warnings.filterwarnings('ignore')

        eeg_tensors = []
        mag_tensors = []

        # Get all subject folders in the specified mode
        mode_path = Path(dataset_path) / mode
        subject_folders = sorted([f for f in mode_path.iterdir() if f.is_dir()])

        print(f"Loading {mode} data from {len(subject_folders)} subjects...")

        # Iterate through each subject folder
        for subject_folder in tqdm(subject_folders, desc="Loading subjects"):
            # Load EEG shards
            eeg_shard_folder = subject_folder / "EEG_shards"
            if eeg_shard_folder.exists():
                eeg_files = sorted(eeg_shard_folder.glob("*.pt"))
                for eeg_file in eeg_files:
                    eeg_tensor = torch.load(eeg_file)
                    eeg_tensors.append(eeg_tensor)
            
            # Load MAG shards
            mag_shard_folder = subject_folder / "MAG_shards"
            if mag_shard_folder.exists():
                mag_files = sorted(mag_shard_folder.glob("*.pt"))
                for mag_file in mag_files:
                    mag_tensor = torch.load(mag_file)
                    mag_tensors.append(mag_tensor)

        # Concatenate all tensors
        if eeg_tensors:
            concatenated_eeg = torch.cat(eeg_tensors, dim=2)  # Concatenate along windows dimension
        else:
            concatenated_eeg = None
            
        if mag_tensors:
            concatenated_mag = torch.cat(mag_tensors, dim=2)  # Concatenate along windows dimension
        else:
            concatenated_mag = None

        return concatenated_eeg, concatenated_mag

        
    def _compute_wavelet_transform(data_1d, sampling_rate=250, frequencies=None, wavelet='cmor1.5-1.0'):
        """
        Compute the continuous wavelet transform for a 1D signal.
        
        Args:
            data_1d (1D array): The signal values (e.g., one channel over time).
            sampling_rate (float): Sampling rate of the signal (Hz).
            frequencies (array): Array of frequencies to analyze. If None, creates a default set.
            wavelet (str): Name of the wavelet to use (default: 'cmor1.5-1.0').
            
        Returns:
            (coeffs, freqs): 
                coeffs is a 2D numpy array of shape (num_freqs, num_time_points)
                freqs is a 1D numpy array of frequencies corresponding to each row of coeffs
        """
        if frequencies is None:
            # Example frequency range 1-100 Hz
            frequencies = np.logspace(np.log10(1), np.log10(100), num=60)
        
        # Convert frequencies to scales
        scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
        
        # Perform continuous wavelet transform
        coeffs, _ = pywt.cwt(data_1d, scales, wavelet)
        return coeffs, frequencies

    def _filter_wavelet_bands(eeg_data, mag_data, eeg_channel=13, mag_channel=21):
        """
        For the given EEG and MAG channel & data, compute wavelet transforms, 
        keeping only alpha, beta, and gamma frequency bands. 
        
        Returns two tensors:
        - wavelet_bands_eeg: (3, time_points, total_windows) for alpha, beta, gamma
        - wavelet_bands_mag: (3, time_points, total_windows)
        
        Where each index on the 0th dimension corresponds to:
        0 => alpha band average (8-13 Hz)
        1 => beta band average (13-30 Hz)
        2 => gamma band average (30-100 Hz)
        """
        # Define frequency bands
        band_dict = {
            'alpha': (8, 13),
            'beta':  (13, 30),
            'gamma': (30, 100),
        }
        
        # Grab shapes
        # EEG shape => (num_eeg_channels, 275, total_windows)
        # We'll focus on just "eeg_channel" across all windows
        num_windows = eeg_data.shape[2]
        time_points = eeg_data.shape[1]
        
        # Create empty arrays to store the results for alpha, beta, gamma
        # shape => (3, time_points, num_windows)
        wavelet_bands_eeg = np.zeros((3, time_points, num_windows), dtype=np.float32)
        wavelet_bands_mag = np.zeros((3, time_points, num_windows), dtype=np.float32)
        
        # Iterate over each window, compute wavelet, average power in each band
        for w in range(num_windows):
            # Extract signals for this window
            eeg_signal = eeg_data[eeg_channel, :, w].cpu().numpy()
            mag_signal = mag_data[mag_channel, :, w].cpu().numpy()
            
            # Compute wavelet
            eeg_coeffs, freqs = _compute_wavelet_transform(eeg_signal)
            mag_coeffs, _     = _compute_wavelet_transform(mag_signal)
            
            # For each band (alpha, beta, gamma), compute the average magnitude across freq range
            for i, (band_name, (fmin, fmax)) in enumerate(band_dict.items()):
                # Identify which rows in the wavelet transform correspond to the band
                band_mask = (freqs >= fmin) & (freqs <= fmax)
                
                # EEG band average across freq dimension => shape (time_points,)
                band_eeg_vals = np.mean(np.abs(eeg_coeffs[band_mask, :]), axis=0)
                wavelet_bands_eeg[i, :, w] = band_eeg_vals
                
                # MAG band average
                band_mag_vals = np.mean(np.abs(mag_coeffs[band_mask, :]), axis=0)
                wavelet_bands_mag[i, :, w] = band_mag_vals
        
        # Convert to torch tensors
        wavelet_bands_eeg = torch.from_numpy(wavelet_bands_eeg)
        wavelet_bands_mag = torch.from_numpy(wavelet_bands_mag)
        
        return wavelet_bands_eeg, wavelet_bands_mag