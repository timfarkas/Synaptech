import torch
from pathlib import Path
import warnings
import numpy as np
import pywt
from tqdm import tqdm  
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)

class Wavelet_Transformer():
    def __init__(
        self,
        dataset_path,
        mode='all',
        eeg_channel=all,
        mag_channel=21
    ):
        """
        Applies wavelet transforms to EEG and MEG data, then min-max normalizes
        the wavelet coefficients across runs/subjects before saving.

        Args:
            dataset_path: Root path containing train/val/test subdirectories
            mode: Dataset split to process ('train', 'val', 'test', None, 'all')
            eeg_channel: EEG channel(s) to transform (int or 'all')
            mag_channel: MEG channel(s) to transform (int or 'all')

        Creates .waveleted marker file to avoid reprocessing.
        """

        self.dataset_path = dataset_path
        valid_modes = ['train', 'val', 'test', None, 'all']
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes} but got {mode}")
        self.mode = mode if mode in ['train', 'val', 'test'] else None

        self.eeg_channel = eeg_channel
        self.mag_channel = mag_channel

        # Store global min/max for wavelet data here so they can be used for normalization
        # Format for EEG: shape = (n_channels, 3) if we do 'all' channels
        # Format for MAG: shape = (n_channels, 3)
        # If single-channel, these become shape = (1, 3).
        self.global_eeg_min = None
        self.global_eeg_max = None
        self.global_mag_min = None
        self.global_mag_max = None

        # Check if a .waveleted marker already exists
        self.waveleted_marker = Path(self.dataset_path) / ".waveleted"
        if self.waveleted_marker.exists():
            pass
        else:
            # Automatically run the wavelet filtering (only if marker does not exist)
            self.process_shards_individually()
            # Create the .waveleted marker so we don't re-run next time
            self.waveleted_marker.touch()
            print(f"Finished wavelet transform.")

    def process_shards_individually(self):
        """
        Adjusted logic:
        1) First pass: gather global min/max stats from wavelet data across runs/subjects (no saving yet).
        2) Second pass: re-run wavelet transform, apply min-max normalization, then save wavelet shards.

        This ensures wavelet data is normalized per electrode (or channel) across all runs/subjects.
        """

        # We do two passes over the shards:

        # ============================= PASS 1 =============================
        # Gather wavelet min/max for each channel/band across runs and subjects
        self._gather_stats_pass()

        # ============================= PASS 2 =============================
        # Now we have global min/max in memory, do wavelet transform again,
        # apply min-max normalization, and save shards
        self._normalize_and_save_pass()

    def _gather_stats_pass(self):
        """
        Pass 1: Loop over each shard, compute wavelet transform, and track global min/max 
        for EEG and MAG (per channel, per band). We do NOT save anything here.
        """

        # We'll always process these three modes if mode was not specifically set:
        if self.mode in ['train', 'val', 'test']:
            modes_to_process = [self.mode]
        else:
            modes_to_process = ['train', 'val', 'test']

        for mode in modes_to_process:
            mode_path = Path(self.dataset_path) / mode
            if not mode_path.exists():
                print(f"Skipping non-existent mode folder {mode_path}")
                continue

            pt_files = []
            subject_folders = [f for f in mode_path.iterdir() if f.is_dir()]

            for subj_folder in subject_folders:
                # EEG shards
                eeg_shards_path = subj_folder / "EEG_shards"
                if eeg_shards_path.exists():
                    for pt_file in sorted(eeg_shards_path.glob("*.pt")):
                        pt_files.append(("EEG", pt_file))

                # MAG shards
                mag_shards_path = subj_folder / "MAG_shards"
                if mag_shards_path.exists():
                    for pt_file in sorted(mag_shards_path.glob("*.pt")):
                        pt_files.append(("MAG", pt_file))

            # Single progress bar for all EEG+MAG shards in this mode
            for shard_type, pt_file in tqdm(pt_files, desc=f"Computing wavelets for [{mode}]"):
                data = torch.load(pt_file)
                if shard_type == "EEG":
                    wavelet_eeg, _ = self._filter_wavelet_bands(
                        data, 
                        None,  
                        self.eeg_channel,  
                        None,
                        gather_stats_only=True
                    )
                    # wavelet_eeg shape: 
                    #   single channel: [3, time_points, num_windows]
                    #   multiple channels: [n_eeg_channels, 3, time_points, num_windows]
                    self._update_global_stats(wavelet_eeg, is_eeg=True)

                else:  # shard_type == "MAG"
                    _, wavelet_mag = self._filter_wavelet_bands(
                        None, 
                        data,  
                        None, 
                        self.mag_channel,
                        gather_stats_only=True
                    )
                    # wavelet_mag shape similar to wavelet_eeg above
                    self._update_global_stats(wavelet_mag, is_eeg=False)

    def _normalize_and_save_pass(self):
        """
        Pass 2: After we have global min/max, re-run wavelet transform, min-max normalize 
        wavelet data, and save it.
        """

        # We'll always process these three modes if mode was not specifically set:
        if self.mode in ['train', 'val', 'test']:
            modes_to_process = [self.mode]
        else:
            modes_to_process = ['train', 'val', 'test']

        for mode in modes_to_process:
            mode_path = Path(self.dataset_path) / mode
            if not mode_path.exists():
                print(f"Skipping non-existent mode folder {mode_path}")
                continue

            pt_files = []
            subject_folders = [f for f in mode_path.iterdir() if f.is_dir()]

            for subj_folder in subject_folders:
                # EEG shards
                eeg_shards_path = subj_folder / "EEG_shards"
                if eeg_shards_path.exists():
                    out_eeg_wavelet_dir = subj_folder / "EEG_WAVELET_shards"
                    out_eeg_wavelet_dir.mkdir(exist_ok=True)
                    for pt_file in sorted(eeg_shards_path.glob("*.pt")):
                        pt_files.append(("EEG", pt_file, out_eeg_wavelet_dir))

                # MAG shards
                mag_shards_path = subj_folder / "MAG_shards"
                if mag_shards_path.exists():
                    out_mag_wavelet_dir = subj_folder / "MAG_WAVELET_shards"
                    out_mag_wavelet_dir.mkdir(exist_ok=True)
                    for pt_file in sorted(mag_shards_path.glob("*.pt")):
                        pt_files.append(("MAG", pt_file, out_mag_wavelet_dir))

            for shard_type, pt_file, output_dir in tqdm(pt_files, desc=f"Normalizing & saving wavelets for [{mode}]"):
                data = torch.load(pt_file)
                if shard_type == "EEG":
                    wavelet_eeg, _ = self._filter_wavelet_bands(
                        data, 
                        None,  
                        self.eeg_channel,  
                        None,
                        gather_stats_only=False
                    )
                    # Min-max normalize the wavelet data
                    wavelet_eeg = self._apply_minmax_normalization(wavelet_eeg, is_eeg=True)
                    wavelet_eeg = wavelet_eeg.half() # convert to float16

                    out_name = pt_file.stem + "_wavelet.pt"
                    torch.save(wavelet_eeg.float(), output_dir / out_name)

                else:  # shard_type == "MAG"
                    _, wavelet_mag = self._filter_wavelet_bands(
                        None, 
                        data,  
                        None, 
                        self.mag_channel,
                        gather_stats_only=False
                    )
                    # Min-max normalize
                    wavelet_mag = self._apply_minmax_normalization(wavelet_mag, is_eeg=False)
                    wavelet_mag = wavelet_mag.half() #convert to float16

                    out_name = pt_file.stem + "_wavelet.pt"
                    torch.save(wavelet_mag.float(), output_dir / out_name)

    def _update_global_stats(self, wavelet_data, is_eeg=True):
        """
        Updates self.global_eeg_min/self.global_eeg_max or self.global_mag_min/self.global_mag_max
        given a new wavelet shard. We look at the entire wavelet data (channel, band, time, windows).
        We only gather the absolute min and max across time+windows for each channel and band.

        wavelet_data shape can be:
         - single-channel: (3, time, windows)
         - multi-channel: (n_channels, 3, time, windows)
        """
        if wavelet_data is None:
            return

        # Convert to numpy for min/max
        wavelet_data_np = wavelet_data.cpu().numpy()

        if len(wavelet_data_np.shape) == 3:
            # shape is [3, time, windows] => single channel
            n_channels = 1
        else:
            # shape is [n_channels, 3, time, windows]
            n_channels = wavelet_data_np.shape[0]

        # Initialize min/max arrays if not done
        if is_eeg:
            if self.global_eeg_min is None or self.global_eeg_max is None:
                # (n_channels, 3)
                self.global_eeg_min = np.full((n_channels, 3), np.inf, dtype=np.float32)
                self.global_eeg_max = np.full((n_channels, 3), -np.inf, dtype=np.float32)
        else:
            if self.global_mag_min is None or self.global_mag_max is None:
                self.global_mag_min = np.full((n_channels, 3), np.inf, dtype=np.float32)
                self.global_mag_max = np.full((n_channels, 3), -np.inf, dtype=np.float32)

        # Update stats
        if n_channels == 1:
            # wavelet_data_np shape: (3, time, windows)
            for band_i in range(3):
                band_vals = wavelet_data_np[band_i, :, :]  # shape: (time, windows)
                local_min = band_vals.min()
                local_max = band_vals.max()
                if is_eeg:
                    self.global_eeg_min[0, band_i] = min(self.global_eeg_min[0, band_i], local_min)
                    self.global_eeg_max[0, band_i] = max(self.global_eeg_max[0, band_i], local_max)
                else:
                    self.global_mag_min[0, band_i] = min(self.global_mag_min[0, band_i], local_min)
                    self.global_mag_max[0, band_i] = max(self.global_mag_max[0, band_i], local_max)
        else:
            # wavelet_data_np shape: (n_channels, 3, time, windows)
            for ch_i in range(n_channels):
                for band_i in range(3):
                    band_vals = wavelet_data_np[ch_i, band_i, :, :]  # shape: (time, windows)
                    local_min = band_vals.min()
                    local_max = band_vals.max()
                    if is_eeg:
                        self.global_eeg_min[ch_i, band_i] = min(self.global_eeg_min[ch_i, band_i], local_min)
                        self.global_eeg_max[ch_i, band_i] = max(self.global_eeg_max[ch_i, band_i], local_max)
                    else:
                        self.global_mag_min[ch_i, band_i] = min(self.global_mag_min[ch_i, band_i], local_min)
                        self.global_mag_max[ch_i, band_i] = max(self.global_mag_max[ch_i, band_i], local_max)

    def _apply_minmax_normalization(self, wavelet_data, is_eeg=True):
        """
        Applies min-max normalization on wavelet_data using the global min/max
        previously gathered. Normalizes per [channel, band] across time+windows.
        wavelet_data shape:
         - single-channel: [3, time, windows]
         - multi-channel: [n_channels, 3, time, windows]
        """

        if wavelet_data is None:
            return None

        wavelet_data_np = wavelet_data.cpu().numpy()
        if len(wavelet_data_np.shape) == 3:
            # Single channel
            n_channels = 1
        else:
            # Multiple channels
            n_channels = wavelet_data_np.shape[0]

        # Retrieve appropriate global arrays
        if is_eeg:
            gmin = self.global_eeg_min
            gmax = self.global_eeg_max
        else:
            gmin = self.global_mag_min
            gmax = self.global_mag_max

        # Safety check
        if gmin is None or gmax is None:
            # No stats gathered (shouldn't happen)
            return wavelet_data  # just return as is

        # Normalize
        if n_channels == 1:
            # wavelet_data_np -> shape [3, T, W]
            for band_i in range(3):
                band_vals = wavelet_data_np[band_i, :, :]
                min_val = gmin[0, band_i]
                max_val = gmax[0, band_i]
                denom = (max_val - min_val) if (max_val - min_val) != 0 else 1e-8
                wavelet_data_np[band_i, :, :] = (band_vals - min_val) / denom
        else:
            # shape [n_channels, 3, T, W]
            for ch_i in range(n_channels):
                for band_i in range(3):
                    band_vals = wavelet_data_np[ch_i, band_i, :, :]
                    min_val = gmin[ch_i, band_i]
                    max_val = gmax[ch_i, band_i]
                    denom = (max_val - min_val) if (max_val - min_val) != 0 else 1e-8
                    wavelet_data_np[ch_i, band_i, :, :] = (band_vals - min_val) / denom

        return torch.from_numpy(wavelet_data_np)

    def _filter_wavelet_bands(self, eeg_data, mag_data, eeg_channel=13, mag_channel=21, gather_stats_only=False):
        """
        Returns (wavelet_bands_eeg, wavelet_bands_mag).

        If 'all' is specified for eeg_channel or mag_channel, we process each channel 
        in the corresponding data. Then wavelet_bands_eeg / wavelet_bands_mag will have shape:
            [n_eeg_channels, 3, time_points, num_windows] or
            [n_mag_channels, 3, time_points, num_windows]
        respectively.
                OR
        If an integer channel index is specified, the shape is 
            [3, time_points, num_windows]. 

        band_dict indicates the frequency bands computed from cwt:
        - alpha: (8, 13)
        - beta: (13, 30)
        - gamma: (30, 100)

        gather_stats_only (bool): If True, we just compute wavelet data for stats, 
        ignoring any normalization.
        """

        band_dict = {
            'alpha': (8, 13),
            'beta':  (13, 30),
            'gamma': (30, 100),
        }

        wavelet_bands_eeg = None
        wavelet_bands_mag = None

        # Handle EEG data (if it exists)
        if eeg_data is not None:
            n_channels_eeg = eeg_data.shape[0]
            time_points = eeg_data.shape[1]
            num_windows = eeg_data.shape[2]

            # Figure out which channels to process
            if eeg_channel == 'all':
                channels_to_use = range(n_channels_eeg)
            else:
                channels_to_use = [eeg_channel]

            multiple_eeg_channels = (len(channels_to_use) > 1)
            if multiple_eeg_channels:
                wavelet_bands_eeg = np.zeros(
                    (len(channels_to_use), 3, time_points, num_windows),
                    dtype=np.float32
                )
            else:
                wavelet_bands_eeg = np.zeros(
                    (3, time_points, num_windows),
                    dtype=np.float32
                )

            for idx, ch in enumerate(channels_to_use):
                for w in range(num_windows):
                    eeg_signal = eeg_data[ch, :, w].cpu().numpy()
                    eeg_coeffs, freqs = self._compute_wavelet_transform(eeg_signal)

                    for band_i, (fmin, fmax) in enumerate(band_dict.values()):
                        band_mask = (freqs >= fmin) & (freqs <= fmax)
                        # Apply log1p to the absolute values of coefficients
                        band_vals = np.log1p(np.mean(np.abs(eeg_coeffs[band_mask, :]), axis=0))
                        if multiple_eeg_channels:
                            wavelet_bands_eeg[idx, band_i, :, w] = band_vals
                        else:
                            wavelet_bands_eeg[band_i, :, w] = band_vals

            wavelet_bands_eeg = torch.from_numpy(wavelet_bands_eeg)

        # Handle MAG data (if it exists)
        if mag_data is not None:
            n_channels_mag = mag_data.shape[0]
            time_points = mag_data.shape[1]
            num_windows = mag_data.shape[2]

            if mag_channel == 'all':
                channels_to_use = range(n_channels_mag)
            else:
                channels_to_use = [mag_channel]

            multiple_mag_channels = (len(channels_to_use) > 1)
            if multiple_mag_channels:
                wavelet_bands_mag = np.zeros(
                    (len(channels_to_use), 3, time_points, num_windows),
                    dtype=np.float32
                )
            else:
                wavelet_bands_mag = np.zeros(
                    (3, time_points, num_windows), 
                    dtype=np.float32
                )

            for idx, ch in enumerate(channels_to_use):
                for w in range(num_windows):
                    mag_signal = mag_data[ch, :, w].cpu().numpy()
                    mag_coeffs, freqs = self._compute_wavelet_transform(mag_signal)

                    for band_i, (fmin, fmax) in enumerate(band_dict.values()):
                        band_mask = (freqs >= fmin) & (freqs <= fmax)
                        # Apply log1p to the absolute values of coefficients
                        band_vals = np.log1p(np.mean(np.abs(mag_coeffs[band_mask, :]), axis=0))
                        if multiple_mag_channels:
                            wavelet_bands_mag[idx, band_i, :, w] = band_vals
                        else:
                            wavelet_bands_mag[band_i, :, w] = band_vals

            wavelet_bands_mag = torch.from_numpy(wavelet_bands_mag)

        return wavelet_bands_eeg, wavelet_bands_mag

    def _compute_wavelet_transform(self, data_1d, sampling_rate=275, frequencies=None, wavelet='cmor1.5-1.0'):
        """
        Returns (coeffs, freqs). Adjust wavelet parameters to suit your data.

        cwt output shape: (#scales, len(data_1d))
        The frequencies array has length #scales, matching the 0th dimension of coeffs.
        """
        if frequencies is None:
            # e.g. from 1Hz to 100Hz in log space
            frequencies = np.logspace(np.log10(1), np.log10(100), num=60)
        scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
        coeffs, _ = pywt.cwt(data_1d, scales, wavelet)
        return coeffs, frequencies