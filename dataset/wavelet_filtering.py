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
        mode='train',
        eeg_channel=13,
        mag_channel=21
    ):
        """
        Applies wavelet transforms to EEG and MEG data.

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
        1) Always process 'train', 'val', 'test' – if self.mode=None or 'all', we do all three.
        2) Collect all .pt shards (EEG & MAG) from all subjects in each mode.
        3) Display exactly one progress bar per mode with the message:
        "Computing wavelet transform for {mode}:"
        4) Load each PT file, apply wavelet transform, and save to disk.
        """

        # We'll always process these three modes.
        # If the user specified 'train', 'val', or 'test', we'll process only that one.
        if self.mode in ['train', 'val', 'test']:
            modes_to_process = [self.mode]
        else:
            modes_to_process = ['train', 'val', 'test']

        for mode in modes_to_process:
            mode_path = Path(self.dataset_path) / mode
            if not mode_path.exists():
                print(f"Skipping non-existent mode folder {mode_path}")
                continue

            # Gather all PT files (EEG & MAG) for this mode
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

            # Single progress bar for all EEG+MAG shards in this mode
            from tqdm import tqdm
            for shard_type, pt_file, output_dir in tqdm(pt_files, desc=f"Computing wavelets for [{mode}]"):
                data = torch.load(pt_file)
                if shard_type == "EEG":
                    wavelet_eeg, _ = self._filter_wavelet_bands(
                        data, 
                        None,  
                        self.eeg_channel,  
                        None
                    )
                    out_name = pt_file.stem + "_wavelet.pt"
                    torch.save(wavelet_eeg.float(), output_dir / out_name)
                else:  # shard_type == "MAG"
                    _, wavelet_mag = self._filter_wavelet_bands(
                        None, 
                        data,  
                        None, 
                        self.mag_channel
                    )
                    out_name = pt_file.stem + "_wavelet.pt"
                    torch.save(wavelet_mag.float(), output_dir / out_name)

    def _filter_wavelet_bands(self, eeg_data, mag_data, eeg_channel=13, mag_channel=21):
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
        """


        band_dict = {
            'alpha': (8, 13),
            'beta': (13, 30),
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
                        band_vals = np.mean(np.abs(eeg_coeffs[band_mask, :]), axis=0)
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
                        band_vals = np.mean(np.abs(mag_coeffs[band_mask, :]), axis=0)
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
        import numpy as np
        import pywt

        if frequencies is None:
            frequencies = np.logspace(np.log10(1), np.log10(100), num=60)
        scales = pywt.frequency2scale(wavelet, frequencies / sampling_rate)
        coeffs, _ = pywt.cwt(data_1d, scales, wavelet)
        return coeffs, frequencies