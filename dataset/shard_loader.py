import os
import torch
import random
import logging
from torch.utils.data import Dataset

class ShardDataLoader:
    def __init__(self, dataset_path, mode='train', 
                 window_size=275, logger=None, verbose=False,
                 wavelet=False):
        """
        Args:
            dataset_path (str): Path to the dataset directory.
            mode (str): 'train', 'val', or 'test'.
            window_size (int): Size of windows (unused for wavelet shards, but kept for continuity).
            logger (logging.Logger): Logger for logging messages.
            verbose (bool): If True, logs additional details.
            wavelet (bool): If True, load from EEG_WAVELET_shards/MAG_WAVELET_shards 
                            instead of EEG_shards/MAG_shards.

        Note:
          - Normal shards shape: [n_channels_eeg, window_size, n_windows] or [n_channels_mag, window_size, n_windows].
          - Wavelet shards shape: [3, time_points, n_windows] or [n_channels_wavelet, time_points, n_windows].
        """
        self.dataset_path = dataset_path
        self.mode = mode
        self.window_size = window_size
        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose
        self.wavelet = wavelet

        self.shard_pairs = self._load_shard_pairs()
        if verbose:
            self.logger.info(f"ShardDataLoader init: wavelet={self.wavelet}, mode={self.mode}, len(shard_pairs)={len(self.shard_pairs)}")

    def _load_shard_pairs(self):
            """
            Loads and pairs EEG and MAG shards for the given mode.
            """
            shard_pairs = []
            mode_path = os.path.join(self.dataset_path, self.mode)
            
            if not os.path.isdir(mode_path):
                self.logger.error(f"Mode path does not exist: {mode_path}")
                return shard_pairs

            subjects = sorted(os.listdir(mode_path))
            
            for subject in subjects:
                subject_path = os.path.join(mode_path, subject)
                if not os.path.isdir(subject_path):
                    continue

                # Decide which subfolder to load based on wavelet flag
                if self.wavelet:
                    eeg_shard_dir = os.path.join(subject_path, 'EEG_WAVELET_shards')
                    mag_shard_dir = os.path.join(subject_path, 'MAG_WAVELET_shards')
                else:
                    eeg_shard_dir = os.path.join(subject_path, 'EEG_shards')
                    mag_shard_dir = os.path.join(subject_path, 'MAG_shards')

                if not os.path.isdir(eeg_shard_dir) or not os.path.isdir(mag_shard_dir):
                    if self.verbose:
                        self.logger.warning(f"EEG/MAG shards not found for subject {subject}")
                    continue

                # Identify EEG and MAG shard files
                if self.wavelet:
                    eeg_suffix = "_eeg_wavelet.pt"
                    mag_suffix = "_mag_wavelet.pt"
                else:
                    eeg_suffix = "_eeg.pt"
                    mag_suffix = "_mag.pt"

                eeg_shard_files = [f for f in os.listdir(eeg_shard_dir) if f.endswith(eeg_suffix)]
                mag_shard_files = [f for f in os.listdir(mag_shard_dir) if f.endswith(mag_suffix)]

                # Build dictionaries for matching run names
                if self.wavelet:
                    def get_run_name(filename):
                        # Extract just the run number part (e.g., "run_01" from "run_01_eeg_wavelet.pt")
                        return "_".join(filename.split("_")[:2])

                    eeg_shards_dict = {
                        get_run_name(f): os.path.join(eeg_shard_dir, f)
                        for f in eeg_shard_files
                    }
                    mag_shards_dict = {
                        get_run_name(f): os.path.join(mag_shard_dir, f)
                        for f in mag_shard_files
                    }
                else:
                    eeg_shards_dict = {
                        os.path.splitext(f)[0].replace('_eeg', ''): os.path.join(eeg_shard_dir, f)
                        for f in eeg_shard_files
                    }
                    mag_shards_dict = {
                        os.path.splitext(f)[0].replace('_mag', ''): os.path.join(mag_shard_dir, f)
                        for f in mag_shard_files
                    }

                # Pair up shards that share the same run name
                common_runs = set(eeg_shards_dict.keys()) & set(mag_shards_dict.keys())
                
                if len(common_runs) == 0:
                    if self.verbose:
                        self.logger.warning(f"No common runs found for subject {subject}")
                    continue

                for run_name in common_runs:
                    shard_pairs.append((eeg_shards_dict[run_name], mag_shards_dict[run_name]))

            return shard_pairs

    def shuffle_pairs(self):
        """
        Shuffle the list of shard pairs in-place.
        """
        random.shuffle(self.shard_pairs)
        if self.verbose:
            self.logger.info(f"Shuffled shard pairs. New first pair: {self.shard_pairs[0] if self.shard_pairs else 'None'}")

    def prepare_epoch_dataset(self, sample_length):
        """
        Shuffle the shards and return a Dataset object for the epoch.
        """
        self.shuffle_pairs()

        eeg_tensors = []
        mag_tensors = []

        for i, (eeg_shard_path, mag_shard_path) in enumerate(self.shard_pairs):
            try:
                eeg_shard = torch.load(eeg_shard_path)
                mag_shard = torch.load(mag_shard_path)

                print(f"[DEBUG] Loading pair {i}:")
                print(f"  EEG: {os.path.basename(eeg_shard_path)}, shape={eeg_shard.shape}")
                print(f"  MAG: {os.path.basename(mag_shard_path)}, shape={mag_shard.shape}")
                print(f"  EEG mean={eeg_shard.mean():.3f}, MAG mean={mag_shard.mean():.3f}")

                # Reshape each shard to [channels, -1]
                eeg_shard = eeg_shard.reshape(eeg_shard.shape[0], -1)
                mag_shard = mag_shard.reshape(mag_shard.shape[0], -1)

                eeg_tensors.append(eeg_shard)
                mag_tensors.append(mag_shard)

            except Exception as e:
                print(f"[ShardDataLoader] Error loading shard pair: {str(e)}")
                continue

        # Concatenate all shards
        if eeg_tensors and mag_tensors:
            eeg_tensor = torch.cat(eeg_tensors, dim=1)
            mag_tensor = torch.cat(mag_tensors, dim=1)
        else:
            print("[ShardDataLoader] No valid tensors to concatenate!")
            return None

        dataset = EEGMAGDataset(eeg_tensor, mag_tensor, sample_length)
        return dataset


class EEGMAGDataset(Dataset):
    def __init__(self, eeg_tensor, mag_tensor, sample_length):
        """
        Args:
            eeg_tensor (torch.Tensor): shape [n_channels_eeg, total_samples]
            mag_tensor (torch.Tensor): shape [n_channels_mag, total_samples]
            sample_length (int): chunk length in time points for each sample
        """
        self.eeg_tensor = eeg_tensor
        self.mag_tensor = mag_tensor
        self.sample_length = sample_length

        if eeg_tensor is None or mag_tensor is None:
            self.total_time_points = 0
        else:
            self.total_time_points = eeg_tensor.shape[1]

        self.num_samples = 0
        if self.total_time_points > 0:
            self.num_samples = self.total_time_points // sample_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.sample_length
        end = start + self.sample_length

        eeg_sample = self.eeg_tensor[:, start:end]  # [n_channels_eeg, sample_length]
        mag_sample = self.mag_tensor[:, start:end]  # [n_channels_mag, sample_length]
        return eeg_sample, mag_sample
