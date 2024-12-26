import os
import torch
import random
import logging
from torch.utils.data import Dataset

class ShardDataLoader:
    def __init__(self, dataset_path, mode='train', 
                 window_size=275, logger=None, verbose=False):
        """
        Args:
            dataset_path (str): Path to the dataset directory.
            mode (str): 'train', 'val', or 'test'.
            window_size (int): Size of the windows (should match the window size used in preprocessing).
            logger (logging.Logger): Logger for logging messages.
            verbose (bool): If True, logs additional details.
        """
        self.dataset_path = dataset_path
        self.mode = mode
        self.window_size = window_size
        self.logger = logger or logging.getLogger(__name__)
        self.verbose = verbose

        self.shard_pairs = self._load_shard_pairs()

    def _load_shard_pairs(self):
        """
        Loads and pairs EEG and MAG shards for the given mode.
        Returns:
            list of tuples: List of (eeg_shard_path, mag_shard_path)
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

            eeg_shard_dir = os.path.join(subject_path, 'EEG_shards')
            mag_shard_dir = os.path.join(subject_path, 'MAG_shards')
            if not os.path.isdir(eeg_shard_dir) or not os.path.isdir(mag_shard_dir):
                if self.verbose:
                    self.logger.warning(f"EEG/MAG shards not found for subject {subject}")
                continue

            eeg_shard_files = [f for f in os.listdir(eeg_shard_dir) if f.endswith('_eeg.pt')]
            mag_shard_files = [f for f in os.listdir(mag_shard_dir) if f.endswith('_mag.pt')]

            # Map run names to files for pairing
            eeg_shards_dict = {os.path.splitext(f)[0].replace('_eeg', ''): os.path.join(eeg_shard_dir, f)
                               for f in eeg_shard_files}
            mag_shards_dict = {os.path.splitext(f)[0].replace('_mag', ''): os.path.join(mag_shard_dir, f)
                               for f in mag_shard_files}

            common_runs = set(eeg_shards_dict.keys()) & set(mag_shards_dict.keys())
            if len(common_runs) == 0:
                if self.verbose:
                    self.logger.warning(f"No common runs found for subject {subject}")
                continue

            for run_name in common_runs:
                eeg_shard_path = eeg_shards_dict[run_name]
                mag_shard_path = mag_shards_dict[run_name]
                shard_pairs.append((eeg_shard_path, mag_shard_path))

        if self.verbose:
            self.logger.info(f"Loaded {len(shard_pairs)} shard pairs for mode '{self.mode}'")

        return shard_pairs

    def shuffle_pairs(self):
        """
        Shuffles the list of shard pairs.
        """
        random.shuffle(self.shard_pairs)

    def prepare_epoch_dataset(self, sample_length):
        """
        Shuffles the shards and returns a Dataset object for the epoch.
        Args:
            sample_length (int): Length of samples to extract from the data.
        Returns:
            Dataset: A Dataset object for the epoch.
        """
        self.shuffle_pairs()
        eeg_tensors = []
        mag_tensors = []

        for eeg_shard_path, mag_shard_path in self.shard_pairs:
            # Load the EEG and MAG shards
            eeg_shard = torch.load(eeg_shard_path)  # shape: [n_channels_eeg, window_size, n_windows]
            mag_shard = torch.load(mag_shard_path)  # shape: [n_channels_mag, window_size, n_windows]

            # Merge along the last dimension (n_windows)
            eeg_shard = eeg_shard.reshape(eeg_shard.shape[0], -1)   # [n_channels_eeg, total_samples]
            mag_shard = mag_shard.reshape(mag_shard.shape[0], -1)   # [n_channels_mag, total_samples]

            eeg_tensors.append(eeg_shard)
            mag_tensors.append(mag_shard)

        # Concatenate all shards along the time dimension
        eeg_tensor = torch.cat(eeg_tensors, dim=1)  # [n_channels_eeg, total_samples_all_shards]
        mag_tensor = torch.cat(mag_tensors, dim=1)  # [n_channels_mag, total_samples_all_shards]

        dataset = EEGMAGDataset(eeg_tensor, mag_tensor, sample_length)
        return dataset

class EEGMAGDataset(Dataset):
    def __init__(self, eeg_tensor, mag_tensor, sample_length):
        """
        Args:
            eeg_tensor (torch.Tensor): EEG data tensor of shape [n_channels_eeg, total_samples]
            mag_tensor (torch.Tensor): MAG data tensor of shape [n_channels_mag, total_samples]
            sample_length (int): Length of each sample in time points
        """
        self.eeg_tensor = eeg_tensor
        self.mag_tensor = mag_tensor
        self.sample_length = sample_length
        self.total_time_points = eeg_tensor.shape[1]
        self.num_samples = self.total_time_points // sample_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.sample_length
        end = start + self.sample_length
        eeg_sample = self.eeg_tensor[:, start:end]  # Shape: [n_channels_eeg, sample_length]
        mag_sample = self.mag_tensor[:, start:end]  # Shape: [n_channels_mag, sample_length]
        return eeg_sample, mag_sample