from torch.utils.data import Dataset
import os
import logging
import traceback
import shutil
import zipfile
import random
import re
import numpy as np
import contextlib
import mne
from dataset.utilities import DatasetDownloader
import warnings
import pickle as pkl


"""
Source:
This data is obtained from the OpenfMRI database. 
Its accession number is ds000117.
"""
class OpenFMRIDataSet(Dataset):
    """
    A dataset class for handling OpenFMRI data.

    This class extends the PyTorch Dataset class and provides functionality
    to download, preprocess, and manage OpenFMRI datasets. It supports
    toggling data loading, specifying custom dataset paths, and custom
    download URLs.

    Attributes:
        toggleLoadData (bool): Flag to toggle if data should be loaded upon instantiation.
        datasetPath (str): Path to the dataset directory.
        mode (str): The mode of the dataset, can be 'train', 'val', or 'test'. Defaults to train.
        downloadURLs (list): List of URLs to download the dataset from.
        memoryBound (int): The memory bound in GB.
        VRAMBound (int): The VRAM bound in GB.
        logger (logging.Logger): Logger for logging information and debug messages.
        verbose (bool): Flag to set the logger to debug level for verbose output.
        
        if processingMode == raw:
            sampleRate (int): The sample rate of the dataset in Hz.
            windowLength_ms (float): The length of the window in milliseconds.
            windowLength_frames (int): The length of the window in frames, calculated as windowLength_ms/1000 * sampleRate.

    Methods:
        _loadDataSet(): Loads the dataset by downloading and preprocessing it.
        _downloadDataset(datasetPath=None, downloadURLs=None): Downloads the dataset from specified URLs.
        _processDataset(): Processes the downloaded dataset.
    """
    def __init__(self,**kwargs):
        self.datasetPath = kwargs.get('datasetPath', os.path.join('/srv','synaptech_openfmri'))
        self.mode = kwargs.get('mode', 'train')

        self.eegValuesType = kwargs.get('eegValuesType', np.float32)
        self.megValuesType = kwargs.get('megValuesType', np.float32)

        self.logger = kwargs.get('logger', None)
        self.verbose = kwargs.get('verbose', False)

        if self.logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        self.toggleLoadData = kwargs.get('loadData', True)
        self.downloadURLs = kwargs.get('downloadURLs', None)
        self.toggleProcessData = kwargs.get('processData', True)
        self.processingMode = str(kwargs.get('processingMode', 'raw')).lower()

        if self.toggleLoadData:
            downloader = DatasetDownloader(downloadAndPrepareImmediately=True,
                                            datasetPath = self.datasetPath,
                                           processImmediately=self.toggleProcessData, 
                                           processingMode=self.processingMode,
                                           logger = self.logger, 
                                           verbose = self.verbose, 
                                           downloadURLs = self.downloadURLs)
         
        self.cacheDirectory = os.path.join(self.datasetPath,'.cache')
        ### change datasetPath to point to train, val, or test after loading the data
        self.datasetPath = os.path.join(self.datasetPath,self.mode)

        ### FIF MEG AND EEG INDICES (hardcoded for now)
        self.meg_indices = list(range(2, 306, 3))
        self.eeg_indices = list(range(306, 380, 1))

        self.logger.info(f"\n\nLoading dataset in {self.mode} mode...")
        sampleRate = kwargs.get('sampleRate', 1100)
        windowLength = kwargs.get('windowLength', 250)
        windowOverlap = kwargs.get('windowOverlap', 0.3)

        self._countWindows(self.processingMode, 
                           sampleRate=sampleRate, 
                           windowLength=windowLength, 
                           windowOverlap=windowOverlap)
        self.totalRuns = len(self.fif_files)
        self.emovs ### calculate emovs


    def _countWindows(self, mode, sampleRate=1100, windowLength=250, windowOverlap=0.3):
        if mode != "raw":
            raise NotImplementedError("Non-raw sequences are not yet supported.")
        
        self.sampleRate = sampleRate
        self.windowLength_ms = windowLength
        self.windowOverlap = windowOverlap
        self.windowLength_frames = int(self.windowLength_ms/1000 * self.sampleRate)
        
        self.logger.info(f"Window length (No. frames): {self.windowLength_frames}")

        assert self.datasetPath.split(os.sep)[-1] in ["train", "val", "test"], f"Unexpected dataset path {self.datasetPath}, did you call buildWindows before preparing and processing dataset?"

        # build dict with semantic structure: {'sub_01': ['run01', 'run02', ...], 'sub_02': ['run01', 'run02'],...}  
        self._participantRunsDict = {}
        self._participantRunsArray = []

        for i, subjectFolder in enumerate(os.listdir(self.datasetPath)):
            self._participantRunsDict[subjectFolder] = []
            self._participantRunsArray.append([])

            subjectRuns = []
            for file in os.listdir(os.path.join(self.datasetPath, subjectFolder)):
                if ".fif" in file:
                    self._participantRunsDict[subjectFolder].append(file)
                    subjectRuns.append(file)
                    self._participantRunsArray[i].append(os.path.join(subjectFolder, file))

        #self.logger.debug(self._participantRunsDict)


        #### LOAD CACHED frame counts, window counts, window indices IF EXISTENT
        cacheFileName_frameCounts = f"openfrmi_{self.mode}_{self.windowLength_frames}_raw_frameCounts.cache"
        self._frameCountCacheFile = os.path.join(self.cacheDirectory, cacheFileName_frameCounts)

        cacheFileName_windowCounts = f"openfrmi_{self.mode}_{self.windowLength_frames}_raw_windowCounts.cache"
        self._windowCountCacheFile = os.path.join(self.cacheDirectory, cacheFileName_windowCounts)
        
        cacheFileName_windowIndices = f"openfrmi_{self.mode}_{self.windowLength_frames}_raw_windowIndices.cache"
        self._windowIndicesCacheFile = os.path.join(self.cacheDirectory, cacheFileName_windowIndices)

        if (os.path.exists(self._frameCountCacheFile) and 
            os.path.exists(self._windowCountCacheFile) and 
            os.path.exists(self._windowIndicesCacheFile)):
            
            with open(self._frameCountCacheFile, 'rb') as f:
                self._participantFrameCounts = pkl.load(f)
            
            with open(self._windowCountCacheFile, 'rb') as f:
                self._participantWindowCounts = pkl.load(f)
            
            with open(self._windowIndicesCacheFile, 'rb') as f:
                self._participantWindowIndices = pkl.load(f)
            
            self.totalWindows = sum([count for sublist in self._participantWindowCounts for count in sublist])
            self.logger.info(f"Cache files found. {self.totalWindows} loaded from cache.")
            return 

        # build frame count array with semantic structure: [[run01_frames, run02_frames, ...],  ### first subject
        #                                                   [run01_frames, run02_frames],      ### second subject
        #                                                   ...]  ## etc
        self._participantFrameCounts = []
        for i, subjectFolder in enumerate(self._participantRunsDict):
            self._participantFrameCounts.append([])
            for runFile in self._participantRunsDict[subjectFolder]:
                fif_file = os.path.join(self.datasetPath, subjectFolder, runFile)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                        raw = mne.io.read_raw_fif(fif_file, preload=True)
                    self._participantFrameCounts[i].append(raw.n_times)

        #self.logger.debug(self._participantFrameCounts)

        # build window count array with semantic structure:     [[run01_windows, run02_windows, ...],  ### first subject
        #                                                       [run01_windows, run02_windows],      ### second subject
        #                                                       ...] ## etc
        #
        # build window indices array with semantic structure:   [[run01_start_indices_array, run02_start_indices_array, ...],  ### first subject
        #                                                       [run01_start_indices_array, run02_start_indices_array],      ### second subject
        #                                                       ...] ## etc
        self._participantWindowCounts = []
        self._participantWindowIndices = []
        
        for i, subjectFolder in enumerate(self._participantRunsDict):
            self._participantWindowCounts.append([])
            self._participantWindowIndices.append([])
            for j, runFile in enumerate(self._participantRunsDict[subjectFolder]):
                run_frame_count = self._participantFrameCounts[i][j]
                
                run_window_count = int((run_frame_count - self.windowLength_frames) / (self.windowLength_frames * (1 - self.windowOverlap)) + 1)
                self._participantWindowCounts[i].append(run_window_count)
                
                start_indices = []
                for k in range(run_window_count):
                    start_index = int(k * self.windowLength_frames * (1 - self.windowOverlap))
                    start_indices.append(start_index)
                
                self._participantWindowIndices[i].append(start_indices)

        # Create and write to the frame count cache file
        os.makedirs(os.path.dirname(self._frameCountCacheFile), exist_ok=True)
        frame_count_file = open(self._frameCountCacheFile, 'wb')
        pkl.dump(self._participantFrameCounts, frame_count_file)
        frame_count_file.close()
        
        # Create and write to the window count cache file
        window_count_file = open(self._windowCountCacheFile, 'wb')
        pkl.dump(self._participantWindowCounts, window_count_file)
        window_count_file.close()
        
        # Create and write to the window indices cache file
        window_indices_file = open(self._windowIndicesCacheFile, 'wb')
        pkl.dump(self._participantWindowIndices, window_indices_file)
        window_indices_file.close()

        self.totalWindows = sum([count for sublist in self._participantWindowCounts for count in sublist])
        self.logger.info(f"Found {self.totalWindows} windows...")

    @property
    def fif_files(self):
        """
        Gathers and returns a list of FIF files for each subject and run.

        This property checks if the attribute '_fif_files' already exists. If it does, 
        it returns the existing list. 
        Otherwise, it constructs the list by iterating 
        over the subjects and their corresponding run files in the '_participantRunsDict'. 
        Each FIF file path is constructed by joining the dataset path, subject, and run file.
        
        Returns:
            list: A list of file paths to the FIF files.
        """
        if not hasattr(self, '_fif_files'):
            self._fif_files = []
            for subject in self._participantRunsDict:
                for run_file in self._participantRunsDict[subject]:
                    fif_file = os.path.join(self.datasetPath, subject, run_file)
                    self._fif_files.append(fif_file)
        return self._fif_files

    def _gatherSensorCoordinates(self):
        """
        Gathers sensor coordinates and channel names for EEG and MEG data.

        This method processes each run's FIF file to extract sensor coordinates and channel names
        for both EEG and MEG data. It also retrieves the transformation matrix from device to head
        for MEG data. The method suppresses warnings and redirects output to avoid cluttering the console.

        Returns:
            tuple: A tuple containing:
                - A tuple of lists: (all_eeg_channel_locs, all_eeg_channel_names)
                  where all_eeg_channel_locs is a list of EEG channel locations for each run,
                  and all_eeg_channel_names is a list of EEG channel names for each run.
                - A tuple of lists: (all_meg_channel_locs, all_meg_channel_names)
                  where all_meg_channel_locs is a list of MEG channel locations for each run,
                  and all_meg_channel_names is a list of MEG channel names for each run.
                - A tuple of lists: (all_channel_locs, all_channel_names)
                  where all_channel_locs is a list of all channel locations for each run,
                  and all_channel_names is a list of all channel names for each run.
                - list: all_meg_transforms, a list of transformation matrices from device coordinates to head coordinates
                  for each run (MEG device -> EEG coordinates).
        """
        all_channel_names = []
        all_eeg_channel_names = []
        all_meg_channel_names = []

        all_channel_locs = []
        all_eeg_channel_locs = []
        all_meg_channel_locs = []

        all_meg_transforms = []

        for run in self.fif_files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):    
                    raw = mne.io.read_raw_fif(run, preload=False)
            channels = raw.info['chs']

            all_meg_transforms.append(raw.info['dev_head_t']) ## get transformation matrix device->head

            channel_locs = []
            channel_names = []
            eeg_channel_names = []
            meg_channel_names = []
            
            for channel in channels:
                channel_names.append(channel['ch_name'])
                channel_locs.append([channel['loc'][0],channel['loc'][1],channel['loc'][2]])
            
            meg_channel_locs = np.array(channel_locs)[self.meg_indices]
            eeg_channel_locs = np.array(channel_locs)[self.eeg_indices]
            
            eeg_channel_names = np.array(channel_names)[self.eeg_indices].tolist()
            meg_channel_names = np.array(channel_names)[self.meg_indices].tolist()
            
            all_eeg_channel_locs.append(eeg_channel_locs)
            all_meg_channel_locs.append(meg_channel_locs)
            all_eeg_channel_names.append(eeg_channel_names)
            all_meg_channel_names.append(meg_channel_names)
            all_channel_locs.append(channel_locs)
            all_channel_names.append(channel_names)   

        return (all_eeg_channel_locs,all_eeg_channel_names), (all_meg_channel_locs, all_meg_channel_names), (all_channel_locs,all_channel_names), all_meg_transforms

    @property 
    def emovs(self):
        """
        Getter for the EEG-MEG Offset Vectors (EMOVs).

        This property calculates and returns the EEG-MEG Offset Vectors for each run.
        The EMOVs are computed as the offset vectors between EEG and transformed MEG locations.

        Returns:
            list: A list of shape (n, 3) where n is the number of runs. Each element is a numpy array
                  representing the offset vector for a specific run.
        """
        if not hasattr(self, '_emovs'):
            _, _, (all_channel_locs,all_channel_names), self._meg_transforms = self._gatherSensorCoordinates()

            self._emovs = []
            
            for run_channel_locs, meg_transform in zip(all_channel_locs,self._meg_transforms):
                self._emovs.append(OpenFMRIDataSet.find_emov(run_channel_locs,meg_transform)) 
            assert len(self._emovs) == self.totalRuns, "ERROR, length of list _emovs not equal to number of all runs"
        return self._emovs
    
    @property
    def meg_transforms(self):
        """
        Getter for the MEG transformation matrices.

        This property retrieves the transformation matrices for MEG data.
        If the transformation matrices are not already computed, it gathers
        the sensor coordinates and extracts the transformation matrices.

        Returns:
            list: A list of length n of transformation matrices for each run.
        """
        if not hasattr(self, '_meg_transforms'):
            _, _, _, self._meg_transforms = self._gatherSensorCoordinates()
            assert len(self._meg_transforms) == self.totalRuns, "ERROR, length of list MEG transforms not equal to number of all runs"
        
        return self._meg_transforms

    @staticmethod          
    def transformMEGcoordinates(megCoordinates, megTransform):
        """
        Apply a transformation matrix to MEG coordinates.

        This static method applies a given transformation matrix to the provided
        MEG coordinates using MNE's transformation utilities.

        Parameters:
        - megCoordinates: The MEG coordinates to be transformed.
        - megTransform: The transformation matrix (dev->head, i.e. MEG->EEG) to apply to the MEG coordinates.

        Returns:
        - Transformed MEG coordinates.
        """
        return mne.transforms.apply_trans(megTransform, megCoordinates)

    def __len__(self):
        return self.totalWindows

    def __getitem__(self, idx):
        runningIndex = 0
        absRunIndex = 0
        for subjectIndex in range(len(self._participantWindowCounts)):
            for runIndex in range(len(self._participantWindowCounts[subjectIndex])):
                count = self._participantWindowCounts[subjectIndex][runIndex]
                # The runs cover indices [runningIndex, runningIndex + count - 1]
                if idx < runningIndex + count:
                    # We found the correct run
                    windowIndex = idx - runningIndex
                    window_starting_index = self._participantWindowIndices[subjectIndex][runIndex][windowIndex]
                    fif_file = os.path.join(self.datasetPath, self._participantRunsArray[subjectIndex][runIndex])
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            raw = mne.io.read_raw_fif(fif_file, preload=False)
                    data = raw[:, window_starting_index:window_starting_index+self.windowLength_frames][0]
                    
                    ## returns tuple eeg_data, meg_data (transformed), EMOV
                    eeg_data = data[self.eeg_indices].astype(self.eegValuesType)
                    meg_data = data[self.meg_indices].astype(self.megValuesType)
                    
                    # Normalize EEG and MEG data
                    eeg_data = (eeg_data - np.mean(eeg_data)) / np.std(eeg_data)
                    meg_data = (meg_data - np.mean(meg_data)) / np.std(meg_data)
                    #print(f"mean: {np.mean(meg_data)}, std: {np.std(meg_data)}")

                    ### Append three EMOV rows to eeg_data
                    emov = self.emovs[absRunIndex]
                    emovRows = np.array([np.full(self.windowLength_frames, value) for value in emov])

                    return np.concatenate((eeg_data,emovRows)), meg_data
                else:
                    runningIndex += count
                    absRunIndex += 1
        # If we get here, idx is beyond all runs
        raise IndexError("Index out of range")

    @staticmethod
    def find_emov(run_channel_locs, run_meg_eeg_transform):
        """
        Calculate the EEG-MEG Offset Vector (EMOV).

        This function computes the offset vector between a reference EEG electrode
        and a reference MEG electrode after applying a transformation to the MEG location.

        Parameters:
        - run_channel_locs: List or array of channel locations for a specific run.
        - run_meg_eeg_transform: Transformation matrix to be applied to the MEG location.

        Returns:
        - A numpy array representing the offset vector between the EEG and transformed MEG locations.
        """
        eeg_reference_electrode_index = 379
        meg_reference_electrode_index = 236
        
        eeg_loc = run_channel_locs[eeg_reference_electrode_index]
        meg_loc = run_channel_locs[meg_reference_electrode_index]
        meg_loc = mne.transforms.apply_trans(run_meg_eeg_transform, meg_loc)

        return np.array(eeg_loc) - np.array(meg_loc)