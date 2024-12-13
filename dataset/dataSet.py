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

        self.eegValuesType = kwargs.get('eegValuesType', np.float16)
        self.megValuesType = kwargs.get('megValuesType', np.float16)

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

    def __len__(self):
        return self.totalWindows

    def __getitem__(self, idx):
        runningIndex = 0
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
                    return data[self.eeg_indices], data[self.meg_indices]
                else:
                    runningIndex += count
        # If we get here, idx is beyond all runs
        raise IndexError("Index out of range")

