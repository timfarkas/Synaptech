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

    Methods:
        _loadDataSet(): Loads the dataset by downloading and preprocessing it.
        _downloadDataset(datasetPath=None, downloadURLs=None): Downloads the dataset from specified URLs.
        _processDataset(): Processes the downloaded dataset.
    """
    def __init__(self,**kwargs):
        self.datasetPath = kwargs.get('dataPath', os.path.join('data','openfmri'))
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
                                           processImmediately=self.toggleProcessData, 
                                           processingMode=self.processingMode,
                                           logger = self.logger, 
                                           verbose = self.verbose, 
                                           downloadURLs = self.downloadURLs)
         ### change datasetPath to point to train, val, or test after loading the data
        self.datasetPath = os.path.join(self.datasetPath,self.mode)
        
        self.logger.info(f"Loading dataset in {self.mode} mode...")
        #self._countFrames()

    def __getitem__(self, index):
        pass
 
