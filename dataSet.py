from torch.utils.data import Dataset
import os
import logging
import traceback
import shutil
import zipfile
import random
import re
        
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
        downloadURLs (list): List of URLs to download the dataset from.
        logger (logging.Logger): Logger for logging information and debug messages.
        verbose (bool): Flag to set the logger to debug level for verbose output.

    Methods:
        _loadDataSet(): Loads the dataset by downloading and preprocessing it.
        _downloadDataset(datasetPath=None, downloadURLs=None): Downloads the dataset from specified URLs.
        _processDataset(): Processes the downloaded dataset.
    """
    def __init__(self,**kwargs):
        self.toggleLoadData = kwargs.get('toggleLoadData', True)
        self.datasetPath = kwargs.get('dataPath', os.path.join('data','openfmri'))
        defaultDownloadURLs = [ ## links to normalized data of all participants
            "https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/ds000117_R1.0.0_derivatives_sub01-04.zip",
            "https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/ds000117_R1.0.0_derivatives_sub05-08.zip",
            "https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/ds000117_R1.0.0_derivatives_sub09-12.zip",
            "https://s3.amazonaws.com/openneuro/ds000117/ds000117_R1.0.0/compressed/ds000117_R1.0.0_derivatives_sub13-16.zip"
        ]
        self.downloadURLs = kwargs.get('downloadURLs', defaultDownloadURLs)

        self.logger = kwargs.get('logger', None)
        self.verbose = kwargs.get('verbose', False)

        if self.logger is None:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        if self.toggleLoadData:
            self._loadDataSet()

        

    ### logic for loading (downloading, preprocessing) the dataset
    def _loadDataSet(self):
        """
            Downloads and preprocesses dataset from OpenFMRI server.
        """
        self._downloadDataset()
        self._processDataset()
        
    def _downloadDataset(self, datasetPath=None, downloadURLs=None):
        """
        Downloads the dataset from the specified URLs to the given dataset path.

        Parameters:
        - datasetPath (str, optional): The directory path where the dataset will be downloaded.
          Defaults to the instance's datasetPath attribute if not provided.
        - downloadURLs (list, optional): A list of URLs from which to download the dataset.
          Defaults to the instance's downloadURLs attribute if not provided.

        This method checks if the dataset path exists and creates it if it doesn't.
        It then iterates over the download URLs, downloading each file if it doesn't
        already exist in the dataset path. Successfully downloaded files are logged.
        """
        if datasetPath is None:
            datasetPath = self.datasetPath
        if downloadURLs is None:
            downloadURLs = self.downloadURLs
        if not os.path.exists(datasetPath):
            os.makedirs(datasetPath)
        
        self.downloadedFolders = []
        downloaded_marker = os.path.join(datasetPath, '.downloaded')
        
        if not os.path.exists(downloaded_marker):
            self.logger.info(f"Downloading {len(downloadURLs)} files...")
            for url in downloadURLs:
                file_name = os.path.join(datasetPath, url.split('/')[-1])
                if not os.path.exists(file_name):
                    self.logger.info(f"Downloading {file_name}...")
                    try:
                        os.system(f"wget -O {file_name} {url}")
                        self.downloadedFolders.append(file_name)
                    except Exception as e:
                        self.logger.error(f"Error downloading file {file_name}, {e}")
                        traceback.print_exc()
                else:
                    self.logger.info(f"{file_name} already exists, skipping download.")
            
            if len(self.downloadedFolders) == len(downloadURLs):
                with open(downloaded_marker, 'w') as f:
                    f.write('Download completed successfully.')
                self.logger.info(f"Successfully downloaded {len(self.downloadedFolders)} files.")
        else:
            self.logger.info("Dataset already downloaded. Skipping download.") 
        
    def _processDataset(self):
        """
        Iterates through downloaded folders (each holding part of the dataset) 
        and the participant folders within them. Removes all MRI data and moves 
        participant folders to one dataset folder (specified in self.datasetPath).             
        """
        self.participantCount = 0
        datasetFolder = self.datasetPath
        #### unzips folders in datasetfolder (after downloading earlier)
        self._unzipAndRenameInFolder(datasetFolder)
        ## rearranges and randomizes (on a subject level)
        self._arrangeFolders(datasetFolder)
        self._randomizeSubjectData(datasetFolder)
    
    def _unzipAndRenameInFolder(self, folder, remove=False):
        """
        Unzips all zip files in the specified folder and renames the extracted folders.

        This method checks if the folder has already been unzipped by looking for a marker file.
        If not, it unzips all zip files in the folder, optionally removes the zip files after extraction,
        and renames the extracted folders to a standardized format.

        Parameters:
        folder (str): The path to the folder containing zip files to be unzipped.
        remove (bool): If True, the zip files will be deleted after extraction. Defaults to False.

        Returns:
        None
        """
        unzipped_marker = os.path.join(folder, '.unzipped')
        if os.path.exists(unzipped_marker):
            self.logger.info(f"Folder {folder} is already unzipped. Exiting early.")
            return

        assert all(f.endswith('.zip') or f.startswith('.') for f in os.listdir(folder)), f"Not all files in {folder} are zip files or ignored files, please delete non-zip files and re-run."
        zip_file_count = sum(1 for f in os.listdir(folder) if f.endswith('.zip'))
        self.logger.info(f"Unzipping and renaming {zip_file_count} files in folder: {folder}")
        for zipFile in os.listdir(folder):
            ## unzip
            zipFile = os.path.join(folder, zipFile)
            if zipFile.endswith('.zip'):
                self.logger.info(f"Unzipping {zipFile}...")
                try:
                    with zipfile.ZipFile(zipFile, 'r') as zip_ref:
                        zip_ref.extractall(os.path.dirname(zipFile))
                    if remove:
                        os.remove(zipFile)  # Remove the zip file after extraction
                        self.logger.info(f"Unzipped and removed {zipFile}")
                    else:
                        self.logger.info(f"Unzipped {zipFile}")
                except Exception as e:
                    self.logger.error(f"Error unzipping {zipFile}, {e}")
                    traceback.print_exc()
        for i, unzippedFolder in enumerate(f for f in os.listdir(folder) if not f.endswith('.zip') and not f.startswith('.')):
            print(i,unzippedFolder)
            fromName = os.path.join(folder, unzippedFolder)
            toName = os.path.join(folder, f'folder_{i}')
            os.rename(fromName, toName)

        # Create the .unzipped marker file
        with open(unzipped_marker, 'w') as marker_file:
            marker_file.write('')

    def _arrangeFolders(self, datasetFolder):
        """
        Rearranges the files in the datasetFolder after downloading and unzipping.
        
        This method organizes the dataset by moving participant folders into the main dataset folder,
        ensuring that relevant files are not nested within subdirectories. It prepares the dataset
        for randomization and later access by ensuring a consistent folder structure.

        Args:
            datasetFolder (str): The path to the main dataset folder where the unzipped files are located.
        
        Raises:
            AssertionError: If the number of participant folders does not match the expected count.
        """
        arranged_marker = os.path.join(datasetFolder, '.arranged')
        if os.path.exists(arranged_marker):
            self.logger.info(f"Dataset folder {datasetFolder} is already arranged. Exiting early.")
            return

        self.participantCount = 0
        unzippedFolders = [f for f in os.listdir(datasetFolder) if os.path.isdir(os.path.join(datasetFolder, f))]
        for folder in unzippedFolders:
            ### move participant folders into dataset folder (so they are not nested)
            desiredSubFolder = os.path.join(folder, 'derivatives', 'meg_derivatives')
            OpenFMRIDataSet.moveContentsToParentAndDeleteSub(desiredSubFolder, datasetFolder)
        
        ### go through participant folders and move nested relevant files upwards to participant folder
        participantFolders = [f for f in os.listdir(datasetFolder) if not f.startswith('.') and not f.endswith('.zip')]
        for participantFolder in participantFolders:
            self.participantCount += 1
            desiredSubFolder = os.path.join('ses-meg', 'meg')
            participantFolderPath = os.path.join(datasetFolder,participantFolder)
            OpenFMRIDataSet.moveContentsToParentAndDeleteSub(desiredSubFolder, participantFolderPath)

        ### go through participant folders again and renames fif files to standardized format
        participantFolders = [f for f in os.listdir(datasetFolder) if not f.startswith('.') and not f.endswith('.zip')]
        print(participantFolders)
        for participantFolder in participantFolders: 
            participantFolderPath = os.path.join(datasetFolder,participantFolder)
            for file in os.listdir(participantFolderPath):
                match = re.match(r'.*run-(0[1-6]).*', file)
                if match:
                    new_file_name = f'run_{match.group(1)}' + os.path.splitext(file)[1]
                    os.rename(os.path.join(participantFolderPath, file), os.path.join(participantFolderPath, new_file_name))

        ### at this stage dataset folder should contain n folders (with n being number of participants)
        non_zip_non_dot_folders = [f for f in os.listdir(datasetFolder) if not f.startswith('.') and not f.endswith('.zip')]
        assert len(non_zip_non_dot_folders) == self.participantCount, f"ERROR: Dataset folder contains {len(non_zip_non_dot_folders)} folders, but {self.participantCount} (participant) folders are expected."

        # Create the .arranged marker file
        with open(arranged_marker, 'w') as marker_file:
            marker_file.write('')
    
    def _randomizeSubjectData(self, train_percentage=70, val_percentage=20):
        """
        Randomizes and splits participant data into training, validation, and test sets.

        This method shuffles the participant folders and divides them into three subsets:
        training, validation, and test, based on the specified percentages. The folders
        are then moved into corresponding subdirectories within the dataset path.

        Args:
            train_percentage (int): The percentage of participant data to allocate to the training set.
            val_percentage (int): The percentage of participant data to allocate to the validation set.

        Raises:
            ValueError: If the sum of train_percentage and val_percentage exceeds 100.
        """
        ### split participant folders into train, test, and val subfolders        
        participantFolders = os.listdir(self.datasetPath)
        random.shuffle(participantFolders)

        total_participants = len(participantFolders)
        train_count = int(total_participants * train_percentage / 100)
        val_count = int(total_participants * val_percentage / 100)
        

        train_folder = os.path.join(self.datasetPath, 'train')
        val_folder = os.path.join(self.datasetPath, 'val')
        test_folder = os.path.join(self.datasetPath, 'test')

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        for i, participantFolder in enumerate(f for f in participantFolders if f.endswith('.zip') or f.startswith('.')):
            participantFolderPath = os.path.join(self.datasetPath, participantFolder)
            if i < train_count:
                shutil.move(participantFolderPath, train_folder)
            elif i < train_count + val_count:
                shutil.move(participantFolderPath, val_folder)
            else:
                shutil.move(participantFolderPath, test_folder)

    def __getitem__(self, index):
        pass
 
    @staticmethod
    def moveContentsToParentAndDeleteSub(subfolder, parentfolder, expectedContentCount = None):
        """
        Moves all contents from a subfolder to its parent folder and deletes the subfolder.

        Parameters:
        subfolder (str): The name (relative path) of the subfolder whose contents are to be moved.
        parentfolder (str): The absolute path to the parent folder where contents will be moved.

        Returns:
        None
        """
        subfolder_path = os.path.join(parentfolder, subfolder)
        if expectedContentCount is not None:
            actualContentCount = len(os.listdir(subfolder_path))
            assert actualContentCount == expectedContentCount, (
                f"Expected {expectedContentCount} items, but found {actualContentCount} in {subfolder_path}"
            )
        # Move contents of subfolder to parentfolder
        for item in os.listdir(subfolder_path):
            item_path = os.path.join(subfolder_path, item)
            shutil.move(item_path, parentfolder)
        
        # Delete all folders between parentfolder and subfolder
        current_path = subfolder_path
        while current_path != parentfolder:
            parent_path = os.path.dirname(current_path)
            os.rmdir(current_path)
            current_path = parent_path
        


    


    


