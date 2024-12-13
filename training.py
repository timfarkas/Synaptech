from dataset.dataSet import OpenFMRIDataSet
from dotenv import load_dotenv
import os



load_dotenv()
dataset_path = os.getenv("DATASET_PATH")


### init dataset this will automatically download and preprocess the dataset if run for the first time
train_dataset = OpenFMRIDataSet(mode='train',datasetPath=dataset_path)
val_dataset = OpenFMRIDataSet(mode='val',datasetPath=dataset_path,loadData=False)
test_dataset = OpenFMRIDataSet(mode='test',datasetPath=dataset_path,loadData=False) 


