from dataset.dataSet import OpenFMRIDataSet

### init dataset this will automatically download and preprocess the dataset if run for the first time
train_dataset = OpenFMRIDataSet(mode='train')
val_dataset = OpenFMRIDataSet(mode='val',loadData=False)
test_dataset = OpenFMRIDataSet(mode='test',loadData=False) 




