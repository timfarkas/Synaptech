# Enhancing EEG Accuracy via cross modal MEG inference.
A transformer model trained on simultaneous EEG and MEG recordings to achieve MEG-like super-resolution on EEG headsets.

[]Demo[]

## Overview 🔎
Both MEG and EEG signals originate from the net effect of ionic currents flowing in the dendrites of neurons. Whilst EEG measures electric fields and MEG measures changes in magnetic fields, one directly derives from another. One advantage of MEG over EEG is that the magnetic fields are not distorted by the intervening organic matter, as is the case with electric fields measured by EEG. Thus, MEG offers better spatial resolution than EEG. On the other hand, MEG systems are considerably more expensive than EEG systems, bulky and not portable. The isomorphic corelation in EEG & MEG's signal caused us to hypothesise that deep learning models might be able to use MEG signal to inform better EEG predictions if trained on simultaneouse recording of both. 

Another interesting interpretatiton of this work is that by mapping EEG signal -> MEG signal the model learns a function quantifying the extent to which the scull distorts the electric field!!!

- This code aims to provide developers and researchers a tool to train their own transformer to enhance real world EEG accuracy.

---

## Set-up 🔧
Create anaconda environment
```
conda env create -f environment.yml
conda activate synaptech_env
```

Install dependencies
```
pip install -r requirements.txt
```

Set up .env to point to dataset
```
DATASET_PATH="openfrmi/data"
```

Start your first training run!
```
python training.py
```
(first run init will take some time as it will automatically load, prepare, and preprocess the dataset)

Train the model: 
## Run the model 💥🏃‍♂️🔥
To use the application:
1. Run ```main.py``` to feed some validation data into the transformer and see the model at work..
2. Enjoy :) Go save the world!


---

## Technical Overview 
EEG Channel Count: 74
MEG Channel Count: 306
Frame Count: 6042300 (in test dataset)
