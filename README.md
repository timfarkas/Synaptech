# Enhancing EEG Accuracy via cross modal MEG inference.
A transformer model trained on simultaneous EEG and MEG recordings to achieve near-perfect MEG-like accuracy on EEG headsets.

[]Demo[]

## Overview 🔎
MEG and EEG share not only temporal resolution, but also similar electrical signal features. This makes them well-suited for transformer models, which can accurately learn their dependancies and predict MEG signals from EEG data, thereby enhancing the performance and precision of EEG-based BCIs.
This code aims to provide developers and researchers a tool to train their own transformer to enhance real world EEG accuracy.

---

## Set-up 🔧
Create anaconda environment
```
conda create -n synaptech_env python=3.10 -y 
conda activate synaptech_env
```

Install dependencies
```
pip install -r requirements.txt
```

Download & process the openfmri datasets:
```
cd data_magic
python training.py
```
(will take some time)

Train the model: 
## Run the model 💥🏃‍♂️🔥
To use the application:
1. Run ```main.py``` to feed some validation data into the transformer and see the model at work..
2. Enjoy :) Go save the world!
