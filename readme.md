# set up the environment
1. create python environment 
conda create -n py39 python=3.9  
2. install required packages
pip install -r requirements.txt


# Data Preprocessing
1. preprocess.py: used for preprocessing and obtaining necessary features
2. dataset.py: used to create train, val, test data


# Run

## pre-training
python algorithm/ustad/train.py

After training, the metrics, model, and forecast results will be saved in ./output/

## Anomaly detection module (anomaly_detection)

data_funcs.py: 

1. the file contains the function to construct the inference data for test data, to get predicted values and uncertainty estimation.

2. settings_map is used to choose which dataset and under which setting you want to do inference on. When you load the trained model, if it has uncertainty module, UC_mode = 'all'; otherwise, UC_mode = "None" for vanilla transformer.

inference.py:

1. the file loads the trained mode, contucts infernece and finally perform anomaly detection.
