# set up the environment
1. create python environment 
conda create -n py39 python=3.9  
2. install required packages
pip install -r requirements.txt


# Data Preprocessing
1. preprocess_{datasetname}.py: used for preprocessing and obtaining necessary features
2. dataset_{datasetname}.py: used to create train, val, test data

## datasetnames:
1. HighPrev: this version can be used for format similar to NUMOSIM dataset

# Run

## pre-training
python algorithm/ustad/train.py

After training, the metrics, model, and forecast results will be saved in ./output/

## Anomaly detection

# AD module (AD_prediction_error_uc)
single_model.py: 

1. the file construct the inference data (if not already constructed), loads the trained model and do inference. 

3. settings_map is used to choose which dataset and under which setting you want to do inference on. When you load the trained model, if it has uncertainty module, UC_mode = 'all'; otherwise, UC_mode = "None" for vanilla transformer.
