# Mitigating Enhanced Application Fingerprinting on Tor: A Preliminary Study
A proof of concept to: i) enhance application fingerprinting performance, and ii) mitigate application fingerprinting on Tor.

## How to Test the Baseline and Improved Performance
#### 1. Requirements
  * Python 3.6.7
  * Python libraries specified in the `requirements.txt` file

#### 2. Obtain Baseline and Improved Performance
Run the scripts `run_ml.py` (baseline and the XGBoost-based approach), `run_grid.py` (the hyperparameter tuning-based approach), and `run_dnn.py` (the DNN-based approach).
The results will be saved in multiple CSV and PDF files.

## How to Test the Mitigation Performance
#### 1. Requirements
  * Same as above

#### 2. Obtain Mitigation Performance
Run the script `run_ml_def.py`.
The results will be saved in multiple CSV and PDF files.
You can specify different `csv_path_def` values to observe how mitigation effectiveness changes based on the ratio of random packets.

## Acknowledgment
The original dataset and code are part of the work "Peel the Onion: Recognition of Android Apps Behind the Tor Network," presented at the 15th International Conference on Information Security Practice and Experience (ISPEC2019).

We have added and modified a few scripts.