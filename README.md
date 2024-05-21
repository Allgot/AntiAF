# Mitigating Enhanced Application Fingerprinting on Tor: A Preliminary Study
Proof of Concept to mitigate application fingerprinting (AF) on Tor.

## How to test the baseline performance and the mitigation performance
#### 1. Requirements
  * Python 3.6.7.
  * Python Libraries: see `requirements.txt` file.

#### 2. Get the baseline performance
Run the script `run_ml.py` and `run_dnn.py`. 
The results would be printed in multiple .csv files and one .pdf file.

#### 3. Get the mitigation performance
Run the script `run_ml_def.py`. 
The results would be printed in multiple .csv files and one .pdf file. 
You can specify different `csv_path_def` to see how mitigation effectiveness changes.

## Acknowledgment
Original dataset and code are part of the work "Peel the onion: Recognition of Android apps behind the Tor Network", 15th International Conference on Information Security Practice and Experience (ISPEC2019). 

We have added and modified few scripts.
