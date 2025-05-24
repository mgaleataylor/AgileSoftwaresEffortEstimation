# FineSE-replication-package

This repository contains the source code and data that we used to perform the experiment in the paper titled "Fine-SE: Integrating Semantic Features and Expert Features for Software Effort Estimations".

## Dataset

- We use business data and open source data to train and test the performance of the baselines and our method named FineSE, the sample data is attached in the `data` directory.

Please follow the steps below to reproduce the result.

## Environment Setup

### Python Environment Setup

Run the following command in terminal (or command line) to prepare the virtual environment.

```shell
conda create -n finese python=3.8
conda activate finese
pip install -r requirements.txt
```

## Experiment Result Replication Guide


### **FineSE Implementation**

To  reproduce the results of FineSE on open source data, run the following command:

- FineSE

```bash
python EXPERIMENT 1 - Replication of Original Fines-SE Results/FineSE.py
```

