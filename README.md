# Robo-Hiker MLOps: Cloud Model Training Pipeline 🤖🏔️

This repository contains the production training pipeline for the Robo-Hiker Mountain Age Prediction Sentinel. It is designed to be version-controlled in Git and executed either locally or inside **Amazon SageMaker Script Mode**.

## 📁 Repository Structure
```text
sagemaker-mlops-demo/
├── abalone.csv          # Training dataset (features & rings target)
├── train.py             # Modular ML training script (Script Mode compatible)
├── requirements.txt     # Python dependencies
├── models/              # Output directory for serialized model artifacts
└── README.md            # Project documentation & runbook
```

## 🚀 How to Run Training
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run training with default hyperparameters
python train.py

# 3. Run training with custom hyperparameters (Script Mode)
python train.py --n_estimators 150 --max_depth 6 --learning_rate 0.05
```

## 📊 Target Metrics
* **RMSE Target:** $\le 2.30$ rings
* **R² Target:** $\ge 0.50$
