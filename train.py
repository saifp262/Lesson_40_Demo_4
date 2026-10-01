# ==============================================================================
# 🤖 ROBO-HIKER CLOUD SENTINEL - MLOPS TRAINING PIPELINE
# Compatible with Amazon SageMaker Script Mode & Local Execution
# ==============================================================================

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def parse_args():
    parser = argparse.ArgumentParser(description="Train Robo-Hiker Ring Prediction Model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of boosting trees")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum tree depth")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--data_path", type=str, default="abalone.csv", help="Path to input CSV")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory to save model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("=" * 70)
    print("🤖 ROBO-HIKER CLOUD SENTINEL: SAGEMAKER MLOPS TRAINING PIPELINE")
    print("=" * 70)
    print(f"⚙️ Hyperparameters: n_estimators={args.n_estimators}, max_depth={args.max_depth}, lr={args.learning_rate}")
    
    # 1. Load Data
    script_dir = os.path.dirname(__file__)
    data_file = os.path.join(script_dir, args.data_path) if not os.path.isabs(args.data_path) else args.data_path
    
    if not os.path.exists(data_file):
        print(f"❌ Error: Dataset {data_file} not found!")
        sys.exit(1)
        
    print(f"📂 Loading data from: {data_file}")
    df = pd.read_csv(data_file)
    print(f"📊 Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # 2. Features and Target
    target = "Rings"
    X = df.drop(columns=[target])
    y = df[target]
    
    categorical_cols = ['Sex']
    numeric_cols = [c for c in X.columns if c not in categorical_cols]
    
    # 3. Preprocessor Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first'), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )
    
    # 4. Train / Test Split (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"🎯 Split: {X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    
    # 5. Build and Train Model Pipeline
    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        random_state=42
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    print("🚀 Training Gradient Boosting Regressor...")
    pipeline.fit(X_train, y_train)
    
    # 6. Evaluation
    y_pred = pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\n📈 [Model Performance Metrics]")
    print(f"   • RMSE     : {rmse:.4f} rings")
    print(f"   • MAE      : {mae:.4f} rings")
    print(f"   • R² Score : {r2:.4f}")
    
    # 7. Save Model Artifact
    save_dir = os.path.join(script_dir, args.model_dir) if not os.path.isabs(args.model_dir) else args.model_dir
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, "model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\n💾 Model artifact saved successfully: {model_path}")
    print("=" * 70)
    print("✅ MLOPS PIPELINE EXECUTION SUCCESSFUL")
    print("=" * 70)

if __name__ == "__main__":
    main()
