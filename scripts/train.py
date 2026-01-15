import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import joblib
import os
import json

def create_mock_data():
    columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount', 'Class']
    data = np.random.randn(100, 31)
    df = pd.DataFrame(data, columns=columns)
    df['Class'] = np.random.randint(0, 2, 100)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/creditcard_mock.csv', index=False)

def train_and_package():
    df = pd.read_csv('data/creditcard_mock.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_scaled, y)

    os.makedirs('models', exist_ok=True)
    
    # 1. Save standard artifacts
    joblib.dump(scaler, 'models/scaler.joblib')

    # 2. Convert to ONNX (SOTA Optimization)
    initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open("models/model.onnx", "wb") as f:
        f.write(onx.SerializeToString())

    # 3. Generate Governance Manifest (The "Safety Buffer" Logic)
    manifest = {
        "model_version": os.getenv("GITHUB_SHA", "local")[:7],
        "thresholds": {
            "low_risk": 0.3,
            "high_risk": 0.7,
            "action": "trigger_2fa_if_between"
        },
        "input_schema": list(X.columns),
        "performance_target": "latency < 0.1s"
    }
    with open("models/manifest.json", "w") as f:
        json.dump(manifest, f, indent=4)

    print("✅ Model, Scaler, ONNX, and Manifest successfully packaged in models/")

if __name__ == "__main__":
    create_mock_data()
    train_and_package()