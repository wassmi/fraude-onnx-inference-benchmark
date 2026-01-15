import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1. Create Mock Data (Simulating ccfraude_inference_opt.ipynb structure)
def create_mock_data():
    print("Creating mock data...")
    # 30 features (V1-V28, Time, Amount) + 1 Class
    columns = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount', 'Class']
    data = np.random.randn(100, 31)
    df = pd.DataFrame(data, columns=columns)
    df['Class'] = np.random.randint(0, 2, 100)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/creditcard_mock.csv', index=False)

# 2. Train and Save
def train():
    df = pd.read_csv('data/creditcard_mock.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Using the same parameters from your notebook
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X_scaled, y)

    # Save artifacts for the registry
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/rf_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    print("Mock model and scaler saved to models/")

if __name__ == "__main__":
    create_mock_data()
    train()