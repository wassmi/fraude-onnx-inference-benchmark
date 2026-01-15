import joblib
import pandas as pd
import time
from sklearn.metrics import roc_auc_score
import sys

def validate_model():
    print("Starting Gatekeeper validation...")
    
    # 1. Load Data and Artifacts
    df = pd.read_csv('data/creditcard_mock.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    model = joblib.load('models/rf_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    
    # 2. Preprocess
    X_scaled = scaler.transform(X)
    
    # 3. Accuracy Check (ROC-AUC)
    preds = model.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, preds)
    print(f"Model AUC: {auc:.4f}")
    
    # 4. Latency Check (Benchmarking 5,000 samples)
    # Since our mock is small, we will repeat the mock data to simulate load
    X_stress = pd.concat([pd.DataFrame(X_scaled)] * 50).values[:5000]
    
    start_time = time.time()
    model.predict_proba(X_stress)
    latency = time.time() - start_time
    print(f"Inference Latency (5k samples): {latency:.4f}s")
    
    # 5. The "Gatekeeper" Decisions
    # SOTA Practice: Hard-coded thresholds that match our notebook benchmarks
    AUC_THRESHOLD = 0.70  # Lowered for mock data variability
    LATENCY_THRESHOLD = 0.15 # seconds
    
    if auc < AUC_THRESHOLD:
        print("❌ FAILED: Accuracy below threshold.")
        sys.exit(1) # Signals failure to GitHub Actions
        
    if latency > LATENCY_THRESHOLD:
        print("❌ FAILED: Performance regression detected (too slow).")
        sys.exit(1)
        
    print("✅ SUCCESS: Model meets Reliability & Governance standards.")
    sys.exit(0) # Signals success

if __name__ == "__main__":
    validate_model()