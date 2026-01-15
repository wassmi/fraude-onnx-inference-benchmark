import pandas as pd
import numpy as np
import onnxruntime as ort
import joblib
import time
from sklearn.metrics import roc_auc_score
import sys

def validate_production_model():
    print("Starting Production Gatekeeper validation (ONNX)...")
    
    # 1. Load Data and Production Artifacts
    df = pd.read_csv('data/creditcard_mock.csv')
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    # Load the scaler used in the pipeline
    scaler = joblib.load('models/production_scaler.joblib')
    X_scaled = scaler.transform(X).astype(np.float32)
    
    # 2. Initialize ONNX Runtime Session
    try:
        session = ort.InferenceSession("models/model.onnx")
    except Exception as e:
        print(f"❌ FAILED: Could not load ONNX model. Error: {e}")
        sys.exit(1)

    # 3. Accuracy Check (ONNX)
    input_name = session.get_inputs()[0].name
    # ONNX output is usually [probabilities_0, probabilities_1]
    raw_preds = session.run(None, {input_name: X_scaled})[1] 
    # Extract probability for class 1
    preds = [p[1] for p in raw_preds]
    
    auc = roc_auc_score(y, preds)
    print(f"ONNX Model AUC: {auc:.4f}")
    
    # 4. Latency Check (Benchmarking 5,000 samples)
    X_stress = np.tile(X_scaled, (50, 1))[:5000]
    
    start_time = time.time()
    session.run(None, {input_name: X_stress})
    latency = time.time() - start_time
    print(f"ONNX Latency (5k samples): {latency:.4f}s")
    
    # 5. The "Gatekeeper" Decisions (SOTA Thresholds)
    AUC_THRESHOLD = 0.70 
    LATENCY_THRESHOLD = 0.08 # Strict 2026 SOTA target
    
    if auc < AUC_THRESHOLD:
        print("❌ FAILED: Accuracy below threshold.")
        sys.exit(1)
        
    if latency > LATENCY_THRESHOLD:
        print(f"❌ FAILED: Latency {latency:.4f}s exceeds threshold {LATENCY_THRESHOLD}s.")
        sys.exit(1)
        
    print("✅ SUCCESS: ONNX model validated for production.")
    sys.exit(0)

if __name__ == "__main__":
    validate_production_model()