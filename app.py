from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib, pandas as pd

model = joblib.load("churn_model.pkl")
FEATURES = list(model.feature_names_in_)

app = FastAPI(title="Churn Prediction API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/")
def home():
    return {"status": "ok", "expected_fields": FEATURES}

@app.post("/predict")
def predict(customer: dict):
    missing = [f for f in FEATURES if f not in customer]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields: {missing}")
    row = pd.DataFrame([customer])[FEATURES]
    prob = float(model.predict_proba(row)[0][1])
    return {
        "churn": int(prob >= 0.5),
        "churn_probability": round(prob, 4),
        "risk": "high" if prob >= 0.6 else "medium" if prob >= 0.3 else "low",
    }
