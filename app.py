from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib, pandas as pd

model = joblib.load("churn_model.pkl")
FEATURES = list(model.feature_names_in_)   # columns the model was trained on

app = FastAPI(title="Churn Prediction API")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

def build_row(customer: dict) -> dict:
    """Turn the 17 original fields into the columns the model expects.
    Columns like 'International plan_1' are one-hot versions of the original field."""
    row, missing = {}, []
    for f in FEATURES:
        if f in customer:                                  # plain numeric column
            row[f] = float(customer[f])
            continue
        base, _, val = f.rpartition("_")                   # one-hot column
        if base in customer and val.isdigit():
            x = float(customer[base])
            if base == "Customer service calls":
                x = min(x, 9)                              # model saw 0-9 calls
            row[f] = 1.0 if x == float(val) else 0.0
        else:
            missing.append(base or f)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields: {sorted(set(missing))}")
    return row

@app.get("/")
def home():
    inputs = sorted({f.rpartition("_")[0] if "_" in f else f for f in FEATURES}, key=str)
    return {"status": "ok", "send_these_fields": inputs}

@app.post("/predict")
def predict(customer: dict):
    try:
        row = pd.DataFrame([build_row(customer)])[FEATURES]
    except (TypeError, ValueError):
        raise HTTPException(status_code=422, detail="All values must be numbers")
    prob = float(model.predict_proba(row)[0][1])
    return {
        "churn": int(prob >= 0.5),
        "churn_probability": round(prob, 4),
        "risk": "high" if prob >= 0.6 else "medium" if prob >= 0.3 else "low",
    }
