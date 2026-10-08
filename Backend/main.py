import os
import hashlib
import joblib
from typing import Dict, Any, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="NFIP Claims Prediction & Data Integrity API",
    version="1.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS for local and deployed frontend interactions
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Path Resolution & Model Loading ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "Models")

PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "nfip_preprocessor.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "nfip_rf_model.pkl")

preprocessor = None
model = None

# In-memory LRU-style / Hash Cache for identical inputs
PREDICTION_CACHE: Dict[str, Dict[str, Any]] = {}
CACHE_MAX_SIZE = 5000

@app.on_event("startup")
def load_artifacts():
    global preprocessor, model
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to load model artifacts from {MODELS_DIR}: {e}")


# (Put this near the bottom of main.py, below your /predict route)
@app.get("/")
def serve_frontend():
    # Resolve the path to index.html relative to main.py
    base_dir = os.path.dirname(os.path.abspath(__file__))
    frontend_path = os.path.join(base_dir, "..", "Frontend", "index.html")
    return FileResponse(frontend_path)


# --- Schema Definitions ---
class ClaimInput(BaseModel):
    yearOfLoss: int = Field(..., ge=1900, le=2030)
    state: str
    countyCode: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    floodEvent: Optional[str] = "Missing"
    causeOfDamage: str
    ratedFloodZone: str
    occupancyType: int
    numberOfFloorsInTheInsuredBuilding: int
    elevatedBuildingIndicator: int
    primaryResidenceIndicator: int
    totalBuildingInsuranceCoverage: float = Field(..., ge=0)
    totalContentsInsuranceCoverage: float = Field(..., ge=0)
    buildingPropertyValue: Optional[float] = Field(default=None, ge=0)
    buildingReplacementCost: Optional[float] = Field(default=None, ge=0)
    waterDepth: Optional[float] = None
    floodWaterDuration: Optional[float] = None
    loss_month: int = Field(..., ge=1, le=12)
    building_age: Optional[int] = None


class ClaimResponse(BaseModel):
    cleaned_input: Dict[str, Any]
    predicted_building_payment: float
    data_quality_flags: List[str]
    cached: bool


# --- Data Quality Integrity Checks ---
def evaluate_and_clean_claim(raw: Dict[str, Any]):
    cleaned = dict(raw)
    flags: List[str] = []

    # 1. 0s treated as missing in valuation fields
    for field_name in ["buildingPropertyValue", "buildingReplacementCost"]:
        if cleaned.get(field_name) == 0:
            cleaned[field_name] = None
            flags.append(f"FLAG_ZERO_VALUATION: '{field_name}' was 0; converted to null for imputation.")

    # 2. Impossible / extreme geo coordinates
    lat = cleaned.get("latitude")
    lon = cleaned.get("longitude")
    if lat is not None and not (15.0 <= lat <= 72.0):
        flags.append(f"FLAG_OUT_OF_BOUNDS_LAT: Latitude {lat} outside standard US coverage.")
    if lon is not None and not (-180.0 <= lon <= -40.0):
        flags.append(f"FLAG_OUT_OF_BOUNDS_LON: Longitude {lon} outside standard US coverage.")

    # 3. Negative water depth anomaly
    water_depth = cleaned.get("waterDepth")
    if water_depth is not None and water_depth < 0:
        flags.append(f"FLAG_NEGATIVE_DEPTH: waterDepth ({water_depth}) is negative; possible basement or sensor artifact.")

    # 4. Coverage vs Property sanity check
    coverage = cleaned.get("totalBuildingInsuranceCoverage", 0.0)
    prop_val = cleaned.get("buildingPropertyValue")
    if prop_val and coverage > (prop_val * 3.0):
        flags.append("FLAG_OVERINSURANCE: Insurance coverage is over 3x estimated building property value.")

    # 5. Building age validation
    age = cleaned.get("building_age")
    if age is not None and (age < 0 or age > 250):
        flags.append(f"FLAG_ANOMALOUS_AGE: building_age ({age}) is out of expected physical bounds.")

    return cleaned, flags


# --- Inference Endpoint ---
@app.post("/predict", response_model=ClaimResponse)
@limiter.limit("30/minute")
def predict_claim(request: Request, claim: ClaimInput):
    if preprocessor is None or model is None:
        raise HTTPException(status_code=503, detail="Model is still loading or unavailable.")

    raw_dict = claim.dict()
    cleaned_dict, flags = evaluate_and_clean_claim(raw_dict)

    # In-memory Hash Cache
    cache_key = hashlib.sha256(str(sorted(cleaned_dict.items())).encode("utf-8")).hexdigest()
    if cache_key in PREDICTION_CACHE:
        cached_result = PREDICTION_CACHE[cache_key]
        return ClaimResponse(
            cleaned_input=cleaned_dict,
            predicted_building_payment=cached_result["prediction"],
            data_quality_flags=cached_result["flags"],
            cached=True
        )

    # Vectorized / DataFrame preparation for Scikit-Learn
    df_row = pd.DataFrame([cleaned_dict])

    try:
        # Preprocess features
        features_transformed = preprocessor.transform(df_row)

        # Random Forest Inference
        prediction = float(model.predict(features_transformed)[0])
        prediction = max(0.0, round(prediction, 2))
    except Exception as err:
        raise HTTPException(status_code=422, detail=f"Inference failure: {str(err)}")

    # Update Cache with eviction
    if len(PREDICTION_CACHE) >= CACHE_MAX_SIZE:
        PREDICTION_CACHE.pop(next(iter(PREDICTION_CACHE)))

    PREDICTION_CACHE[cache_key] = {"prediction": prediction, "flags": flags}

    return ClaimResponse(
        cleaned_input=cleaned_dict,
        predicted_building_payment=prediction,
        data_quality_flags=flags,
        cached=False
    )


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "preprocessor_loaded": preprocessor is not None,
        "model_loaded": model is not None
    }

@app.get("/cache")
def view_cache():
    """Returns the total number of cached entries and all stored cache records."""
    return {
        "cache_size": len(PREDICTION_CACHE),
        "cached_entries": PREDICTION_CACHE
    }

@app.delete("/cache")
def clear_cache():
    """Clears all cached predictions."""
    PREDICTION_CACHE.clear()
    return {"message": "Cache successfully cleared", "cache_size": len(PREDICTION_CACHE)}