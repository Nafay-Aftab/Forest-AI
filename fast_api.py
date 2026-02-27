from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import io

# ──────────────────────────────────────────────
# App Setup
# ──────────────────────────────────────────────
app = FastAPI(
    title="🌲 Forest Cover Type Predictor API",
    description="Predicts dominant tree species from cartographic / terrain features using a tuned XGBoost model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Model & Preprocessor Loading
# ──────────────────────────────────────────────
try:
    model = joblib.load("champion_xgboost.joblib")
    preprocessor = joblib.load("spatial_preprocessor.joblib")
    print(" Model and preprocessor loaded successfully.")
except Exception as e:
    model = None
    preprocessor = None
    print(f" Could not load model/preprocessor: {e}")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
COVER_TYPES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz",
}


# ──────────────────────────────────────────────
# Feature Engineering  (mirrors notebook exactly)
# ──────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_eng = df.copy()
    df_eng["Euclidean_Distance_To_Hydrology"] = np.sqrt(
        df_eng["Horizontal_Distance_To_Hydrology"] ** 2
        + df_eng["Vertical_Distance_To_Hydrology"] ** 2
    )
    df_eng["Water_Elevation"] = (
        df_eng["Elevation"] - df_eng["Vertical_Distance_To_Hydrology"]
    )
    df_eng["Mean_Hillshade"] = df_eng[
        ["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]
    ].mean(axis=1)
    df_eng["Morning_vs_Afternoon_Sun"] = (
        df_eng["Hillshade_9am"] - df_eng["Hillshade_3pm"]
    )
    df_eng["Distance_To_Amenities"] = (
        df_eng["Horizontal_Distance_To_Roadways"]
        + df_eng["Horizontal_Distance_To_Fire_Points"]
    )
    return df_eng


# ──────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────
class PredictionInput(BaseModel):
    # Core terrain features
    Elevation: float = Field(..., example=2596, description="Elevation in meters")
    Aspect: float = Field(..., example=51, description="Aspect in azimuth degrees (0–360)")
    Slope: float = Field(..., example=3, description="Slope in degrees (0–66)")
    Horizontal_Distance_To_Hydrology: float = Field(..., example=258)
    Vertical_Distance_To_Hydrology: float = Field(..., example=0)
    Horizontal_Distance_To_Roadways: float = Field(..., example=510)
    Horizontal_Distance_To_Fire_Points: float = Field(..., example=6279)
    Hillshade_9am: float = Field(..., example=221, ge=0, le=255)
    Hillshade_Noon: float = Field(..., example=232, ge=0, le=255)
    Hillshade_3pm: float = Field(..., example=148, ge=0, le=255)

    # Wilderness Areas (one-hot, exactly one should be 1)
    Wilderness_Area1: int = Field(0, ge=0, le=1)
    Wilderness_Area2: int = Field(0, ge=0, le=1)
    Wilderness_Area3: int = Field(0, ge=0, le=1)
    Wilderness_Area4: int = Field(0, ge=0, le=1)

    # Soil Types (one-hot, exactly one should be 1)
    Soil_Type1: int = Field(0, ge=0, le=1)
    Soil_Type2: int = Field(0, ge=0, le=1)
    Soil_Type3: int = Field(0, ge=0, le=1)
    Soil_Type4: int = Field(0, ge=0, le=1)
    Soil_Type5: int = Field(0, ge=0, le=1)
    Soil_Type6: int = Field(0, ge=0, le=1)
    Soil_Type7: int = Field(0, ge=0, le=1)
    Soil_Type8: int = Field(0, ge=0, le=1)
    Soil_Type9: int = Field(0, ge=0, le=1)
    Soil_Type10: int = Field(0, ge=0, le=1)
    Soil_Type11: int = Field(0, ge=0, le=1)
    Soil_Type12: int = Field(0, ge=0, le=1)
    Soil_Type13: int = Field(0, ge=0, le=1)
    Soil_Type14: int = Field(0, ge=0, le=1)
    Soil_Type15: int = Field(0, ge=0, le=1)
    Soil_Type16: int = Field(0, ge=0, le=1)
    Soil_Type17: int = Field(0, ge=0, le=1)
    Soil_Type18: int = Field(0, ge=0, le=1)
    Soil_Type19: int = Field(0, ge=0, le=1)
    Soil_Type20: int = Field(0, ge=0, le=1)
    Soil_Type21: int = Field(0, ge=0, le=1)
    Soil_Type22: int = Field(0, ge=0, le=1)
    Soil_Type23: int = Field(0, ge=0, le=1)
    Soil_Type24: int = Field(0, ge=0, le=1)
    Soil_Type25: int = Field(0, ge=0, le=1)
    Soil_Type26: int = Field(0, ge=0, le=1)
    Soil_Type27: int = Field(0, ge=0, le=1)
    Soil_Type28: int = Field(0, ge=0, le=1)
    Soil_Type29: int = Field(0, ge=0, le=1)
    Soil_Type30: int = Field(0, ge=0, le=1)
    Soil_Type31: int = Field(0, ge=0, le=1)
    Soil_Type32: int = Field(0, ge=0, le=1)
    Soil_Type33: int = Field(0, ge=0, le=1)
    Soil_Type34: int = Field(0, ge=0, le=1)
    Soil_Type35: int = Field(0, ge=0, le=1)
    Soil_Type36: int = Field(0, ge=0, le=1)
    Soil_Type37: int = Field(0, ge=0, le=1)
    Soil_Type38: int = Field(0, ge=0, le=1)
    Soil_Type39: int = Field(0, ge=0, le=1)
    Soil_Type40: int = Field(0, ge=0, le=1)


class PredictionResponse(BaseModel):
    cover_type_id: int
    cover_type_name: str
    probabilities: dict[str, float]


# ──────────────────────────────────────────────
# Helper: run prediction on a DataFrame
# ──────────────────────────────────────────────
def _predict_dataframe(df_raw: pd.DataFrame):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model/preprocessor not loaded. Ensure champion_xgboost.joblib and spatial_preprocessor.joblib are present.",
        )
    df_eng = engineer_features(df_raw)
    X_processed = preprocessor.transform(df_eng)
    raw_preds = model.predict(X_processed)            # 0-indexed classes
    probas = model.predict_proba(X_processed)
    return raw_preds, probas


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single(payload: PredictionInput):
    """
    Accepts a single terrain observation and returns the predicted forest cover type
    along with class probabilities.
    """
    df_raw = pd.DataFrame([payload.model_dump()])
    raw_preds, probas = _predict_dataframe(df_raw)

    # Notebook shifts labels: model outputs 0–6, original classes are 1–7
    pred_class = int(raw_preds[0]) + 1
    prob_dict = {
        COVER_TYPES[i + 1]: round(float(probas[0][i]), 4) for i in range(7)
    }

    return PredictionResponse(
        cover_type_id=pred_class,
        cover_type_name=COVER_TYPES.get(pred_class, f"Class {pred_class}"),
        probabilities=prob_dict,
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(file: UploadFile = File(...)):
    """
    Accepts a CSV file (no target column required) and returns predictions for every row.

    The response is a JSON array where each element contains:
    - `row_index`
    - `cover_type_id`
    - `cover_type_name`
    - `probabilities`
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    contents = await file.read()
    try:
        df_raw = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    # Drop target column if accidentally included
    df_raw = df_raw.drop(columns=["Cover_Type"], errors="ignore")

    raw_preds, probas = _predict_dataframe(df_raw)

    results = []
    for i, (pred, prob_row) in enumerate(zip(raw_preds, probas)):
        pred_class = int(pred) + 1
        results.append(
            {
                "row_index": i,
                "cover_type_id": pred_class,
                "cover_type_name": COVER_TYPES.get(pred_class, f"Class {pred_class}"),
                "probabilities": {
                    COVER_TYPES[j + 1]: round(float(prob_row[j]), 4) for j in range(7)
                },
            }
        )

    return {"total_rows": len(results), "predictions": results}