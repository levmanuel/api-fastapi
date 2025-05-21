from fastapi import Security, Depends, FastAPI, HTTPException, status, Body
from fastapi.security.api_key import APIKeyHeader, APIKey
from typing import List
from pydantic import BaseModel
import joblib

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import os

API_KEY = os.getenv("API_KEY", "0000")  # 🔐 En prod, défini via Railway
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

class IrisInput(BaseModel):
    features: List[List[float]]

    class Config:
        schema_extra = {
            "example": {
                "features": [
                    [5.1, 3.5, 1.4, 0.2],
                    [6.2, 2.8, 4.8, 1.8]
                ]
            }
        }

# Fonction de vérification de la clé
async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

app = FastAPI()

base_dir = Path(__file__).resolve().parent

@app.get("/")
def read_root():
    return {"message": "Hello from Railway"}

# Endpoint protégé
@app.get("/auth", dependencies=[Depends(get_api_key)])
async def read_root():
    return {"message": "Hello from Railway — secured with API Key"}

@app.get("/load_data", dependencies=[Depends(get_api_key)])
async def load_data():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "models" / "data.csv"

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Fichier CSV introuvable")

    df = pd.read_csv(csv_path)
    return df.head().fillna("").to_dict(orient="records")
    
@app.post(
    "/predict",
    summary="Prédire l'espèce d'iris",
    description="Fournit une ou plusieurs observations d'iris pour obtenir la prédiction de leur espèce."
)
async def predict(iris: IrisInput = Body(..., example={"features": [[5.1, 3.5, 1.4, 0.2],[6.2, 2.8, 4.8, 1.8]]})):
    try:
        class_names = {0: "setosa", 1: "versicolor", 2: "virginica"}
        base_dir = Path(__file__).resolve().parent
        model_path = base_dir / "models" / "iris_model.pkl"
        model = joblib.load(model_path)

        if not model_path.exists():
            raise HTTPException(status_code=500, detail="Modèle introuvable")
        
        X_input = np.array(iris.features)
        if X_input.ndim == 1:
            X_input = X_input.reshape(1, -1)
        if X_input.shape[1] != 4:
            return {"error": "Chaque observation doit contenir exactement 4 caractéristiques."}
        predictions = model.predict(X_input)
        results = [class_names[pred] for pred in predictions]
        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")