from fastapi import Security, Depends, FastAPI, HTTPException, status
from fastapi.security.api_key import APIKeyHeader, APIKey
from pydantic import BaseModel
import pandas as pd
import pickle
from pathlib import Path
import os

API_KEY = os.getenv("API_KEY", "0000")  # 🔐 En prod, défini via Railway
API_KEY_NAME = "x-api-key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Fonction de vérification de la clé
async def get_api_key(api_key_header: str = Security(api_key_header)) -> str:
    if api_key_header == API_KEY:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from Railway"}

# Endpoint protégé
@app.get("/auth", dependencies=[Depends(get_api_key)])
async def read_root():
    return {"message": "Hello from Railway — secured with API Key"}

# base_dir = Path(__file__).resolve().parent
# model_path = base_dir / "models" / "logistic_regression_model.pkl"
# with model_path.open("rb") as f:
#     model = pickle.load(f)

# # Modèle de données complet pour le dataset Breast Cancer
# class BreastCancerInput(BaseModel):
#     mean_radius: float
#     mean_texture: float
#     mean_perimeter: float
#     mean_area: float
#     mean_smoothness: float
#     mean_compactness: float
#     mean_concavity: float
#     mean_concave_points: float
#     mean_symmetry: float
#     mean_fractal_dimension: float
#     radius_error: float
#     texture_error: float
#     perimeter_error: float
#     area_error: float
#     smoothness_error: float
#     compactness_error: float
#     concavity_error: float
#     concave_points_error: float
#     symmetry_error: float
#     fractal_dimension_error: float
#     worst_radius: float
#     worst_texture: float
#     worst_perimeter: float
#     worst_area: float
#     worst_smoothness: float
#     worst_compactness: float
#     worst_concavity: float
#     worst_concave_points: float
#     worst_symmetry: float
#     worst_fractal_dimension: float

# # Endpoint de prédiction
# @app.post("/predict")
# def predict(data: BreastCancerInput):
#     # Convertir les données reçues en DataFrame
#     input_data = pd.DataFrame([data.dict()])
    
#     # Obtenir la prédiction
#     prediction = model.predict(input_data)
    
#     # Retourner le résultat de la prédiction
#     return {"prediction": int(prediction[0])}  # 0 ou 1 selon le modèle