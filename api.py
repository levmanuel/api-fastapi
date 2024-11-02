from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
import pickle

# Charger le modèle avec pickle depuis le dossier models
with open("models/logistic_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Créer l'application FastAPI
app = FastAPI()

from pydantic import BaseModel

# Modèle de données complet pour le dataset Breast Cancer
class BreastCancerInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

# Endpoint de prédiction
@app.post("/predict")
def predict(data: BreastCancerInput):
    # Convertir les données reçues en DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Obtenir la prédiction
    prediction = model.predict(input_data)
    
    # Retourner le résultat de la prédiction
    return {"prediction": int(prediction[0])}  # 0 ou 1 selon le modèle