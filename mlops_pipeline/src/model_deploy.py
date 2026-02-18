import os
from datetime import date
from typing import List, Optional

import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel



#---------------------------------------------
# 1) Configuración
#---------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(
    BASE_DIR, "..", "artifacts", "models", "logistic_regression_end_to_end_pipeline.joblib"
)

APP_HOST = "0.0.0.0"
APP_PORT = 5040
THRESHOLD = 0.53

#---------------------------------------------
# 2) Cargar modelo end-to-end (raw -> pred) generado por model_training_evaluation.py
#---------------------------------------------
modelo = None
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No existe el modelo en: {MODEL_PATH}")
    modelo = joblib.load(MODEL_PATH)
    print(f"Modelo cargado con éxito: {MODEL_PATH}")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    modelo = None

#---------------------------------------------
# 3) Inicializar de la aplicación: FastAPI
#---------------------------------------------
app = FastAPI(
    title ="API de predicción de pago a tiempo",
    description = "Despliega un modelo de ML para predecir si un cliente pagará a tiempo.",
    version = "1.0.0"
)

#---------------------------------------------
# 4) Schemas
#---------------------------------------------
class PredictionInput(BaseModel):
    tipo_credito: str
    fecha_prestamo: date
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje: Optional[float] = None
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: float
    creditos_sector_financiero: int
    creditos_sector_cooperativo: int
    creditos_sector_real: int
    promedio_ingresos_datacredito: float
    tendencia_ingresos: str
    
class PredictionBatch(BaseModel):
    records: List[PredictionInput]

#---------------------------------------------
# 5) Endpoints API
#---------------------------------------------

# Endpoint de saludo
@app.get("/saludo")
def saludo():
    return{"Mensaje:": "Bienvenido, la API esta corriendo correctamente y esta usando un modelo de ML para hacer predicciones."}

# Endpoint para hacer predicciones con el modelo entrenado
@app.post("/predict")
def predict_batch(payload: PredictionBatch):
    if modelo is None:
        raise HTTPException(status_code=500, detail="El modelo no pudo ser cargado.")

    try:
        input_df = pd.DataFrame([r.model_dump() for r in payload.records])

        # ✅ importante: fecha a datetime (muchos pipelines esperan datetime64)
        if "fecha_prestamo" in input_df.columns:
            input_df["fecha_prestamo"] = pd.to_datetime(input_df["fecha_prestamo"], errors="coerce")

        probas_1 = modelo.predict_proba(input_df)[:, 1]
        preds = (probas_1 >= THRESHOLD).astype(int)

        probas_1 = None
        if hasattr(modelo, "predict_proba"):
            probas_1 = modelo.predict_proba(input_df)[:, 1]

        out = []
        for i in range(len(input_df)):
            out.append({
                "pred": int(preds[i]),
                "proba_pago_atiempo": None if probas_1 is None else float(probas_1[i]),
            })

        return {"n_records": len(input_df), "predictions": out}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al hacer la predicción: {e}")

# Cargar el script
if __name__ =="__main__":
    uvicorn.run("model_deploy:app", host = "0.0.0.0", port=5040, reload= True) #comprobar que puerto no esta ocupado