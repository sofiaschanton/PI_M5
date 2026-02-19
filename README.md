# Proyecto Integrador (PI) — MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-API-green)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Monitoring-red)
![Scikit--Learn](https://img.shields.io/badge/scikit--learn-ML-orange)

Pipeline end-to-end de **Machine Learning + MLOps** para predecir **pago a tiempo (`pago_a_tiempo`)**.
Incluye **feature engineering**, **entrenamiento y selección de modelo**, **deploy con FastAPI + Docker** y **monitoreo de data drift**.

---

## 🎯 Objetivo del proyecto
Construir un sistema reproducible que permita:
- Preparar y transformar datos (FE + preprocesamiento)
- Entrenar y evaluar múltiples modelos
- Seleccionar el mejor modelo y umbral (threshold)
- Publicar una API para inferencia
- Monitorear **data drift** (RAW vs preprocesado)

## 🧠 Stack
- Python (pandas, numpy, scikit-learn)
- FastAPI + Uvicorn (serving)
- Docker (containerización)
- Streamlit (monitoring)
- Git/GitHub (control de versiones)
