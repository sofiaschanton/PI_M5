# model_training_evaluation.py: pipeline de entrenamiento y evaluación de modelos
# usando tu run_feature_pipeline() de ft_engineering.py
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from datetime import datetime, timezone
import json

import optuna
from sklearn.model_selection import cross_validate, TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    balanced_accuracy_score, accuracy_score, confusion_matrix, 
    ConfusionMatrixDisplay, make_scorer
)

# Boosting
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Pipeline de FE + preprocess
try:
    from .ft_engineering import ft_pipeline
except ImportError:
    from ft_engineering import ft_pipeline

#-----------------------------------------------------
# Utils
#-----------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_artifacts(
    ft_pipeline,          # artifacts["pipeline"] (tu build_data_pipeline ya fiteado)
    model,                # el estimador final entrenado
    model_name: str,      
    base_dir: str = None  # por defecto: ../artifacts desde src/
):
    # Guardar relativo a la raíz del repo (mlops_pipeline/)
    # Si se ejecutá desde src/, esto sube un nivel.
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")

    pipelines_dir = os.path.join(base_dir, "pipelines")
    models_dir    = os.path.join(base_dir, "models")

    ensure_dir(pipelines_dir)
    ensure_dir(models_dir)

    # 1) FT pipeline (FE + cap + pre)
    ft_path = os.path.join(pipelines_dir, "ft_pipeline.joblib")
    joblib.dump(ft_pipeline, ft_path)

    # 2) Modelo solo
    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)

    # 3) Pipeline completo (RAW -> pred)
    end_to_end_pipeline = Pipeline(steps=[
        ("ft", ft_pipeline),
        ("model", model)
    ])
    full_path = os.path.join(models_dir, f"{model_name}_end_to_end_pipeline.joblib")
    joblib.dump(end_to_end_pipeline, full_path)

    print(f"✅ Guardado ft_pipeline: {ft_path}")
    print(f"✅ Guardado modelo:      {model_path}")
    print(f"✅ Guardado full pipe:   {full_path}")

    return {"ft_pipeline": ft_path, "model": model_path, "end_to_end_pipeline": full_path}

def save_run_metadata(model_name: str, info: dict, base_dir: str = None):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts")

    meta_dir = os.path.join(base_dir, "metadata")
    ensure_dir(meta_dir)

    payload = {
        "model_name": model_name,
        "saved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **info
    }

    path = os.path.join(meta_dir, f"{model_name}_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"✅ Guardada metadata:    {path}")
    return path

def _safe_predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    raise ValueError("Este modelo no soporta predict_proba: no puedo calcular threshold para clase 0.")

def pr_auc_class_0(y_true, y_proba):
    y_true = np.asarray(y_true)
    y0 = (y_true == 0).astype(int)

    y_proba = np.asarray(y_proba)

    # Caso 2D: viene predict_proba completo
    if y_proba.ndim == 2:
        proba_0 = y_proba[:, 0]
    else:
        # Caso 1D: viene proba de clase 1
        proba_0 = 1 - y_proba

    return average_precision_score(y0, proba_0)

pr_auc_0_scorer = make_scorer(pr_auc_class_0, needs_proba=True)

def best_threshold_f1_class0(y_true, y_score_class1, n=300):
    y_true_0 = (np.asarray(y_true) == 0).astype(int)
    p0 = 1 - np.asarray(y_score_class1)

    thresholds = np.linspace(0.0, 1.0, n)
    best_t, best_f1 = 0.5, -1

    for t in thresholds:
        pred_0 = (p0 >= t).astype(int)
        f1_0 = f1_score(y_true_0, pred_0, zero_division=0)
        if f1_0 > best_f1:
            best_f1, best_t = f1_0, t

    return float(best_t), float(best_f1)
#-----------------------------------------------------
# 1) Definir candidatos a modelos
#-----------------------------------------------------
def build_models(random_state: int = 42):
    """Construye la lista de modelos candidatos para comparación."""

    # Logistic Regression
    logreg = LogisticRegression(
        C=1.0,
        penalty="l2",
        max_iter=5000,
        solver="lbfgs",
        random_state=random_state,
        class_weight="balanced")

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced"
    )
     
    # XGB Classifier
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1
    )
    
    # LGM Classifier
    lgb = LGBMClassifier(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1
    )

    # Catboost Classifier
    cat = CatBoostClassifier(
        iterations=800,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False
    )

    models = [
        ("logistic_regression", logreg),
        ("random_forest", rf),
        ("xgboost", xgb),
        ("lightgbm", lgb),
        ("catboost", cat)
    ]
    
    return models

#-----------------------------------------------------
# 2) Optuna para el mejor (solo RF / XGB / CatBoost)
#-----------------------------------------------------
def tune_best_model_with_optuna(best_name, X_train, y_train, cv_folds,
                                optimize_metric="pr_auc", n_trials=50, random_state=42):
    """Optimiza hiperparámetros con Optuna para RF, XGB o CatBoost."""

    scorers = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "pr_auc_0": pr_auc_0_scorer,
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "accuracy": "accuracy",
    }
    
    
    # Validar métrica antes de indexar para evitar KeyError
    if optimize_metric not in scorers:
        raise ValueError(f"optimize_metric='{optimize_metric}' no está en scorers. Opciones: {list(scorers.keys())}")
    scoring = scorers[optimize_metric]


    def objective(trial):
        if best_name == "random_forest":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 900),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "class_weight": "balanced",
                "random_state": random_state,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        elif best_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 900),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "gamma": trial.suggest_float("gamma", 0.0, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
                "eval_metric": "logloss",
                "random_state": random_state,
                "n_jobs": -1,
                "verbosity": 0,
            }
            model = XGBClassifier(**params)

        elif best_name == "catboost":
            params = {
                "iterations": trial.suggest_int("iterations", 200, 1200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 2, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "loss_function": "Logloss",
                "random_seed": random_state,
                "verbose": False,
                "allow_writing_files": False,
            }
            model = CatBoostClassifier(**params)

        else:
            raise ValueError(f"Optuna: sin espacio de búsqueda para '{best_name}' (solo RF/XGB/CatBoost).")

        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1
        )
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value
#-----------------------------------------------------
# 3) Calcular métricas de clasificación
#-----------------------------------------------------

def summarize_classification(y_true, y_pred, y_score=None):
    """
    Métricas de clasificación para binario.
    - y_pred: predicción 0/1
    - y_score: score continuo (predict_proba[:,1] o decision_function) para ROC/PR AUC
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),      # Macro (balanceado entre clases)
        "precision": precision_score(y_true, y_pred, pos_label=1),   # Clase 1: paga (mayoritaria)
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),
    }

    # Para clase 0
    metrics["precision_class_0"] = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics["recall_class_0"] = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics["f1_class_0"] = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

    # Incluir AUCs cuando hay score continuo disponible
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
        metrics["pr_auc_0"] = pr_auc_class_0(y_true, y_score)

    return metrics

#-----------------------------------------------------
# 4) Entrenar, evaluar y seleccionar modelo
#-----------------------------------------------------
def train_and_select_model( X_train, y_train, X_test, y_test,random_state):
    """Entrena candidatos, selecciona por CV y evalúa el mejor en test."""
   
    models = build_models(random_state=random_state)

    results = []

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "bal_acc": "balanced_accuracy",
        "f1": "f1",
        "recall": "recall",           
        "precision": "precision",
        "pr_auc_0":pr_auc_0_scorer
    }

    cv_folds =TimeSeriesSplit(n_splits=5)

    for name, model in models:
        print(f'Entrenando y evaluando {name}..')
        
        # -------------------------
        # 1) Validación cruzada en train: evalúa generalización
        # -------------------------
        cv_scores = cross_validate(
            model, 
            X_train, y_train,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=False,
        )

        # -------------------------
        # 2) Entrenar modelo final con todos los datos de train
        # -------------------------
        # Fit final
        start_fit = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_fit
        
        # -------------------------
        # 3) PREDICCIÓN TEST 
        # -------------------------
        start_pred = time.perf_counter()
        y_score_test = _safe_predict_proba(model, X_test)
        y_pred_test = (y_score_test >= 0.5).astype(int)
        pred_time = time.perf_counter() - start_pred
        pred_time_per_sample = pred_time / len(X_test)

        result = {
            "name": name,
            "model": model,

            # CV
            "cv_roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(cv_scores["test_roc_auc"].std()),
            "cv_pr_auc_mean": float(cv_scores["test_pr_auc"].mean()),
            "cv_pr_auc_std":  float(cv_scores["test_pr_auc"].std()),
            "cv_pr_auc_0_mean": float(cv_scores["test_pr_auc_0"].mean()),
            "cv_pr_auc_0_std": float(cv_scores["test_pr_auc_0"].std()),
            "cv_bal_acc_mean": float(cv_scores["test_bal_acc"].mean()),
            "cv_recall_mean": float(cv_scores["test_recall"].mean()),
            "cv_f1_mean": float(cv_scores["test_f1"].mean()),
            # tiempos 
            "train_time_s": float(train_time),
            "pred_time_s": float(pred_time),
            "pred_time_per_sample_ms": float(pred_time_per_sample * 1000),
        }

        results.append(result)

        print(
        f"{name:>18} | "
        f"CV PR-AUC Class 0: {result['cv_pr_auc_0_mean']:.4f} "
        f"(±{result['cv_pr_auc_0_std']:.4f}) | "
        f"CV ROC-AUC: {result['cv_roc_auc_mean']:.4f} "
        f"(±{result['cv_roc_auc_std']:.4f}) | "
        f"Train: {train_time:.2f}s  | "
        f"Predict: {pred_time:.2f}s"
)

    # -------------------------
    # 4) Selección robusta (mean - std)
    # -------------------------

    # Mejor por PR-AUC Class0
    best = max(results, key=lambda x: x["cv_pr_auc_0_mean"] - x["cv_pr_auc_0_std"])

    print(f"\n✅ Mejor modelo final (PR-AUC clase 0): {best['name']}")
    print(f"{best['cv_pr_auc_0_mean']:.4f} (±{best['cv_pr_auc_0_std']:.4f})")

    best_model = best["model"]

    #Reentrenar con todo el train
    best_model.fit(X_train, y_train)

    # -------------------------
    # 5) THRESHOLD ÓPTIMO (OOF)
    # -------------------------
    oof_scores = np.zeros(len(y_train))

    for train_idx, val_idx in cv_folds.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]

        best_model.fit(X_tr, y_tr)
        oof_scores[val_idx] = _safe_predict_proba(best_model, X_val) # <-- clase 1

    best_threshold, best_metric = best_threshold_f1_class0(y_train, oof_scores)
    best["best_threshold"] = float(best_threshold)
    print(f"🎯 Threshold óptimo (f1 class 0) en CV train: {best_threshold:.4f} | F1 class 0: {best_metric:.4f}")

    # -------------------------
    # 6) EVALUACIÓN FINAL EN TEST
    # -------------------------
    y_score_best = _safe_predict_proba(best_model, X_test)  # proba clase 1

    pred_0 = ((1 - y_score_best) >= best_threshold).astype(int)   # decide "es 0" 
    y_pred_best = np.where(pred_0 == 1, 0, 1)                     # convierte a 0/1 final

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred_best))

    best["test_metrics"] = summarize_classification(
        y_test,
        y_pred_best,
        y_score=y_score_best
    )


    # Optuna solo si aplica
    tuned_model = None
    tuned_info = None

    if best["name"] in {"random_forest", "xgboost", "catboost"}:
        best_params, best_cv_score = tune_best_model_with_optuna(
            best_name=best["name"],
            X_train=X_train,
            y_train=y_train,
            cv_folds=cv_folds,
            optimize_metric="pr_auc",
            n_trials=50,
            random_state=random_state
        )

        print(f"\n🔧 Optuna mejor CV (pr_auc): {best_cv_score:.4f}")
        print("Mejores hiperparámetros:", best_params)

        # construir tuned model
        if best["name"] == "random_forest":
            tuned_model = RandomForestClassifier(**best_params, class_weight="balanced",
                                                random_state=random_state, n_jobs=-1)
        elif best["name"] == "xgboost":
            tuned_model = XGBClassifier(**best_params, eval_metric="logloss",
                                        random_state=random_state, n_jobs=-1, verbosity=0)
        elif best["name"] == "catboost":
            tuned_model = CatBoostClassifier(**best_params, loss_function="Logloss",
                                             random_seed=random_state, verbose=False,
                                             allow_writing_files=False)

        tuned_model.fit(X_train, y_train)

        # evaluar tuned en test
        y_pred_t = tuned_model.predict(X_test)
        y_score_t = _safe_predict_proba(tuned_model, X_test)
        tuned_test = summarize_classification(y_test, y_pred_t, y_score=y_score_t)

        tuned_info = {
            "name": f"{best['name']}_tuned",
            "model": tuned_model,
            "best_optuna_cv": best_cv_score,
            "best_params": best_params,
            "test_metrics": tuned_test
        }

    # retornamos best base (dict), results y (opcional) tuned_info
    return best, results, tuned_info

#-----------------------------------------------------
# 4) Reporte: tabla resumen + gráficos comparativos
#-----------------------------------------------------
def build_results_table(results):
    """Convierte resultados de entrenamiento en una tabla comparativa (solo CV/train)."""
    rows = []
    for r in results:
        rows.append({
            "model": r.get("name", "unknown"),

            # CV
            "cv_pr_auc_mean": r.get("cv_pr_auc_mean", np.nan),
            "cv_roc_auc_mean":  r.get("cv_roc_auc_mean", np.nan),
            "cv_bal_acc_mean":  r.get("cv_bal_acc_mean", np.nan),
            "cv_recall_mean": r.get("cv_recall_mean", np.nan),
            "cv_f1_mean": r.get("cv_f1_mean", np.nan),
            

            # tiempos
            "train_time_s":     r.get("train_time_s", np.nan),
            "pred_time_s": r.get("pred_time_s", np.nan),
            "pred_time_per_sample_ms": r.get("pred_time_per_sample_ms", np.nan),
        })

    df = pd.DataFrame(rows)

    for c in df.columns:
        if c != "model":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.round(4).reset_index(drop=True)


def build_final_test_table(best_model, tuned_info=None):
    """Construye una tabla con métricas de test solo para modelos finales."""
    rows = []

    best_tm = best_model.get("test_metrics", {}) or {}
    rows.append({
        "model": best_model.get("name", "best_model"),
        "test_pr_auc":  best_tm.get("pr_auc", np.nan),
        "test_roc_auc": best_tm.get("roc_auc", np.nan),
        "test_bal_acc":   best_tm.get("balanced_acc", np.nan),
        "test_recall":  best_tm.get("recall", np.nan),
        "test_precision": best_tm.get("precision", np.nan),
        "test_f1":      best_tm.get("f1", np.nan),
    })

    if tuned_info is not None:
        tuned_tm = tuned_info.get("test_metrics", {}) or {}
        rows.append({
            "model": tuned_info.get("name", "tuned_model"),
            "test_pr_auc":  tuned_tm.get("pr_auc", np.nan),
            "test_roc_auc": tuned_tm.get("roc_auc", np.nan),
            "test_bal_acc":   tuned_tm.get("balanced_acc", np.nan),
            "test_recall":  tuned_tm.get("recall", np.nan),
            "test_precision": tuned_tm.get("precision", np.nan),
            "test_f1":      tuned_tm.get("f1", np.nan),
        })

    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != "model":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.round(4).reset_index(drop=True)

def plot_model_comparisons(df_results):
    """Gráfica métricas clave para comparar modelos de forma visual."""
    metrics_to_plot = [
        ("cv_recall_mean", "CV Recall  (mean)"),
        ("cv_pr_auc_mean", "CV PR-AUC (mean)"),
        ("train_time_s", "Tiempo entrenamiento (s)"),
    ]

    for col, title in metrics_to_plot:
        if col not in df_results.columns:
            continue
        if df_results[col].isna().all():
            continue

        plt.figure(figsize=(10, 4))
        plt.bar(df_results["model"], df_results[col])
        plt.xticks(rotation=45, ha="right")
        plt.title(title)
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

def display_confusion_matrix(
    model,
    X,
    y,
    threshold: float = 0.5,
    threshold_class: int = 1,
    title: str = ""
):
    # Predicción con threshold si hay probabilidades
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        if threshold_class == 0:
            pred_0 = ((1 - proba) >= threshold).astype(int)
            y_pred = np.where(pred_0 == 1, 0, 1)
        else:
            y_pred = (proba >= threshold).astype(int)
    else:
        # fallback si el modelo no da proba
        y_pred = model.predict(X)

    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, values_format="d")
    ax.set_title(title or f"Confusion Matrix (threshold={threshold:.4f}, class={threshold_class})")
    plt.tight_layout()
    plt.show()

    return cm
#-----------------------------------------------------
# 5) MAIN
#-----------------------------------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../mlops_pipeline/src
    PROJECT_DIR = os.path.dirname(BASE_DIR)                        # .../mlops_pipeline
    MODEL_DIR = os.path.join(PROJECT_DIR, "models") 
    DROP_COLS = ["puntaje"]     # Se excluye por tener una correlación de 0,92 con el target

    X_train_p, X_test_p, y_train, y_test, artifacts = ft_pipeline(
        target="pago_atiempo",
        test_size=0.2,
        split_type="temporal",
        random_state=42,
        stratify=True,
        drop_cols=DROP_COLS, 
        verbose=True
    )

    best_model, results, tuned_info = train_and_select_model(
        X_train_p, y_train, X_test_p, y_test, random_state=42
    )

    df_results = build_results_table(results)
    df_final_test = build_final_test_table(best_model, tuned_info)

    print("\n" + "=" * 60)
    print("TABLA COMPARATIVA (CV/TRAIN)")
    print(df_results.to_string(index=False))
    print("=" * 60)

    print("\n" + "=" * 60)
    print("TABLA FINAL (TEST)")
    print(df_final_test.to_string(index=False))
    print("=" * 60)

    plot_model_comparisons(df_results)

    thr = best_model.get("best_threshold", 0.5)

    cm_best = display_confusion_matrix(
        model=best_model["model"],
        X=X_test_p,
        y=y_test,
        threshold=thr,
        threshold_class=0,
        title=f"{best_model['name']} - TEST"
    )

    if tuned_info is not None:
        thr_tuned = tuned_info.get("best_threshold", thr)  # si no, usa el mismo
        cm_tuned = display_confusion_matrix(
            model=tuned_info["model"],
            X=X_test_p,
            y=y_test,
            threshold=thr_tuned,
            threshold_class=0,
            title=f"{tuned_info['name']} - TEST"
        )

    print("\n" + "=" * 60)
    print(f"MEJOR MODELO BASE (por robust pr_auc CV): {best_model['name']}")
    print("CV (mean):",
          {
              "pr_auc": round(best_model.get("cv_pr_auc_mean", np.nan), 4),
              "roc_auc": round(best_model.get("cv_roc_auc_mean", np.nan), 4),
              "balanced_acc": round(best_model.get("cv_bal_acc_mean", np.nan), 4),
              "recall": round(best_model.get("cv_recall_mean", np.nan), 4),
          })
    print("TEST:", {k: round(v, 4) for k, v in best_model["test_metrics"].items()})
    print("=" * 60)

    if tuned_info is not None:
        print("\n" + "=" * 60)
        print(f"MODELO TUNEADO: {tuned_info['name']}")
        print(f"Optuna best CV (PR_auc): {tuned_info['best_optuna_cv']:.4f}")
        print("TEST:", {k: round(v, 4) for k, v in tuned_info["test_metrics"].items()})
        print("=" * 60)

      # Guardar mejor modelo (base)
    paths_best = save_artifacts(
        ft_pipeline=artifacts["pipeline"],
        model=best_model["model"],
        model_name=best_model["name"]
    )

    # Preparar tuned_info para JSON (sacar el objeto model)
    tuned_json = {}
    if tuned_info is not None:
        tuned_json = tuned_info.copy()
        tuned_json.pop("model", None)

    best_threshold = float(best_model.get("best_threshold", 0.5))

    # Guardar metadata del mejor modelo
    metadata_path = save_run_metadata(
        model_name=best_model["name"],
        info={
            "target": "pago_atiempo",
            "drop_cols": DROP_COLS,
            "split": {
                "type": "temporal",
                "test_size": 0.2,
                "random_state": 42,
                "stratify": True
            },
            "threshold": {
                "default": best_threshold,
                "method": "cv_optimal",          # o "youden", "f1", etc.
                "metric_optimized": "f1_0"  # poné lo que usaste realmente
            },  
            "metrics": {
                "cv_mean": {
                    "pr_auc": best_model.get("cv_pr_auc_mean", None),
                    "roc_auc": best_model.get("cv_roc_auc_mean", None),
                    "balanced_acc": best_model.get("cv_bal_acc_mean", None),
                    "recall": best_model.get("cv_recall_mean", None),
                },
                "test": best_model.get("test_metrics", {})
            },
            "tuned_info": tuned_json
        }
    )

    # Guardar modelo tuneado si existe
    if tuned_info is not None:
        paths_tuned = save_artifacts(
            ft_pipeline=artifacts["pipeline"],
            model=tuned_info["model"],
            model_name=tuned_info["name"]
        )

        tuned_meta = tuned_info.copy()
        tuned_meta.pop("model", None)

        metadata_tuned_path = save_run_metadata(
            model_name=tuned_info["name"],
            info={
                "target": "pago_atiempo",
                "drop_cols": DROP_COLS,
                "split": {
                    "type": "temporal",
                    "test_size": 0.2,
                    "random_state": 42,
                    "stratify": True
                },
                "metrics": {
                    "best_optuna_cv_pr_auc": tuned_info.get("best_optuna_cv", None),
                    "test": tuned_info.get("test_metrics", {})
                },
                "tuned_info": tuned_meta
            }
        )