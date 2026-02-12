# model_training_evaluation.py: pipeline de entrenamiento y evaluación de modelos
# usando tu run_feature_pipeline() de ft_engineering.py
import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, make_scorer,
    balanced_accuracy_score
)

# Boosting
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Pipeline de FE + preprocess
from ft_engineering import run_feature_pipeline

# ============================================================
# Utils
# ============================================================
def save_model(model, model_name, path=None):
    """Guarda un modelo serializado en el directorio desde donde se ejecuta el script."""
    save_dir = path or os.getcwd()
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"{model_name}.pkl")
    joblib.dump(model, file_path)
    print(f"Modelo guardado en: {file_path}")

def _safe_predict_proba(model, X):
    """Obtiene score continuo de un modelo para métricas AUC."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None

# ============================================================
# 1) Definir candidatos a modelos
# ============================================================
def build_models(random_state: int = 42):
    """Construye la lista de modelos candidatos para comparación."""

    # Logistic Regression
    logreg = LogisticRegression(max_iter=5000, class_weight="balanced")

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
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
    
    # Definir modelos de nivel 1
    level1_models = [
        ('xgboost', xgb),
        ('lightgbm', lgb),
        ('catboost', cat)
    ]

    # Meta-learner: Regresión Logística regularizada
    meta_learner = LogisticRegression(
        C=1.0,      #Regularización
        max_iter=1000,
        random_state=random_state
    )

    # Crear Stacking Classifier
    stacking_model = StackingClassifier(
        estimators=level1_models,
        final_estimator=meta_learner,
        cv=5,                        #Validación cruzada con el fin de evitar overfitting
        stack_method = 'predict_proba',     #Probabilidades
        n_jobs=-1
    )

    models = [
        ("logistic_regression", logreg),
        ("random_forest", rf),
        ("xgboost", xgb),
        ("lightgbm", lgb),
        ("catboost", cat),
        ("stacking_ensemble", stacking_model)
    ]
    
    return models


# 2) Calcular métricas de clasificación
# ============================================================

def summarize_classification(y_true, y_pred, y_score=None):
    """
    Métricas de clasificación para binario.
    - y_pred: predicción 0/1
    - y_score: score continuo (predict_proba[:,1] o decision_function) para ROC/PR AUC
    """
    metrics = {
        # Globales
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),

        # Clase 1 
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),

        # Clase 0 
        "precision_0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

    # AUCs (requieren score)
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    return metrics


# ============================================================
# 2) Optuna para el mejor (solo RF / XGB / CatBoost)
# ============================================================
def tune_best_model_with_optuna(best_name, X_train, y_train, cv_folds,
                                optimize_metric="recall_0", n_trials=50, random_state=42):
    """Optimiza hiperparámetros con Optuna para RF, XGB o CatBoost."""

    scorers = {
        "recall_0": make_scorer(recall_score, pos_label=0),
        "f1_0": make_scorer(f1_score, pos_label=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "accuracy": "accuracy",
        "balanced_acc": "balanced_accuracy",
    }
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

# ============================================================
# 3) Entrenar, evaluar y seleccionar modelo
# ============================================================
def train_and_select_model( X_train, y_train, X_test, y_test,random_state=42):
    """Entrena candidatos, selecciona por CV y evalúa el mejor en test."""
    models = build_models(random_state=random_state)

    scoring = {
        "recall_0": make_scorer(recall_score, pos_label=0),
        "f1_0": make_scorer(f1_score, pos_label=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "balanced_acc": "balanced_accuracy",
    }

    results = []
    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    for name, model in models:
        print(f'Entrenando y evaluando {name}..')
        
        # Validación cruzada en train: evalúa generalización
        # -------------------------
        cv_scores = cross_validate(
            model, X_train, y_train,
            cv=cv_folds,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        # 2) Entrenar modelo final con todos los datos de train
        # -------------------------
        # ---- Fit final ----
        start_fit = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - start_fit

        result = {
            "name": name,
            "model": model,

            # CV
            "cv_recall0_mean": float(cv_scores["test_recall_0"].mean()),
            "cv_recall0_std":  float(cv_scores["test_recall_0"].std()),
            "cv_f1_0_mean":    float(cv_scores["test_f1_0"].mean()),
            "cv_f1_0_std":     float(cv_scores["test_f1_0"].std()),

            "cv_pr_auc_mean":  float(cv_scores["test_pr_auc"].mean()),
            "cv_roc_auc_mean": float(cv_scores["test_roc_auc"].mean()),
            "cv_bal_acc_mean": float(cv_scores["test_balanced_acc"].mean()),

            # TEST: se evalua solo para el mejor final (blind test)
            "test_metrics": {},

            # tiempos 
            "train_time_s": float(train_time),
        }

        results.append(result)

        print(
            f"{name:>18} | "
            f"CV Recall0: {result['cv_recall0_mean']:.4f} (±{result['cv_recall0_std']:.4f}) | "
            f"Train: {train_time:.2f}s"
        )

    # Selección robusta por clase 0 (mean - std)
    best = max(results, key=lambda x: x["cv_recall0_mean"] - x["cv_recall0_std"])
    print(f"\n✅ Mejor modelo base (robusto por recall_0): {best['name']}")

    # Evaluacion en test SOLO del mejor modelo base
    y_pred_best = best["model"].predict(X_test)
    y_score_best = _safe_predict_proba(best["model"], X_test)
    best["test_metrics"] = summarize_classification(y_test, y_pred_best, y_score=y_score_best)

    # Optuna solo si aplica
    tuned_model = None
    tuned_info = None

    if best["name"] in {"random_forest", "xgboost", "catboost"}:
        best_params, best_cv_score = tune_best_model_with_optuna(
            best_name=best["name"],
            X_train=X_train,
            y_train=y_train,
            cv_folds=cv_folds,
            optimize_metric="recall_0",
            n_trials=50,
            random_state=random_state
        )

        print(f"\n🔧 Optuna mejor CV (recall_0): {best_cv_score:.4f}")
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
    

# ============================================================
# 5) Reporte: tabla resumen + gráficos comparativos
# ============================================================
def build_results_table(results):
    """Convierte resultados de entrenamiento en una tabla comparativa (solo CV/train)."""
    rows = []
    for r in results:
        rows.append({
            "model": r.get("name", "unknown"),

            # CV
            "cv_recall_0_mean": r.get("cv_recall0_mean", np.nan),
            "cv_recall_0_std":  r.get("cv_recall0_std", np.nan),
            "cv_pr_auc_mean":   r.get("cv_pr_auc_mean", np.nan),
            "cv_roc_auc_mean":  r.get("cv_roc_auc_mean", np.nan),
            "cv_bal_acc_mean":  r.get("cv_bal_acc_mean", np.nan),

            # tiempos
            "train_time_s":     r.get("train_time_s", np.nan),
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
        "test_pr_auc":      best_tm.get("pr_auc", np.nan),
        "test_roc_auc":     best_tm.get("roc_auc", np.nan),
        "test_bal_acc":     best_tm.get("balanced_acc", np.nan),
        "test_f1":          best_tm.get("f1", np.nan),
        "test_precision_0": best_tm.get("precision_0", np.nan),
        "test_recall_0":    best_tm.get("recall_0", np.nan),
        "test_f1_0":        best_tm.get("f1_0", np.nan),
    })

    if tuned_info is not None:
        tuned_tm = tuned_info.get("test_metrics", {}) or {}
        rows.append({
            "model": tuned_info.get("name", "tuned_model"),
            "test_pr_auc":      tuned_tm.get("pr_auc", np.nan),
            "test_roc_auc":     tuned_tm.get("roc_auc", np.nan),
            "test_bal_acc":     tuned_tm.get("balanced_acc", np.nan),
            "test_f1":          tuned_tm.get("f1", np.nan),
            "test_precision_0": tuned_tm.get("precision_0", np.nan),
            "test_recall_0":    tuned_tm.get("recall_0", np.nan),
            "test_f1_0":        tuned_tm.get("f1_0", np.nan),
        })

    df = pd.DataFrame(rows)
    for c in df.columns:
        if c != "model":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.round(4).reset_index(drop=True)


def plot_model_comparisons(df_results):
    """Grafica métricas clave para comparar modelos de forma visual."""
    metrics_to_plot = [
        ("cv_recall_0_mean", "CV Recall clase 0 (mean)"),
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

# ============================================================
# 4) MAIN
# ============================================================
if __name__ == "__main__":

    X_train_p, X_test_p, y_train, y_test, artifacts = run_feature_pipeline(
        target="pago_atiempo",
        test_size=0.2,
        split_type="random",
        random_state=42,
        stratify=True,
        drop_cols=["puntaje"],
        run_quality=False,
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

    print("\n" + "=" * 60)
    print(f"MEJOR MODELO BASE (por robust recall_0 CV): {best_model['name']}")
    print("CV (mean):",
          {
              "pr_auc": round(best_model.get("cv_pr_auc_mean", np.nan), 4),
              "roc_auc": round(best_model.get("cv_roc_auc_mean", np.nan), 4),
              "balanced_acc": round(best_model.get("cv_bal_acc_mean", np.nan), 4),
              "recall_0": round(best_model.get("cv_recall0_mean", np.nan), 4),
          })
    print("TEST:", {k: round(v, 4) for k, v in best_model["test_metrics"].items()})
    print("=" * 60)

    if tuned_info is not None:
        print("\n" + "=" * 60)
        print(f"MODELO TUNEADO: {tuned_info['name']}")
        print(f"Optuna best CV (recall_0): {tuned_info['best_optuna_cv']:.4f}")
        print("TEST:", {k: round(v, 4) for k, v in tuned_info["test_metrics"].items()})
        print("=" * 60)

    # Guardar mejor modelo base
    save_model(best_model["model"], best_model["name"], path="models")

    # Guardar modelo tuneado si existe
    if tuned_info is not None:
        save_model(tuned_info["model"], tuned_info["name"], path="models")
