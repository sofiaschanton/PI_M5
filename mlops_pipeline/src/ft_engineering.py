# feature_engineering.py: funciones para carga, limpieza, FE y preprocesamiento de datos.

# Librerías básicas
import re
import numpy as np
import pandas as pd
import warnings
from datetime import date
from typing import Optional, Tuple, Dict, Any, List
try:
    from .cargar_datos import cargar_datos
except ImportError:
    from cargar_datos import cargar_datos

# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

#-----------------------------------------------------
# 1. Cargar datos y renombrar columnas a snake_case
#-----------------------------------------------------
df = cargar_datos(verbose=True).copy()

def rename_cols(verbose: bool = False) -> pd.DataFrame:
    """Renombra columnas a snake_case."""

    # Renombrar columnas a snake_case
    df.columns = [
        re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", col)
        .replace(" ", "_")
        .replace("-", "_")
        .lower()
        for col in df.columns
    ]

    # Visualización rápida (solo si verbose=True)
    if verbose:
        print(df.head())
        df.info()

    return df

#-----------------------------------------------------
# 2. Separación entre features y target
#-----------------------------------------------------
def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) y variable objetivo (y) con validaciones básicas."""
    
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no existe. Columnas: {list(df.columns)}")
   
    y = df[target]
    X = df.drop(columns=[target])
    return X, y

#-----------------------------------------------------
# 3. Filtro de fecha: se utilizan los datos hasta hoy (14/02/2026) para entrenar y el dataset completo se utilizara en producción.
#-----------------------------------------------------
def filter_future_data(
    df: pd.DataFrame,
    date_col: str = "fecha_prestamo",
    cutoff_date: Optional[str] = "2025-12-31",  # fijo y reproducible
    verbose: bool = True
) -> pd.DataFrame:
    """
    Filtra el dataset dejando solo registros con fecha <= cutoff_date.
    - cutoff_date: "YYYY-MM-DD" 
      Si es None, usa la fecha de hoy real (date.today()).
    """
    if date_col not in df.columns:
        if verbose:
            print(f"[WARNING] No existe '{date_col}'. No se filtran datos futuros.")
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    cutoff = pd.to_datetime(cutoff_date) if cutoff_date is not None else pd.to_datetime(date.today())

    before = len(df)
    df = df[df[date_col].notna() & (df[date_col] <= cutoff)].copy()
    after = len(df)

    if verbose:
        print("\n[FILTRO FUTURO]")
        print(f"Cutoff: {cutoff.date()}")
        print(f"Antes: {before} | Después: {after} | Removidas: {before - after}")
        if after > 0:
            print(f"Rango fechas final: {df[date_col].min()} → {df[date_col].max()}")

    return df

#-----------------------------------------------------
# 4. Feature Engineering
#-----------------------------------------------------
def make_features(X: pd.DataFrame, drop_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Aplica limpieza y feature engineering sobre el conjunto de entrada."""
    X = X.copy()

    # Drop columnas para excluir columnas (ej: puntaje)
    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    # Fecha: crear features y borrar fecha original (evita leakage por timestamp directo)
    date_col = "fecha_prestamo"

    if date_col in X.columns:
        dt = pd.to_datetime(X[date_col], errors="coerce")
        
        if dt.notna().any():
            X[f"{date_col}_anio"] = dt.dt.year
            X[f"{date_col}_mes"] = dt.dt.month
            X[f"{date_col}_dia"] = dt.dt.day
            X[f"{date_col}_dow"] = dt.dt.dayofweek
            X[f"{date_col}_q"] = dt.dt.quarter

        X = X.drop(columns=[date_col], errors="ignore")


    # Limpieza edad: invalida outliers imposibles
    if "edad_cliente" in X.columns:
        age = pd.to_numeric(X["edad_cliente"], errors="coerce")
        age = age.where((age >= 18) & (age <= 100), np.nan)
        X["edad_cliente"] = age
    
    # Limpieza de puntajes negativos
    for c in ["puntaje", "puntaje_datacredito"]:
        if c in X.columns:
            s = pd.to_numeric(X[c], errors="coerce")
            X[c] = s.where(s >= 0, np.nan)

    # Limpieza de numéricos dentro de categóricas 
    def _nan_if_numeric_string(s: pd.Series) -> pd.Series:
        s = s.astype("string").str.strip()
        mask_num = s.str.fullmatch(r"-?\d+(\.\d+)?").fillna(False)
        s = s.mask(mask_num, other=np.nan)
        return s
    
    for c in ["tendencia_ingresos", "tipo_laboral"]:
        if c in X.columns:
            X[c] = _nan_if_numeric_string(X[c])
    
    # Normalización de categóricas: strip + espacios + minúsculas
    for c in ["tendencia_ingresos", "tipo_laboral"]:
        if c in X.columns:
            X[c] = (
                X[c]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)  # colapsa espacios dobles
                .str.lower()
            )

    # Forzar códigos numéricos a categórica (OHE)
    if "tipo_credito" in X.columns:
        tc = pd.to_numeric(X["tipo_credito"], errors="coerce")

        # si hay valores inválidos -> NaN
        # pasar a string categórica
        X["tipo_credito"] = tc.astype("Int64").astype("string")

        # Agrupar categorías raras (solo 4 y 9 explícitas; el resto OTROS)
        X["tipo_credito"] = X["tipo_credito"].where(X["tipo_credito"].isin(["4", "9"]), "OTROS")


    # Pasar columnas "string" a object para que np.nan sea el missing real
    for c in X.columns:
        if pd.api.types.is_string_dtype(X[c].dtype):
            X[c] = X[c].astype(object)

    # Reemplazar cualquier NA pandas por np.nan
    X = X.where(pd.notna(X), np.nan)

    return X

#-----------------------------------------------------
# 4. Transformer: QuantileCapper (aprende caps en TRAIN)
#-----------------------------------------------------
heavy_tail_cols = [
    "salario_cliente",
    "total_otros_prestamos",
    "capital_prestado",
    "cuota_pactada",
    "saldo_total",
    "saldo_principal",
    "saldo_mora",
    "saldo_mora_codeudor",
    "promedio_ingresos_datacredito"]

class QuantileCapper(BaseEstimator, TransformerMixin):
    """Clip por cuantiles aprendido en TRAIN y aplicado igual en TEST/NEW."""
    def __init__(self, cols=None, low_q=0.01, high_q=0.99):
        self.cols = cols
        self.low_q = low_q
        self.high_q = high_q

    def fit(self, X, y=None):
        X = X.copy()
        cols = self.cols or X.select_dtypes(include="number").columns

        self.caps_ = {}
        for c in cols:
            s = pd.to_numeric(X[c], errors="coerce").dropna()
            if s.empty:
                continue
            lo, hi = s.quantile([self.low_q, self.high_q])
            if lo < hi:
                self.caps_[c] = (float(lo), float(hi))
        return self

    def transform(self, X):
        X = X.copy()
        for c, (lo, hi) in self.caps_.items():
            if c in X.columns:
                X[c] = pd.to_numeric(X[c], errors="coerce").clip(lo, hi)
        return X
    
#-----------------------------------------------------
# 4. Preprocesador: num + ord + cat
#-----------------------------------------------------
 # Selectores (se ejecutan en runtime con el X que entra al pipeline)
def num_cols(X):
    return X.select_dtypes(include=["number"]).columns.tolist()

def ord_cols(X):
    return [c for c in ["tendencia_ingresos"] if c in X.columns]

def cat_cols(X):
    cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return [c for c in cols if c != "tendencia_ingresos"]

def build_preprocessor() -> ColumnTransformer:
    """Define el preprocesador por tipo de variable (numérica, ordinal y categórica)."""

    # Orden de 'tendencia_ingresos' 
    ord_order = [["decreciente", "estable", "creciente", "desconocido"]]

    #  Pipeline numéricos a imputar con median.
    num_pipe =  Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler())
            ])
            
    # Pipeline categóricos ordinales (tendencia_ingresos) con OrdinalEncoder.
    ord_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="desconocido")),
        ("ord", OrdinalEncoder(
            categories=ord_order,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])
    
    # Pipeline categóricos a imputar con most_frequent y OneHotEncoder.
    cat_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(drop='first',handle_unknown="ignore", sparse_output=False))
            ])
    

    # ColumnTransformer para aplicar pipelines específicos a cada tipo de columna.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),   
            ("ord", ord_pipe, ord_cols),
            ("cat", cat_pipe, cat_cols), 
        ],
        remainder="drop",
        verbose_feature_names_out=False
        )
    
    return preprocessor

#-----------------------------------------------------
# 5. Pipeline completo: FE + Preprocess
#-----------------------------------------------------
def build_data_pipeline(
        drop_cols: Optional[List[str]] = None,
        cap_cols: Optional[List[str]] = None,
        low_q: float = 0.01,
        high_q: float = 0.99
    ) -> Pipeline:
    """Crea pipeline completo de FE, CAP y preprocesamiento."""
    
    fe = FunctionTransformer(
        make_features,
        validate=False,
        kw_args={"drop_cols": ([] if drop_cols is None else list(drop_cols))}
    )

    capper = QuantileCapper(
        cols=(cap_cols if cap_cols is not None else heavy_tail_cols),
        low_q=low_q,
        high_q=high_q,
   )

    return Pipeline(steps=[
        ("fe", fe),
        ("cap", capper),
        ("pre", build_preprocessor()),
    ])

#-----------------------------------------------------
# 6. Split train/test random/temporal
#-----------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    split_type: str = "temporal",   # "random" | "temporal"
    random_state: int = 42,
    stratify: bool = True,
    verbose: bool = True,
    date_col: str = "fecha_prestamo",
):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.

    Parámetros:
    X : pd.DataFrame
        Matriz de features.
    y : pd.Series
        Variable objetivo.
    test_size : float, default=0.2
        Proporción de datos destinada a test (entre 0 y 1).
    split_type : str, default="random"
        Tipo de split:
        - "random": usa train_test_split (opcionalmente estratificado).
        - "temporal": divide por orden temporal (requiere DatetimeIndex).
    random_state : int, default=42
        Semilla para reproducibilidad (solo en split random).
    stratify : bool, default=True
        Si True y split_type="random", mantiene proporción de clases.
    verbose : bool, default=True
        Si True, imprime información del split y balance de clases.

    Retorna:
    --------
    X_train, X_test : pd.DataFrame
        Features de entrenamiento y prueba.
    y_train, y_test : pd.Series
        Target de entrenamiento y prueba.

    Notas:
    ------
    - El split temporal simula un escenario real: entrena con el pasado y evalúa en el futuro.
    - El split random estratificado es útil cuando el target está desbalanceado.
    """

    if not (0 < test_size < 1):
        raise ValueError("test_size debe estar entre 0 y 1 (ej: 0.2).")

    if len(X) != len(y):
        raise ValueError(f"X e y tienen distinta longitud: len(X)={len(X)} vs len(y)={len(y)}")

    if split_type == "random":
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if stratify else None
        )

    if split_type == "temporal" and verbose:
        print("temporal split: random_state y stratify se ignoran (split por fecha, sin mezclar pasado/futuro).")
        # Temporal: NO stratify
        if date_col not in X.columns:
            raise KeyError(f"Para split temporal, X debe tener la columna '{date_col}'.")

        X_ = X.copy()
        y_ = y.copy()

        X_[date_col] = pd.to_datetime(X_[date_col], errors="coerce")
        mask = X_[date_col].notna()

        X_ = X_.loc[mask].copy()
        y_ = y_.loc[mask].copy()

        order = np.argsort(X_[date_col].values)
        X_ = X_.iloc[order].reset_index(drop=True)
        y_ = y_.iloc[order].reset_index(drop=True)

        cut = int(len(X_) * (1 - test_size))
        X_train, X_test = X_.iloc[:cut].copy(), X_.iloc[cut:].copy()
        y_train, y_test = y_.iloc[:cut].copy(), y_.iloc[cut:].copy()

        if verbose:
            print(f"[temporal split] train={X_train.shape} test={X_test.shape}")
            print("train max date:", X_train[date_col].max())
            print("test  min date:", X_test[date_col].min())

        return X_train, X_test, y_train, y_test

    raise ValueError("split_type debe ser 'random' o 'temporal'")


#-----------------------------------------------------
# 7. Función que construye el pipeline completo.
#-----------------------------------------------------
def ft_pipeline(
    target: str = "pago_atiempo",
    test_size: float = 0.2,
    split_type: str = "temporal",   # "random" | "temporal"
    random_state: int = 42,
    stratify: bool = True,
    drop_cols: Optional[list] = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Ejecuta el flujo completo de feature engineering + preprocesamiento:
    1) Carga y renombra columnas
    2) Cutoff de fechas
    3) Split X/y
    4) Split train/test (random o temporal)
    5) Fit/transform del pipeline SOLO en train (evita leakage)
    6) Devuelve X_train/X_test ya preprocesados como DataFrames con nombres de features
    """

    if verbose:
        print("\n" + "=" * 60)
        print("INICIO PIPELINE: FEATURE ENGINEERING + PREPROCESS")
        print("=" * 60)

    # 1) Carga y renombre de columnas
    df = cargar_datos(verbose=True).copy()
    df = rename_cols(verbose=verbose)
     
    # 2) Cutoff de fechas (usa default "2026-02-14" definido en filter_future_data)
    df = filter_future_data(df, date_col="fecha_prestamo", verbose=verbose)

    if verbose:
        print("\n[1/7] Carga de datos + renombrado a snake_case")
        print(f"Shape inicial : {df.shape[0]} filas x {df.shape[1]} columnas")

    # (siempre) en temporal no hay stratify
    if split_type == "temporal":
        stratify = False
        if verbose: 
            print("\n[2/7] Split temporal (stratify desactivado)")
            if "fecha_prestamo" in df.columns:
                fp = pd.to_datetime(df["fecha_prestamo"], errors="coerce")
                print(f"Columna temporal: fecha_prestamo")
                print(f"Rango fechas: {fp.min()}  →  {fp.max()}")
            else:
                print("WARNING: no existe 'fecha_prestamo' en df (split temporal fallará)")

    # 3) Split X/y (desde el MISMO df ya ordenado si temporal)
    X, y = split_features_target(df, target=target)
    if verbose:
        print("\n[3/7] Separación features (X) y target (y)")
        print(f"      Target: '{target}'")
        print(f"      X shape: {X.shape} | y shape: {y.shape}")

    # 4) Split train/test
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=test_size,
        split_type=split_type,
        random_state=random_state,
        stratify=stratify,
        verbose=verbose
    )
    if verbose:
        print("\n[4/7] Split train/test")
        print(f"      Tipo split : {split_type}")
        print(f"      test_size  : {test_size}")

        if split_type == "random":
            print(f"      stratify   : {bool(stratify)}")
            print(f"      seed       : {random_state}")

        print(f"      Train shape: {X_train.shape} | Test shape: {X_test.shape}")

        # Distribución de clases
        dist_train = y_train.value_counts(normalize=True).round(4).to_dict()
        dist_test  = y_test.value_counts(normalize=True).round(4).to_dict()
        print("      Distribución clases (proporción)")
        print(f"      - Train: {dist_train}")
        print(f"      - Test : {dist_test}")

        if split_type == "temporal":
            print(f"      Train max fecha_prestamo: {X_train['fecha_prestamo'].max()}")
            print(f"      Test  min fecha_prestamo: {X_test['fecha_prestamo'].min()}")


    # Asegurar alineación (por si algo tocó índices)
    X_train, y_train = X_train.align(y_train, axis=0, join="inner")
    X_test,  y_test  = X_test.align(y_test, axis=0, join="inner")

    # Asserts críticos
    assert len(X_train) == len(y_train), f"Desalineado train: X={len(X_train)} y={len(y_train)}"
    assert len(X_test) == len(y_test), f"Desalineado test: X={len(X_test)} y={len(y_test)}"
    assert target not in X_train.columns
    assert target not in X_test.columns

    # 5) Pipeline preprocess (fit SOLO en train)
    pipe = build_data_pipeline(drop_cols=drop_cols)

    if verbose:
        print("\n[5/7] Fit del pipeline SOLO en train (evita leakage)")
        print("      Pipeline: FE (FunctionTransformer) + CAP (QuantileCapper) + Preprocess (ColumnTransformer)")

    X_train_arr = pipe.fit_transform(X_train, y_train)
    X_test_arr  = pipe.transform(X_test)

    # Feature names
    try:
        feature_names = pipe.named_steps["pre"].get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(X_train_arr.shape[1])])

    
    if verbose:
        print("\n[6/7] Transformación completada")
        print(f"      Features finales: {len(feature_names)}")

    # 6) Dataframes finales
    X_train_p = pd.DataFrame(X_train_arr, columns=feature_names).reset_index(drop=True)
    X_test_p  = pd.DataFrame(X_test_arr,  columns=feature_names).reset_index(drop=True)

    if verbose:
        print("\n[6/7] Transformación completada")
        print(f"      Features finales: {len(feature_names)}")
        print(f"      X_train_p: {X_train_p.shape} | X_test_p: {X_test_p.shape}")


    # y también lo reseteamos para mantener consistencia
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    artifacts = {
        "pipeline": pipe,
        "preprocessor": pipe.named_steps["pre"],
        "feature_names": list(feature_names),
        "split_config": {
            "test_size": test_size,
            "split_type": split_type,
            "random_state": random_state,
            "stratify": bool(stratify) if split_type == "random" else False,
        },
        "class_balance_train": y_train.value_counts(normalize=True).to_dict(),
        "class_balance_test": y_test.value_counts(normalize=True).to_dict(),
        "zeros_train": int((y_train == 0).sum()),
        "zeros_test": int((y_test == 0).sum()),
        # trazabilidad del cutoff
        "cutoff_date": "2026-02-14",
        "rows_after_cutoff_filter": int(len(df)),
    }

    if verbose:
        print("\n[7/7] Artefactos generados")
        print("      - pipeline (fe + cap + preprocessor)")
        print("      - feature_names")
        print("      - split_config + balance de clases")
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETADO")
        print("=" * 60)

    return X_train_p, X_test_p, y_train, y_test, artifacts


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    X_train_p, X_test_p, y_train, y_test, artifacts = ft_pipeline(
        target="pago_atiempo",
        test_size=0.2,
        split_type="temporal",
        random_state=42,
        stratify=True,
        drop_cols= [],
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("EJECUCIÓN LOCAL OK")
    print("=" * 60)
    print("X_train_p:", X_train_p.shape)
    print("X_test_p :", X_test_p.shape)
    print("Balance train:", artifacts["class_balance_train"])
    print("Balance test :", artifacts["class_balance_test"])
    print("Total features out:", len(artifacts["feature_names"]))
    print("Cutoff usado:", artifacts["cutoff_date"])
    print("Filas luego del filtro:", artifacts["rows_after_cutoff_filter"])