# feature_engineering.py: funciones para carga, limpieza, FE y preprocesamiento de datos.

# Librerías básicas
import numpy as np
import pandas as pd
import re
from cargar_datos import cargar_datos
import warnings
from typing import Optional, Tuple, Dict, Any

# Sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split

# 1. Cargar datos y renombrar columnas a snake_case
def load_and_rename_data(verbose: bool = False) -> pd.DataFrame:
    """Carga los datos usando cargar_datos() y renombra columnas a snake_case."""

    df = cargar_datos()
    df = df.copy()

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


# 2. Evaluación de calidad de datos.
def evaluate_data_quality(
    df: pd.DataFrame,
    umbral_nulos: float = 30.0,
    verbose: bool = True,
    check_mixed_types: bool = True,
    sample_size: int = 50_000,
    random_state: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Evalúa calidad del dataset:
      - nulos (conteo y %)
      - columnas que superan un umbral de nulos
      - dtypes de pandas
      - columnas con tipos internos mezclados (opcional; puede ser costoso)

    Retorna un dict con dataframes de resultados.
    """
    df = df.copy()

    # Valores nulos y porcentaje
    missing_count = df.isna().sum()
    missing_pct = (df.isna().mean() * 100).round(2)

    missing_df = (
        pd.DataFrame({"faltantes": missing_count, "%": missing_pct})
        .sort_values("%", ascending=False)
    )

    cols_missing_umbral = missing_df[missing_df["%"] > umbral_nulos]

    # Tipos pandas
    tipos_df = df.dtypes.astype(str).to_frame("dtype").sort_values("dtype")

    def tipos_internos(df_: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in df_.columns:
            s = df_[col].dropna()
            if s.empty:
                continue
            if len(s) > sample_size:
                s = s.sample(sample_size, random_state=random_state)
            tipos = s.map(type).value_counts()
            if len(tipos) > 1:
                rows.append({
                    "columna": col,
                    "dtype_pandas": str(df_[col].dtype),
                    "tipos_internos": {t.__name__: int(n) for t, n in tipos.items()}
                })
        return pd.DataFrame(rows)

    mix = tipos_internos(df) if check_mixed_types else pd.DataFrame()

    if verbose:
        if not cols_missing_umbral.empty:
            print(f"\nColumnas con más del {umbral_nulos}% de nulos:")
            print(cols_missing_umbral)
        else:
            print(f"\nNo hay columnas con más del {umbral_nulos}% de nulos.")

        print("\nColumnas con tipos internos mezclados:")
        print("No se detectaron columnas con tipos mezclados." if mix.empty else mix)

    return {
        "missing": missing_df,
        "cols_muchos_nulos": cols_missing_umbral,
        "tipos": tipos_df,
        "mix_types": mix
    }

# 3. Ordenar por fecha_prestamo (ascendente).
def temporal_order(df: pd.DataFrame, date_col: str = "fecha_prestamo") -> pd.DataFrame:
    """Ordena el dataset por fecha y asegura tipo datetime en la columna temporal."""
    df = df.copy()

    if date_col not in df.columns:
        raise KeyError(f"No existe la columna '{date_col}' en el DataFrame.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    # ordenar y dejar índice limpio (evita desalineación)
    df = df.sort_values(date_col).reset_index(drop=True)

    # check
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        raise TypeError("La columna temporal no quedó en datetime.")

    return df

# 4. Separación entre features y target
def split_features_target(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa features (X) y variable objetivo (y) con validaciones básicas."""
    if not isinstance(target, str):
        raise TypeError("target debe ser str (nombre de columna).")

    if target not in df.columns:
        raise ValueError(f"Target '{target}' no existe. Columnas: {list(df.columns)}")

    y = df[target]
    X = df.drop(columns=[target])
    return X, y

# 5. Feature Engineering helpers: 
def build_fe(
    X: pd.DataFrame,
    drop_cols: Optional[list] = None
) -> pd.DataFrame:
    """Aplica limpieza y feature engineering sobre el conjunto de entrada."""
    X = X.copy()

    # Drop columns (ej: puntaje)
    if drop_cols:
        X = X.drop(columns=[c for c in drop_cols if c in X.columns], errors="ignore")

    #  FE de fecha 
    date_col = "fecha_prestamo"
    dt = None

    # Caso 1: fecha como columna
    if date_col in X.columns:
        X[date_col] = pd.to_datetime(X[date_col], errors="coerce")
        dt = X[date_col]

    # Caso 2: fecha como índice datetime
    elif isinstance(X.index, pd.DatetimeIndex):
        # convertir a serie datetime alineada a X
        dt = pd.to_datetime(pd.Series(X.index, index=X.index), errors="coerce")

    # Crear features si dt existe (y no es todo NaT)
    if dt is not None and dt.notna().any():
        X[f"{date_col}_anio"] = dt.dt.year.astype("Int64")
        X[f"{date_col}_mes"]  = dt.dt.month.astype("Int64")
        X[f"{date_col}_dia"]  = dt.dt.day.astype("Int64")
        X[f"{date_col}_dow"]  = dt.dt.dayofweek.astype("Int64")

        # Dropear fecha original si era columna
        if date_col in X.columns:
            X = X.drop(columns=[date_col], errors="ignore")

        # Si venía como índice datetime, limpiar índice
        if isinstance(X.index, pd.DatetimeIndex):
            X = X.reset_index(drop=True)


    # Limpieza edad 
    col_age = "edad_cliente"
    if col_age in X.columns:
        X[col_age] = pd.to_numeric(X[col_age], errors="coerce")
        X.loc[(X[col_age] < 18) | (X[col_age] > 100), col_age] = np.nan

    # Limpieza numéricos "puros" dentro de categóricas 
    def clean_numeric_in_categorical(df_: pd.DataFrame, col: str, numeric_regex: str = r"-?\d+(\.\d+)?"):
        if col not in df_.columns:
            return df_
        s = df_[col].astype("string").str.strip()
        mask_numeric = s.str.fullmatch(numeric_regex).fillna(False)
        df_.loc[mask_numeric, col] = np.nan
        return df_

    for c in [ "tendencia_ingresos", "tipo_laboral"]:
        X = clean_numeric_in_categorical(X, c)
    
    # Normalización de categóricas: strip + espacios + minúsculas
    cat_norm_cols = ["tendencia_ingresos", "tipo_laboral"]

    for c in cat_norm_cols:
        if c in X.columns:
            X[c] = (
                X[c]
                .astype("string")
                .str.strip()
                .str.replace(r"\s+", " ", regex=True)  # colapsa espacios dobles
                .str.lower()
            )

    # EDA-based FE
    def to_num(s: pd.Series) -> pd.Series:
        return pd.to_numeric(s, errors="coerce")

    # Saldos: NaN -> 0 (si no hay info, se asume que no tiene saldo)
    saldo_cols = ["saldo_mora", "saldo_total", "saldo_principal", "saldo_mora_codeudor"]
    for c in saldo_cols:
        if c in X.columns:
            X[c] = to_num(X[c]).fillna(0)

    # Flags
    if "saldo_mora" in X.columns:
        mora = X["saldo_mora"]
        X["tiene_mora"] = (mora != 0).astype("Int64")
        X["mora_pos"]   = (mora > 0).astype("Int64")
        X["mora_neg"]   = (mora < 0).astype("Int64")

    if "saldo_total" in X.columns:
        st = X["saldo_total"]
        X["tiene_saldo"] = (st != 0).astype("Int64")
        X["saldo_pos"]   = (st > 0).astype("Int64")
        X["saldo_neg"]   = (st < 0).astype("Int64")

    if "saldo_mora_codeudor" in X.columns:
        mc = X["saldo_mora_codeudor"]
        X["tiene_mora_codeudor"] = (mc != 0).astype("Int64")
        X["mora_codeudor_pos"]   = (mc > 0).astype("Int64")
        X["mora_codeudor_neg"]   = (mc < 0).astype("Int64")

    # Ratios (evitar división por 0)
    if "salario_cliente" in X.columns:
        salario = to_num(X["salario_cliente"])
        salario_safe = salario.replace(0, np.nan)

        if "cuota_pactada" in X.columns:
            cuota = to_num(X["cuota_pactada"])
            X["ratio_cuota_salario"] = (cuota / salario_safe).astype(float)

        if "saldo_total" in X.columns:
            saldo_total = to_num(X["saldo_total"])
            X["ratio_saldo_salario"] = (saldo_total / salario_safe).astype(float)

    #  Fix types 
    int_cols = [
        "plazo_meses", "edad_cliente", "cuota_pactada", "cant_creditosvigentes",
        "huella_consulta", "creditos_sector_financiero", "creditos_sector_cooperativo",
        "creditos_sector_real",
        "fecha_prestamo_anio", "fecha_prestamo_mes", "fecha_prestamo_dia", "fecha_prestamo_dow",
        "tiene_mora", "mora_pos", "mora_neg",
        "tiene_saldo", "saldo_pos", "saldo_neg",
        "tiene_mora_codeudor", "mora_codeudor_pos", "mora_codeudor_neg",
    ]

    float_cols = [
        "capital_prestado", "salario_cliente", "total_otros_prestamos", "puntaje",
        "puntaje_datacredito", "saldo_mora", "saldo_total", "saldo_principal",
        "saldo_mora_codeudor", "promedio_ingresos_datacredito",
        "ratio_cuota_salario", "ratio_saldo_salario",
    ]

    cat_cols = ["tipo_credito", "tendencia_ingresos", "tipo_laboral"]

    for c in int_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("Int64")


    for c in float_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)

    for c in cat_cols:
        if c in X.columns:
            # 1) pd.NA -> np.nan (para que sklearn no explote)
            X[c] = X[c].astype("object")
            X[c] = X[c].where(pd.notnull(X[c]), np.nan)

            # 2) ahora sí: category
            X[c] = X[c].astype("category")

    X = X.convert_dtypes()
    X = X.replace({pd.NA: np.nan, None: np.nan})
    return X

# 7. Preprocesamiento
def build_preprocessor():
    """Define el preprocesador por tipo de variable (numérica, ordinal y categórica)."""

    # Selectores (se ejecutan en runtime con el X que entra al pipeline)
    def num_cols(X):
        return X.select_dtypes(include=["number"]).columns.tolist()

    def ord_cols(X):
        return [c for c in ["tendencia_ingresos"] if c in X.columns]

    def cat_cols(X):
        cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        return [c for c in cols if c != "tendencia_ingresos"]

    # Orden de 'tendencia_ingresos' 
    ord_order = [["decreciente", "estable", "creciente", "desconocido"]]

    #  Pipeline numéricos a imputar con median.
    num_pipe =  Pipeline(steps=[
                ("imputer", SimpleImputer(missing_values=np.nan,strategy="median")),
                ("scaler", RobustScaler())
            ])
            
    # Pipeline categóricos ordinales (tendencia_ingresos) con OrdinalEncoder.
    ord_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(missing_values=np.nan,strategy="constant", fill_value="desconocido")),
        ("ord", OrdinalEncoder(
            categories=ord_order,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        ))
    ])
    
    # Pipeline categóricos a imputar con most_frequent y OneHotEncoder.
    cat_pipe = Pipeline(steps=[
                ("imputer", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
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

# 8. Pipeline completo: FE + Preprocess
def build_data_pipeline(drop_cols: Optional[list] = None) -> Pipeline:
    """Crea pipeline completo de FE y preprocesamiento."""
    fe = FunctionTransformer(
        build_fe,
        validate=False,
        kw_args={"drop_cols": drop_cols or []}
    )
    return Pipeline(steps=[
        ("fe", fe),
        ("pre", build_preprocessor()),
    ])

# 9. Train/test split random/temporal
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    split_type: str = "random",   # "random" | "temporal"
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

    if split_type == "temporal":
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


# 10. Función que construye el pipeline completo.
def run_feature_pipeline(
    target: str = "pago_atiempo",
    test_size: float = 0.2,
    split_type: str = "random",   # "random" | "temporal"
    random_state: int = 42,
    stratify: bool = True,
    drop_cols: Optional[list] = None,
    run_quality: bool = False,
    umbral_nulos: float = 30.0,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Ejecuta el flujo completo de feature engineering + preprocesamiento:
    1) Carga y renombra columnas
    2) (Opcional) reporte de calidad
    3) (Opcional) set de índice temporal si split_type="temporal"
    4) Split X/y
    5) Split train/test (random o temporal)
    6) Fit/transform del pipeline SOLO en train (evita leakage)
    7) Devuelve X_train/X_test ya preprocesados como DataFrames con nombres de features
    """

    if verbose:
        print("\n[1/7] Cargando y renombrando dataset...")

    df = load_and_rename_data(verbose=False)

    if verbose:
        print(f"      Shape inicial: {df.shape}")
        print(f"      Columnas: {len(df.columns)}")

    quality_report = None
    if run_quality:
        quality_report = evaluate_data_quality(df, umbral_nulos=umbral_nulos, verbose=verbose)

    # ---------
    # Garantizar que fecha sea datetime para ordenar si temporal.
    # ---------
    
    if split_type == "temporal":
        if "fecha_prestamo" not in df.columns:
            raise KeyError("No existe 'fecha_prestamo' para split temporal.")
        df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")
        df = df.dropna(subset=["fecha_prestamo"]).sort_values("fecha_prestamo").reset_index(drop=True)

        # En temporal, stratify no aplica
        stratify = False

    # Split X/y (desde el MISMO df ya ordenado si temporal)
    if verbose:
        print("\n[2/7] Separando features y target...")
        print(f"      Target: {target}")

    X, y = split_features_target(df, target=target)


    # Split train/test
    X_train, X_test, y_train, y_test = split_data(
        X, y,
        test_size=test_size,
        split_type=split_type,
        random_state=random_state,
        stratify=stratify,
        verbose=verbose
    )
    if verbose:
        print("\n[3/7] División train/test completada")
        print(f"      Train shape: {X_train.shape}")
        print(f"      Test shape : {X_test.shape}")
        print("\nDistribución de clases:")
        print("      Train:", y_train.value_counts(normalize=True).round(3).to_dict())
        print("      Test :", y_test.value_counts(normalize=True).round(3).to_dict())

    # ---------
    # FE (aplicarlo en train y test)
    # ---------
    X_train = build_fe(X_train, drop_cols=drop_cols or [])
    X_test  = build_fe(X_test,  drop_cols=drop_cols or [])

    if verbose:
        print("\n[4/7] Feature Engineering aplicado")
        print(f"      Nuevas columnas train: {X_train.shape[1]}")

    # Asegurar alineación (por si algo tocó índices)
    X_train, y_train = X_train.align(y_train, axis=0, join="inner")
    X_test,  y_test  = X_test.align(y_test, axis=0, join="inner")

    # Asserts críticos
    assert len(X_train) == len(y_train), f"Desalineado train: X={len(X_train)} y={len(y_train)}"
    assert len(X_test) == len(y_test), f"Desalineado test: X={len(X_test)} y={len(y_test)}"
    assert target not in X_train.columns
    assert target not in X_test.columns

    # Pipeline preprocess (fit SOLO en train)
    pipe = build_data_pipeline(drop_cols=drop_cols)

    if verbose:
        print("\n[5/7] Fitteando preprocesador SOLO en train...")

    X_train_arr = pipe.fit_transform(X_train)
    X_test_arr  = pipe.transform(X_test)

    # Feature names
    try:
        feature_names = pipe.named_steps["pre"].get_feature_names_out()
    except Exception:
        feature_names = np.array([f"feature_{i}" for i in range(X_train_arr.shape[1])])
    
    if verbose:
        print("\n[6/7] Transformación completada")
        print(f"      Features finales: {len(feature_names)}")

    # RangeIndex para evitar líos por DatetimeIndex
    X_train_p = pd.DataFrame(X_train_arr, columns=feature_names).reset_index(drop=True)
    X_test_p  = pd.DataFrame(X_test_arr,  columns=feature_names).reset_index(drop=True)

    # y también lo reseteamos para mantener consistencia
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)

    artifacts = {
        "pipeline": pipe,
        "preprocessor": pipe.named_steps["pre"],
        "feature_names": list(feature_names),
        "quality_report": quality_report,
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
    }

    if verbose:
        print(f"OK: X_train_p = {X_train_p.shape}")
        print(f"OK: X_test_p  = {X_test_p.shape}")
        print(f"OK: total_features_out = {len(feature_names)}")
        print("=" * 60)
        print("PIPELINE COMPLETADO")
        print("=" * 60)

    return X_train_p, X_test_p, y_train, y_test, artifacts


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    X_train_p, X_test_p, y_train, y_test, artifacts = run_feature_pipeline(
        target="pago_atiempo",
        test_size=0.2,
        split_type="random",
        random_state=42,
        stratify=True,
        drop_cols= ['puntaje'],
        run_quality=False,
        verbose=True
    )
    
    print("\nListo ")
    print("X_train_p:", X_train_p.shape)
    print("X_test_p :", X_test_p.shape)
    print("Balance train:", artifacts["class_balance_train"])
    print("Balance test :", artifacts["class_balance_test"])
