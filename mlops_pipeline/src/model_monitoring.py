# model_monitoring.py
#
# Script de monitoreo de data drift para un modelo de ML:
# - Comparar un conjunto de referencia (histórico) con uno nuevo (reciente).
# - Medir drift en variables originales ("raw") y en variables preprocesadas
#   (el espacio real en el que opera el modelo).
# - Visualizar resultados y dar una recomendación automática (semáforo final).


#----------------------------------------------
# 0. Importar librerías
#----------------------------------------------
import numpy as np
import pandas as pd
import streamlit as st      # librería para crear aplicaciones web interactivas
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Tests estadísticos usados para drift
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon

# Funciones del pipeline de features.
from ft_engineering import (
    rename_cols,                # Carga dataset Base_de_datos.xlsx y renombra columnas a snake case
    split_features_target,      # Divide feature de target
    split_data,                 # Divide los datos en train y test, puede ser temporal o rando,
    num_cols, cat_cols, ord_cols              
)

#----------------------------------------------
# 1. Configuración de la página de Streamlit
#----------------------------------------------
st.set_page_config(page_title="Monitoreo de Data Drift", layout="wide")

DATE_COL = "fecha_prestamo"
TARGET = "pago_atiempo"

# Sidebar
# st.sidebar.header("Configuración de Datos")
# split_type_ui = st.sidebar.selectbox("Tipo de split", ["temporal", "random"], index=0)
# ref_ratio = st.sidebar.slider("Proporción referencia (ref_ratio)", 0.4, 0.95, 0.70, 0.01)
# random_state = st.sidebar.number_input("Random state (solo random)", min_value=0, value=42, step=1)
# use_stratify = st.sidebar.checkbox("Stratify (solo random)", value=True)

st.sidebar.header("Parámetros de drift")
alpha = st.sidebar.slider("P-value (alpha)", 0.001, 0.2, 0.05, 0.001)  # usar alpha explícito en todo el flujo

null_warn = st.sidebar.number_input("Δ nulos warn (proporción)", value=0.25)
null_crit = st.sidebar.number_input("Δ nulos crit (proporción)", value=0.50)
psi_bins = st.sidebar.slider("Bins PSI", 5, 30, 10, 1)
jsd_bins  = st.sidebar.slider("Bins JSD (numéricas)", 5, 50, 20, 1)

# Umbrales 
# - warn: alerta moderada
# - crit: alerta crítica
psi_warn= st.sidebar.number_input("PSI warn", value=0.10, step=0.01, format="%.2f")
psi_crit = st.sidebar.number_input("PSI crit", value=0.25, step=0.01, format="%.2f")

jsd_warn = st.sidebar.number_input("JSD warn", value=0.10, step=0.01, format="%.2f")
jsd_crit = st.sidebar.number_input("JSD crit", value=0.20, step=0.01, format="%.2f")

#----------------------------------------------
# 2. Funciones de cálculo
#----------------------------------------------

def calculate_psi(ref, new, feature, n_bins=10):

    # Crear bins usando percentiles de referencia
    _,bins = pd.qcut(ref,q=n_bins, duplicates='drop', retbins=True)

    # Contar observaciones en cada bin
    ref_counts = pd.cut(ref, bins=bins, include_lowest=True).value_counts().sort_index()
    new_counts = pd.cut(new, bins=bins, include_lowest=True).value_counts().sort_index()

    # Convertir a proporciones
    ref_pct = ref_counts / ref_counts.sum()
    new_pct = new_counts / new_counts.sum()

    # PSI: suma ponderada de diferencias logarítmicas
    psi = np.sum((new_pct - ref_pct)*np.log((new_pct + 1e-10)/(ref_pct + 1e-10)))
    return psi

def calculate_psi_cat(ref, new, feature):
    ref_counts = ref.value_counts().sort_index()
    new_counts = new.value_counts().sort_index()

    all_categories = sorted(set(ref_counts.index) | set(new_counts.index))
    ref_counts = ref_counts.reindex(all_categories, fill_value=0)
    new_counts = new_counts.reindex(all_categories, fill_value=0)

    ref_pct = ref_counts.values / (ref_counts.values.sum() + 1e-12)
    new_pct = new_counts.values / (new_counts.values.sum() + 1e-12)

    psi = np.sum((new_pct - ref_pct) * np.log((new_pct + 1e-10) / (ref_pct + 1e-10)))
    return float(psi)


def calculate_jsd (ref, new, feature, n_bins=10):

    # Crear histogramas normalizados
    ref_hist, bins = np.histogram(ref, bins=n_bins)
    new_hist, _ = np.histogram(new, bins=bins)

    # Normalizar a proporciones (distribuciones de probabilidad)
    ref_pct = ref_hist / ref_hist.sum()
    new_pct = new_hist / new_hist.sum()

    # Jensem-Shannon divergence
    jsd = jensenshannon(ref_pct, new_pct)
    return float(jsd)

def calculate_jsd_cat(ref, new, feature):
    ref_counts = ref.value_counts().sort_index()
    new_counts = new.value_counts().sort_index()

    all_categories = sorted(set(ref_counts.index) | set(new_counts.index))
    ref_counts = ref_counts.reindex(all_categories, fill_value=0)
    new_counts = new_counts.reindex(all_categories, fill_value=0)

    ref_pct = ref_counts.values / (ref_counts.values.sum() + 1e-12)
    new_pct = new_counts.values / (new_counts.values.sum() + 1e-12)

    return float(jensenshannon(ref_pct, new_pct))

def calculate_chi_square(ref, new, feature):

    # Contar frecuencias
    ref_counts = ref.value_counts().sort_index()
    new_counts = new.value_counts().sort_index()

    # Alinear categorías (en caso de que falten algunas)
    all_categories = sorted(set(ref_counts.index) | set(new_counts.index))
    ref_counts = ref_counts.reindex(all_categories,fill_value=0)
    new_counts = new_counts.reindex(all_categories,fill_value=0)

    # Tabla de contingencia
    contingency_table = np.array([ref_counts.values, new_counts.values])

    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return chi2, p_value, dof
    
# -----------------------------------------
# Detección de drift
# -----------------------------------------
def detect_drift(
    X_ref: pd.DataFrame,
    X_new: pd.DataFrame,
    alpha: float = 0.05,
    excluded_cols=None,
    psi_bins: int = 10,
    jsd_bins: int = 10,
    psi_warn: float = 0.10,
    psi_crit: float = 0.25,
    jsd_warn: float = 0.10,
    jsd_crit: float = 0.20,
    null_warn: float = 0.20,
    null_crit: float = 0.50
) -> pd.DataFrame:
    """
    DATA DRIFT por feature (RAW o PRE).
    - Numéricas: KS + PSI + JSD
    - Categóricas: Chi2 + PSI_cat + JSD_cat
    Devuelve: métricas + drift (p<alpha) + alert_psi/alert_jsd
    """

    excluded_cols = set(excluded_cols or [])
    common_cols = [c for c in X_ref.columns if c in X_new.columns and c not in excluded_cols]

    rows = []
    num_set = set(num_cols(X_ref))
    cat_set = set(cat_cols(X_ref)) | set(ord_cols(X_ref))

    for col in common_cols:
        #  Drift de nulos 
        nulls_ref = float(X_ref[col].isna().mean())
        nulls_new = float(X_new[col].isna().mean())
        delta_nulls = float(nulls_new - nulls_ref)
        abs_delta_nulls = abs(delta_nulls)

        alert_nulls = "crit" if abs_delta_nulls >= null_crit else (
            "warn" if abs_delta_nulls >= null_warn else "ok"
        )
        a = X_ref[col].dropna()
        b = X_new[col].dropna()

        if len(a) < 10 or len(b) < 10:
            continue

        # defaults para stats (así no te rompe en la tabla)
        mean_ref = mean_new = delta_mean = np.nan
        median_ref = median_new = delta_median = np.nan
        top_ref = top_new = None
        top_ref_pct = top_new_pct = delta_top_pct = np.nan


        if col in num_set:
            a_num = pd.to_numeric(a, errors="coerce").dropna()
            b_num = pd.to_numeric(b, errors="coerce").dropna()
            if len(a_num) < 10 or len(b_num) < 10:
                continue

            ks_stat, p_value = ks_2samp(a_num, b_num)
            psi = calculate_psi(a_num, b_num, feature=col, n_bins=int(psi_bins))
            jsd = calculate_jsd(a_num, b_num, feature=col, n_bins=int(jsd_bins))

            # ✅ stats de resumen
            mean_ref = float(a_num.mean())
            mean_new = float(b_num.mean())
            delta_mean = float(mean_new - mean_ref)

            median_ref = float(a_num.median())
            median_new = float(b_num.median())
            delta_median = float(median_new - median_ref)

            feat_type = "numeric"
            stat = float(ks_stat)

        elif col in cat_set:
            a_str = a.astype(str)
            b_str = b.astype(str)

            chi2_stat, p_value, dof = calculate_chi_square(a_str, b_str, feature=col)
            psi = calculate_psi_cat(a_str, b_str, feature=col)
            jsd = calculate_jsd_cat(a_str, b_str, feature=col)

            # ✅ moda + proporción
            ref_vc = a_str.value_counts(normalize=True)
            new_vc = b_str.value_counts(normalize=True)

            if not ref_vc.empty:
                top_ref = str(ref_vc.index[0])
                top_ref_pct = float(ref_vc.iloc[0])

            if not new_vc.empty:
                top_new = str(new_vc.index[0])
                top_new_pct = float(new_vc.iloc[0])

            if pd.notna(top_ref_pct) and pd.notna(top_new_pct):
                delta_top_pct = float(top_new_pct - top_ref_pct)

            feat_type = "categorical"
            stat = float(chi2_stat)

        else:
            # si no está en listas, lo ignoramos (simple)
            continue

        alert_psi = "crit" if psi >= psi_crit else ("warn" if psi >= psi_warn else "ok")
        alert_jsd  = "crit" if jsd >= jsd_crit else ("warn" if jsd >= jsd_warn else "ok")

        drift_flag = bool(p_value < alpha)
        alert_nulls_flag = bool((alert_nulls == "crit"))

        rows.append({
            "feature": col,
            "type": feat_type,
            "stat": stat,
            "p_value": float(p_value),
            "psi": float(psi),
            "jsd": float(jsd),

            "nulls_ref": nulls_ref,
            "nulls_new": nulls_new,
            "delta_nulls": delta_nulls,
            "alert_nulls": alert_nulls,

            "alert_psi": alert_psi,
            "alert_jsd": alert_jsd,
            "drift": drift_flag,
            "nulls": alert_nulls_flag,

            "mean_ref": mean_ref,
            "mean_new": mean_new,
            "delta_mean": delta_mean,
            "median_ref": median_ref,
            "median_new": median_new,
            "delta_median": delta_median,

            "top_ref": top_ref,
            "top_new": top_new,
            "top_ref_pct": top_ref_pct,
            "top_new_pct": top_new_pct,
            "delta_top_pct": delta_top_pct,
        })

    # Evitar KeyError cuando no hay features evaluables
    if not rows:
        return pd.DataFrame(columns=[
            "feature", "type", "stat", "p_value", "psi", "jsd",
            "nulls_ref", "nulls_new", "delta_nulls", "alert_nulls",
            "alert_psi", "alert_jsd", "drift", "nulls",
            "mean_ref", "mean_new", "delta_mean",
            "median_ref", "median_new", "delta_median",
            "top_ref", "top_new", "top_ref_pct", "top_new_pct", "delta_top_pct",
        ])

    return (
        pd.DataFrame(rows)
        .sort_values(["drift", "p_value"], ascending=[False, True])
        .reset_index(drop=True)
    )

#----------------------------------------------
# 3. Cargar datos y preprocesamiento
#----------------------------------------------
df = rename_cols(verbose=False).copy()
X, y = split_features_target(df, target=TARGET)
X_ref, X_new, y_ref, y_new = split_data(
    X,
    y,
    test_size=0.3,
    split_type="temporal",
    random_state=None,
    stratify=False,
    date_col=DATE_COL,
)

#----------------------------------------------
# 4. Visualizaciones
#----------------------------------------------
def render_drift_visuals(df_data_ref, df_data_new, drift_results, title_suffix, psi_bins, jsd_bins):
    """
    Renderiza: Heatmap, Comparación de distribuciones, y Drift temporal (si hay DATE_COL).
    Requiere columnas en drift_results: feature, type, psi, jsd, p_value
    """
    if drift_results.empty:
        st.warning(f"No hay datos suficientes para visualizar {title_suffix}")
        return

    # --- A. Heatmap de Intensidad (Top 10) ---
    st.subheader(f"🌐 Heatmap de Intensidad de Drift (Top 10) ({title_suffix})")

    df_num = drift_results[drift_results["type"] == "numeric"].copy()
    if df_num.empty:
        st.info("No hay variables numéricas para mostrar KS en el heatmap.")
    else:
        # Normalizar a 0–1 para que las métricas sean comparables en color
        ks_norm = df_num["stat"].clip(0, 1)  # KS ya vive en [0,1]
        psi_scaled = (df_num["psi"] / max(psi_crit, 1e-12)).clip(0, 1)  # PSI escalado por umbral crítico
        jsd_norm = df_num["jsd"].clip(0, 1)    # JSD en [0,1]

        hm = pd.DataFrame({
            "PSI (escalado)": psi_scaled.values,
            "KS Statistic": ks_norm.values,
            "JSD": jsd_norm.values
        }, index=df_num["feature"].astype(str).values)

        # Score promedio para ranking + Top 10
        hm["__score__"] = hm.mean(axis=1)
        top10 = hm.sort_values("__score__", ascending=False).drop(columns="__score__").head(10)

        # Ordenar esas Top 10 por: PSI -> KS -> JSD
        top10_sorted = top10.sort_values(
            by=["PSI (escalado)", "KS Statistic", "JSD"],
            ascending=[False, False, False],
            kind="mergesort"
        )

        # Score promedio para ranking + Top 10
        hm["__score__"] = hm.mean(axis=1)
        hm = hm.sort_values("__score__", ascending=False).drop(columns="__score__").head(10)

        fig_hm, ax_hm = plt.subplots(figsize=(10, max(2.5, 0.6 * len(top10_sorted))))
        sns.heatmap(
            top10_sorted,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn_r",   # verde bajo -> rojo alto
            vmin=0, vmax=1,
            linewidths=0.5,
            cbar_kws={"label": "Intensidad de Drift"},
            ax=ax_hm
        )
        ax_hm.set_title("Intensidad de Drift por Métrica (Verde=Bajo, Rojo=Alto)")
        ax_hm.set_xlabel("")
        ax_hm.set_ylabel("")
        st.pyplot(fig_hm)
        plt.close(fig_hm)

        st.caption("Top 10 por score promedio; dentro del Top 10 se ordena por PSI→KS→JSD (todas en escala 0-1).")

    # --- B. Comparación ---
    st.divider()
    st.subheader(f"📊 Comparación de distribuciones ({title_suffix})")

    target_feat = st.selectbox(
        f"Selecciona variable ({title_suffix}):",
        drift_results["feature"].tolist(),
        key=f"select_{title_suffix}"
    )

    feat_type = drift_results.loc[drift_results["feature"] == target_feat, "type"].values[0]

    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        # NUMÉRICAS: distribuciones superpuestas (como tu ejemplo)
        if feat_type == "numeric":
            fig, ax = plt.subplots(figsize=(7, 4))

            sns.histplot(
                df_data_ref[target_feat],
                stat="density",
                kde=True,
                label="Referencia",
                alpha=0.5,
                ax=ax
            )
            sns.histplot(
                df_data_new[target_feat],
                stat="density",
                kde=True,
                label="Actual",
                alpha=0.5,
                ax=ax
            )

            ax.set_title(f"Comparación de Distribuciones: {target_feat}")
            ax.set_xlabel(target_feat)
            ax.set_ylabel("Densidad")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        # CATEGÓRICAS: barras agrupadas (como tu ejemplo)
        else:
            fig, ax = plt.subplots(figsize=(7, 4))

            ref_prop = (
                df_data_ref[target_feat]
                .dropna()
                .astype(str)
                .value_counts(normalize=True)
            )
            new_prop = (
                df_data_new[target_feat]
                .dropna()
                .astype(str)
                .value_counts(normalize=True)
            )

            all_cats = sorted(set(ref_prop.index) | set(new_prop.index))
            ref_prop = ref_prop.reindex(all_cats, fill_value=0)
            new_prop = new_prop.reindex(all_cats, fill_value=0)

            x = np.arange(len(all_cats))
            width = 0.35

            ax.bar(x - width/2, ref_prop.values, width, label="Referencia")
            ax.bar(x + width/2, new_prop.values, width, label="Actual")

            ax.set_xticks(x)
            ax.set_xticklabels(all_cats, rotation=45, ha="right")
            ax.set_ylabel("Proporción")
            ax.set_title(f"Comparación de Categorías: {target_feat}")
            ax.legend()
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

    with col_v2:
        res = drift_results[drift_results["feature"] == target_feat].iloc[0]

        st.metric("PSI", f"{res['psi']:.4f}")
        st.metric("JSD Distance", f"{res['jsd']:.4f}")
        st.metric("P-Value", f"{res['p_value']:.4f}")

        # ✅ NUEVO: resumen por tipo
        if feat_type == "numeric":
            st.divider()
            st.caption("Resumen (numérica)")
            st.metric("Mean ref", f"{res['mean_ref']:.3f}")
            st.metric("Mean new", f"{res['mean_new']:.3f}")
            st.metric("Δ mean", f"{res['delta_mean']:.3f}")
            st.metric("Median ref", f"{res['median_ref']:.3f}")
            st.metric("Median new", f"{res['median_new']:.3f}")
            st.metric("Δ median", f"{res['delta_median']:.3f}")
        else:
            st.divider()
            st.caption("Resumen (categórica)")
            st.write(f"**Top ref:** `{res['top_ref']}` ({res['top_ref_pct']:.1%})" if pd.notna(res["top_ref_pct"]) else f"**Top ref:** `{res['top_ref']}`")
            st.write(f"**Top new:** `{res['top_new']}` ({res['top_new_pct']:.1%})" if pd.notna(res["top_new_pct"]) else f"**Top new:** `{res['top_new']}`")
            if pd.notna(res["delta_top_pct"]):
                st.write(f"**Δ top pct:** {res['delta_top_pct']:.1%}")

    if "alert_psi" in drift_results.columns and "alert_jsd" in drift_results.columns:
        st.caption(f"Alert PSI: **{res['alert_psi']}** | Alert JSD: **{res['alert_jsd']}**")

    # --- C. Análisis Temporal ---
    st.divider()
    st.subheader("📈 Análisis temporal del drift")

    if DATE_COL not in df_data_new.columns:
        st.info(f"No existe {DATE_COL} en los datos analizados, no se puede hacer análisis temporal.")
        return

    freq = st.selectbox("Ventana temporal", ["W", "M", "Y"], index=1)
    metric_to_track = st.selectbox("Métrica a trackear", ["psi", "jsd"], index=0)
    top_k = st.slider("Top K features a mostrar", 3, 20, 10, 1)

    xnew = df_data_new.copy()
    xnew[DATE_COL] = pd.to_datetime(xnew[DATE_COL], errors="coerce")
    xnew = xnew.dropna(subset=[DATE_COL])

    if len(xnew) == 0:
        st.warning("No hay fechas válidas para análisis temporal.")
        return

    top_features = (
        drift_results.sort_values(["drift", "psi", "jsd"], ascending=[False, False, False])
        ["feature"].head(top_k).tolist()
    )

    xnew["window"] = xnew[DATE_COL].dt.to_period(freq).astype(str)

    series_rows = []
    for w, chunk in xnew.groupby("window"):
        for feat in top_features:
            if feat not in chunk.columns or feat not in df_data_ref.columns:
                continue

            a = df_data_ref[feat].dropna()
            b = chunk[feat].dropna()

            if len(a) < 10 or len(b) < 10:
                continue

            ftype = drift_results.loc[drift_results["feature"] == feat, "type"].values[0]

            if ftype == "numeric":
                a_num = pd.to_numeric(a, errors="coerce").dropna()
                b_num = pd.to_numeric(b, errors="coerce").dropna()
                if len(a_num) < 10 or len(b_num) < 10:
                    continue

                psi_val = calculate_psi(a_num, b_num, feature=feat, n_bins=int(psi_bins))
                jsd_val = calculate_jsd(a_num, b_num, feature=feat, n_bins=int(jsd_bins))
            else:
                a_str = a.dropna().astype(str)
                b_str = b.dropna().astype(str)
                if len(a_str) < 10 or len(b_str) < 10:
                    continue

                psi_val = calculate_psi_cat(a_str, b_str, feature=feat)
                jsd_val = calculate_jsd_cat(a_str, b_str, feature=feat)

            series_rows.append({"window": w, "feature": feat, "psi": psi_val, "jsd": jsd_val})

    trend = pd.DataFrame(series_rows)
    if trend.empty:
        st.info("No se pudo calcular drift por ventanas (pocos datos).")
        return

    pivot = (
        trend.pivot_table(index="window", columns="feature", values=metric_to_track, aggfunc="mean")
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], marker="o", label=str(col))

    if metric_to_track == "psi":
        ax.axhline(psi_crit, color="red", linestyle="--", alpha=0.5, label="PSI crit")
        ax.axhline(psi_warn, color="orange", linestyle="--", alpha=0.5, label="PSI warn")

    ax.set_xlabel("Ventana Temporal")
    ax.set_ylabel(metric_to_track.upper())
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.caption(f"Tabla temporal (Métrica: {metric_to_track.upper()})")
    st.dataframe(trend.sort_values(["window", "feature"]), use_container_width=True)

#----------------------------------------------
# 5. Alertas
#----------------------------------------------  
def build_alert_message(drift_df: pd.DataFrame, alpha: float, psi_crit: float, jsd_crit: float):
    if drift_df.empty:
        st.info("Sin datos suficientes para evaluar drift.")
        return

    n = len(drift_df)
    drift_rate = float(drift_df["drift"].mean())

    n_psi_crit = int((drift_df["alert_psi"] == "crit").sum()) if "alert_psi" in drift_df else 0
    n_psi_warn = int((drift_df["alert_psi"] == "warn").sum()) if "alert_psi" in drift_df else 0
    n_jsd_crit  = int((drift_df["alert_jsd"]  == "crit").sum()) if "alert_jsd"  in drift_df else 0
    n_jsd_warn  = int((drift_df["alert_jsd"]  == "warn").sum()) if "alert_jsd"  in drift_df else 0

    # Top 3 por magnitud combinada (PSI y JS) para mencionar por nombre
    tmp = drift_df.copy()
    tmp["score_mag"] = tmp["psi"].fillna(0) / max(psi_crit, 1e-12) + tmp["jsd"].fillna(0) / max(jsd_crit, 1e-12)
    top3 = tmp.sort_values(["score_mag", "psi", "jsd"], ascending=False).head(3)
    top_txt = ", ".join([f"`{r.feature}` (PSI={r.psi:.3f}, JSD={r.jsd:.3f})" for r in top3.itertuples()])

    # Drift estadístico sin magnitud (p<alpha pero PSI/JSD en ok)
    if "alert_psi" in drift_df and "alert_jsd" in drift_df:
        mag_ok = (drift_df["alert_psi"].eq("ok") & drift_df["alert_jsd"].eq("ok"))
        n_stat_only = int((drift_df["drift"] & mag_ok).sum())
    else:
        n_stat_only = 0

    # Mensajes
    if (n_psi_crit + n_jsd_crit) > 0:
        st.error(
            f"🔴 **Drift crítico por magnitud**: PSI crítico en **{n_psi_crit}** vars, JSD crítico en **{n_jsd_crit}** vars "
            f"(drift rate total: **{drift_rate:.1%}**).\n\n"
            f"Más afectadas: {top_txt}\n\n"
            f"**Acción:** revisar cambios en origen/negocio para esas variables y monitorear próximas ventanas; "
            f"si persiste, considerar reentrenar."
        )
    elif (n_psi_warn + n_jsd_warn) >= 3 or drift_rate >= 0.20:
        st.warning(
            f"🟠 **Drift moderado**: {int(drift_df['drift'].sum())}/{n} variables con p-value < {alpha:.3f}. "
            f"Alertas warn (PSI: {n_psi_warn}, JSD: {n_jsd_warn}).\n\n"
            f"Top cambios: {top_txt}\n\n"
            f"**Acción:** monitoreo intensivo y validación de pipeline/segmentación."
        )
    elif n_stat_only > 0:
        st.info(
            f"🟡 **Cambios estadísticos leves**: {n_stat_only} variables con p-value < {alpha:.3f} "
            f"pero sin magnitud relevante (PSI/JSD dentro de umbrales).\n\n"
            f"Top cambios: {top_txt}\n\n"
            f"**Acción:** continuar monitoreo (posible variación normal o efecto de tamaño muestral)."
        )
    else:
        st.success(
            f"🟢 **RAW estable**: drift rate **{drift_rate:.1%}** y sin alertas warn/crit.\n\n"
            f"Variables con mayor variación relativa: {top_txt}"
        )


#----------------------------------------------
# 6. Dashboard tabs
#----------------------------------------------
st.title("Dashboard de monitoreo de Data Drift (RAW)")
st.write("Esta aplicación detecta desviaciones (data drift) en los datos de entrada. No evalúa performance del modelo.")

# Mostrar corte temporal si aplica
if DATE_COL in X_ref.columns and DATE_COL in X_new.columns:
    try:
        cutoff_ref = pd.to_datetime(X_ref[DATE_COL], errors="coerce").max()
        cutoff_new = pd.to_datetime(X_new[DATE_COL], errors="coerce").min()
        st.caption(f"Corte temporal: max(ref)={cutoff_ref} | min(new)={cutoff_new}")
    except Exception:
        pass

tab_raw, tab_pre, tab_resumen, tab_predict = st.tabs(["🔍 Datos Crudos (Raw)","Datos Preprocesados","✅ Resumen Final", "Predicción"])

# --- TAB RAW ---
with tab_raw:
    excluded_cols = [DATE_COL]  # siempre fuera del drift

    drift_raw = detect_drift(
        X_ref, X_new,
        alpha=alpha,  # FIX: usar valor del sidebar
        excluded_cols=excluded_cols,
        psi_bins=psi_bins,
        jsd_bins=jsd_bins,
        psi_warn=psi_warn, 
        psi_crit=psi_crit,
        jsd_warn=jsd_warn, 
        jsd_crit=jsd_crit
    )

    if drift_raw.empty:
        st.warning("No hay suficiente información para calcular drift.")
    else:
        raw_pct = float(drift_raw["drift"].mean())

        # Semáforo simple por magnitud
        n_crit = int((drift_raw["alert_psi"].eq("crit") | drift_raw["alert_jsd"].eq("crit")).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Features con drift (p<alpha)", f"{raw_pct:.1%}")
        c2.metric("Features críticas (PSI/JSD)", n_crit)
        c3.metric("Features analizadas", int(len(drift_raw)))

        # ✅ AVISO DETALLADO 
        build_alert_message(drift_raw, alpha=alpha, psi_crit=psi_crit, jsd_crit=jsd_crit)  # Usar valor del sidebar

        render_drift_visuals(X_ref, X_new, drift_raw, "Raw", psi_bins=psi_bins, jsd_bins=jsd_bins)
        st.dataframe(drift_raw, use_container_width=True)

# --- TAB PRE ---
with tab_pre:

    # 1) Cargar pipeline entrenado 
    @st.cache_resource
    def load_fe_pipeline():
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, "..", "artifacts", "pipelines", "ft_pipeline.joblib")
        return joblib.load(path)

    pipe = load_fe_pipeline()

    # 2) Transformar REF y NEW al espacio del modelo
    X_ref_pre_arr = pipe.transform(X_ref)
    X_new_pre_arr = pipe.transform(X_new)

    # 3) Nombres de features del preprocessor
    try:
        feat_names = pipe.named_steps["pre"].get_feature_names_out()
        feat_names = [str(x) for x in feat_names]
    except Exception:
        feat_names = [f"f_{i}" for i in range(X_ref_pre_arr.shape[1])]

    X_ref_pre = pd.DataFrame(X_ref_pre_arr, columns=feat_names)
    X_new_pre = pd.DataFrame(X_new_pre_arr, columns=feat_names)

    # 4) Drift PRE (misma lógica que RAW)
    excluded_cols = []  # en PRE no está fecha_prestamo

    drift_pre = detect_drift(
        X_ref_pre, X_new_pre,
        alpha=alpha,  # usar valor del sidebar
        excluded_cols=excluded_cols,
        psi_bins=psi_bins,
        jsd_bins=jsd_bins,
        psi_warn=psi_warn,
        psi_crit=psi_crit,
        jsd_warn=jsd_warn,
        jsd_crit=jsd_crit
    )

    if drift_pre.empty:
        st.warning("No hay suficiente información para calcular drift PRE.")
    else:
        pre_pct = float(drift_pre["drift"].mean())

        # Semáforo simple por magnitud
        n_crit = int((drift_pre["alert_psi"].eq("crit") | drift_pre["alert_jsd"].eq("crit")).sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Features con drift (p<alpha)", f"{pre_pct:.1%}")
        c2.metric("Features críticas (PSI/JSD)", n_crit)
        c3.metric("Features analizadas", int(len(drift_pre)))

        # ✅ AVISO DETALLADO
        build_alert_message(
            drift_pre,
            alpha=alpha,  #  usar valor del sidebar
            psi_crit=psi_crit,
            jsd_crit=jsd_crit
        )

        render_drift_visuals(X_ref_pre, X_new_pre, drift_pre, "Pre", psi_bins=psi_bins, jsd_bins=jsd_bins)
        st.dataframe(drift_pre, use_container_width=True)

# --- TAB RESUMEN ---
with tab_resumen:
    st.header("Veredicto del Sistema de data drift (RAW y PRE)")

    if drift_raw.empty and drift_pre.empty:
        st.info("No hay resultados de drift para resumir.")
    else:
        col_raw, col_pre = st.columns(2)

        with col_raw:
            st.subheader("🔍 RAW (datos crudos)")

            if drift_raw.empty:
                st.warning("No hay drift RAW.")

            else: 
                raw_pct = float(drift_raw["drift"].mean())
                n_crit_psi = int((drift_raw["psi"].fillna(0) >= psi_crit).sum())
                n_crit_jsd  = int((drift_raw["jsd"].fillna(0)  >= jsd_crit).sum())

                m1, m2, m3 = st.columns(3)
                m1.metric("Drift rate (p<alpha)", f"{raw_pct:.1%}")
                m2.metric("PSI crítico", n_crit_psi)
                m3.metric("JSD crítico", n_crit_jsd)

                if raw_pct >= 0.30 or (n_crit_psi + n_crit_jsd) >= 3:
                    st.error("### ❌ REVISIÓN / POSIBLE REENTRENAMIENTO: Drift alto detectado en RAW.")
                    st.write("Sugerencias: revisar fuente de datos, cambios de negocio, y variables top con mayor drift.")
                elif raw_pct >= 0.10 or (n_crit_psi + n_crit_jsd) > 0:
                    st.warning("### ⚠️ MONITOREO ACTIVADO: cambios moderados en RAW.")
                    st.write("Sugerencias: seguir monitoreando próximas ventanas y validar variables afectadas.")
                else:
                    st.success("### ✅ OPERACIÓN NORMAL: RAW consistente.")

                st.divider()
                st.subheader("Top variables por magnitud")
                top_num = drift_raw[drift_raw["type"] == "numeric"].sort_values("psi", ascending=False).head(10)
                top_cat = drift_raw[drift_raw["type"] == "categorical"].sort_values("jsd", ascending=False).head(10)
                st.write("**Numéricas (Top PSI)**")
                st.dataframe(top_num, use_container_width=True)
                st.write("**Categóricas (Top JSD)**")
                st.dataframe(top_cat, use_container_width=True)
        
        with col_pre: 
            st.subheader("⚙️ PRE (espacio del modelo)")

            if drift_pre.empty:
                st.warning("No hay drift PRE.")
            else:
                pre_pct = float(drift_pre["drift"].mean())
                n_crit_psi_pre = int((drift_pre["psi"].fillna(0) >= psi_crit).sum())
                n_crit_jsd_pre = int((drift_pre["jsd"].fillna(0) >= jsd_crit).sum())

                m1, m2, m3 = st.columns(3)
                m1.metric("Drift rate", f"{pre_pct:.1%}")
                m2.metric("PSI crítico", n_crit_psi_pre)
                m3.metric("JSD crítico", n_crit_jsd_pre)

                # Mensaje semáforo PRE (más importante que RAW)
                if pre_pct >= 0.30 or (n_crit_psi_pre + n_crit_jsd_pre) >= 3:
                    st.error("❌ Drift alto en PRE (impacto probable en el modelo).")
                    st.caption("Sugerencia: evaluar reentrenamiento.")
                elif pre_pct >= 0.10 or (n_crit_psi_pre + n_crit_jsd_pre) > 0:
                    st.warning("⚠️ Drift moderado en PRE.")
                    st.caption("Validar variables más afectadas y monitorear.")
                else:
                    st.success("✅ PRE consistente.")

                st.divider()
                st.caption("Top variables PRE")
                top_num_pre = (
                    drift_pre[drift_pre["type"] == "numeric"]
                    .sort_values("psi", ascending=False)
                    .head(10)
                )
                top_cat_pre = (
                    drift_pre[drift_pre["type"] == "categorical"]
                    .sort_values("jsd", ascending=False)
                    .head(10)
                )

                st.write("**Numéricas (Top PSI)**")
                st.dataframe(top_num_pre, use_container_width=True, height=250)
                st.write("**Categóricas (Top JSD)**")
                st.dataframe(top_cat_pre, use_container_width=True, height=250)

        # =========================
        # VEREDICTO FINAL (1 sola vez)
        # =========================
        st.divider()
        st.subheader("✅ Veredicto Final")

        # Defaults si alguno está vacío
        raw_pct = float(drift_raw["drift"].mean()) if not drift_raw.empty else 0.0
        pre_pct = float(drift_pre["drift"].mean()) if not drift_pre.empty else 0.0

        ncrit_raw = int(((drift_raw.get("alert_psi", pd.Series()).eq("crit")) |
                         (drift_raw.get("alert_jsd", pd.Series()).eq("crit"))).sum()) if not drift_raw.empty else 0

        ncrit_pre = int(((drift_pre.get("alert_psi", pd.Series()).eq("crit")) |
                         (drift_pre.get("alert_jsd", pd.Series()).eq("crit"))).sum()) if not drift_pre.empty else 0

        # Regla: PRE manda
        if pre_pct >= 0.30 or ncrit_pre >= 3:
            st.error("🔴 Drift crítico en PRE → considerar reentrenamiento.")
        elif pre_pct >= 0.10 or ncrit_pre > 0:
            st.warning("🟠 Drift moderado en PRE → monitorear y revisar variables top.")
        elif raw_pct >= 0.30 or ncrit_raw >= 3:
            st.warning("🟠 Drift alto en RAW, pero PRE estable → revisar cambios de negocio sin urgencia.")
        else:
            st.success("🟢 Operación normal: RAW y PRE consistentes.")
            
with tab_predict:
    import json
    import requests

    st.header("🚀 Model Deploy (Batch JSON / CSV)")
    st.caption("Subí un batch por JSON o CSV. Streamlit lo envía a la API /predict.")

    api_url = st.text_input("URL de la API", value="http://localhost:5040").strip()

    c0, c1 = st.columns([1, 2])
    with c0:
        if st.button("Probar /saludo"):
            try:
                r = requests.get(f"{api_url}/saludo", timeout=10)
                st.write("Status:", r.status_code)
                st.json(r.json())
            except Exception as e:
                st.error(f"No pude conectar con la API: {e}")

    st.divider()

    mode = st.radio("Formato de batch", ["JSON", "CSV"], horizontal=True)

    # ---------
    # JSON MODE
    # ---------
    if mode == "JSON":
        st.subheader("📦 Batch JSON")

        example_payload = {
            "records": [
                {
                    "tipo_credito": "consumo",
                    "fecha_prestamo": "2024-01-15",
                    "capital_prestado": 1000000.0,
                    "plazo_meses": 12,
                    "edad_cliente": 35,
                    "tipo_laboral": "dependiente",
                    "salario_cliente": 2500000.0,
                    "total_otros_prestamos": 0.0,
                    "cuota_pactada": 120000.0,
                    "puntaje": None,
                    "puntaje_datacredito": 650.0,
                    "cant_creditosvigentes": 2,
                    "huella_consulta": 1,
                    "saldo_mora": 0.0,
                    "saldo_total": 500000.0,
                    "saldo_principal": 500000.0,
                    "saldo_mora_codeudor": 0.0,
                    "creditos_sectorFinanciero": 1,
                    "creditos_sectorCooperativo": 0,
                    "creditos_sectorReal": 1,
                    "promedio_ingresos_datacredito": 2400000.0,
                    "tendencia_ingresos": "estable"
                },
                {
                    "tipo_credito": "vehiculo",
                    "fecha_prestamo": "2024-02-10",
                    "capital_prestado": 2500000.0,
                    "plazo_meses": 24,
                    "edad_cliente": 41,
                    "tipo_laboral": "independiente",
                    "salario_cliente": 3200000.0,
                    "total_otros_prestamos": 1000000.0,
                    "cuota_pactada": 180000.0,
                    "puntaje": 0.72,
                    "puntaje_datacredito": 720.0,
                    "cant_creditosvigentes": 3,
                    "huella_consulta": 2,
                    "saldo_mora": 0.0,
                    "saldo_total": 1200000.0,
                    "saldo_principal": 1200000.0,
                    "saldo_mora_codeudor": 0.0,
                    "creditos_sectorFinanciero": 2,
                    "creditos_sectorCooperativo": 0,
                    "creditos_sectorReal": 1,
                    "promedio_ingresos_datacredito": 3100000.0,
                    "tendencia_ingresos": "creciente"
                }
            ]
        }

        payload_str = st.text_area(
            "Pegá el JSON (debe tener la clave `records`)",
            value=json.dumps(example_payload, indent=2),
            height=380
        )

        if st.button("Enviar batch (JSON)"):
            try:
                payload = json.loads(payload_str)
                if "records" not in payload or not isinstance(payload["records"], list):
                    st.error("El JSON debe tener una clave `records` que sea una lista.")
                    st.stop()
            except Exception as e:
                st.error(f"JSON inválido: {e}")
                st.stop()

            try:
                r = requests.post(f"{api_url}/predict", json=payload, timeout=60)
                st.write("Status:", r.status_code)
                if r.status_code >= 400:
                    st.error("Error desde la API")
                try:
                    st.json(r.json())
                except Exception:
                    st.write(r.text)
            except Exception as e:
                st.error(f"No pude llamar /predict: {e}")

    # ---------
    # CSV MODE
    # ---------
    else:
        st.subheader("📄 Batch CSV")

        st.caption("El CSV debe tener columnas con los mismos nombres que tu schema (PredictionInput).")
        uploaded = st.file_uploader("Subí tu CSV", type=["csv"])

        if uploaded is not None:
            try:
                df_csv = pd.read_csv(uploaded)
                st.write("Preview CSV:")
                st.dataframe(df_csv.head(20), use_container_width=True)
            except Exception as e:
                st.error(f"No pude leer el CSV: {e}")
                st.stop()

            # Convertir fecha_prestamo a string ISO si viene como datetime
            if "fecha_prestamo" in df_csv.columns:
                df_csv["fecha_prestamo"] = pd.to_datetime(df_csv["fecha_prestamo"], errors="coerce").dt.date.astype(str)

            # ✅ JSON-safe
            df_csv = df_csv.replace([np.nan, np.inf, -np.inf], None)

            # Armar payload JSON para tu API
            records = df_csv.to_dict(orient="records")
            payload = {"records": records}

            st.write("Payload generado (primeros 2 records):")
            st.code(json.dumps({"records": records[:2]}, indent=2), language="json")

            if st.button("Enviar batch (CSV → JSON)"):
                try:
                    r = requests.post(f"{api_url}/predict", json=payload, timeout=120)
                    st.write("Status:", r.status_code)

                    if r.status_code >= 400:
                        st.error("Error desde la API")
                        try:
                            st.json(r.json())
                        except Exception:
                            st.write(r.text)
                        st.stop()

                    resp = r.json()
                    st.subheader("Respuesta API")
                    st.json(resp)

                    # Unir predicciones al CSV (opcional)
                    if "predictions" in resp and isinstance(resp["predictions"], list):
                        preds_df = pd.DataFrame(resp["predictions"])
                        out_df = pd.concat([df_csv.reset_index(drop=True), preds_df], axis=1)
                        st.subheader("CSV + Predicciones")
                        st.dataframe(out_df, use_container_width=True)

                        # Descargar resultado
                        csv_bytes = out_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Descargar CSV con predicciones",
                            data=csv_bytes,
                            file_name="predicciones.csv",
                            mime="text/csv"
                        )

                except Exception as e:
                    st.error(f"No pude llamar /predict: {e}")

    st.divider()
    st.info("Asegurate de tener corriendo FastAPI en 5040: `uvicorn src.model_deploy:app --reload --host 0.0.0.0 --port 5040`")