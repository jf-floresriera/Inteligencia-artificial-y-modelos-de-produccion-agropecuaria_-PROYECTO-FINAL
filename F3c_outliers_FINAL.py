#!/usr/bin/env python3
"""
FASE 3c: DETECCION DE VALORES ATIPICOS - Cultivo de CAUCHO (Hevea brasiliensis)
Usa el dataset final ya con indices extraidos en las 3 fechas: 20160913, 20161222, 20170814
Archivo de entrada: 10_dataset_modelado_FINAL.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"]})

DATASET_CSV = os.path.expanduser("~/analisis_gulupa/10_dataset_modelado_FINAL.csv")
OUT_DIR = os.path.expanduser("~/analisis_gulupa/16_deteccion_atipicos")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_CSV)
print(f"Registros cargados: {len(df)}")

vars_campo = [c for c in ["LAI4", "LAI5", "%Cob"] if c in df.columns]
resumen_campo = []
for var in vars_campo:
    serie = df[var].dropna()
    z = (serie - serie.mean()) / serie.std()
    outliers_z = serie[np.abs(z) > 3]
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf, lim_sup = q1 - 1.5*iqr, q3 + 1.5*iqr
    outliers_iqr = serie[(serie < lim_inf) | (serie > lim_sup)]
    df.loc[outliers_z.index, f"{var}_outlier_zscore"] = True
    df.loc[outliers_iqr.index, f"{var}_outlier_iqr"] = True
    resumen_campo.append({"Variable": var, "n": len(serie), "Outliers_Zscore": len(outliers_z),
                           "Outliers_IQR": len(outliers_iqr), "Lim_inf": round(lim_inf,3), "Lim_sup": round(lim_sup,3)})
pd.DataFrame(resumen_campo).to_csv(os.path.join(OUT_DIR, "Tabla_outliers_campo.csv"), index=False)
print("\n=== Outliers univariados (campo) ===")
print(pd.DataFrame(resumen_campo).to_string(index=False))

predictor_cols = [c for c in df.columns if any(idx in c for idx in ["NDVI_", "SAVI_", "EVI_", "NDRE_", "GNDVI_"])]
df_esp = df.dropna(subset=predictor_cols).copy()
print(f"\nRegistros validos multivariados: {len(df_esp)}/{len(df)}  | Predictores: {len(predictor_cols)}")

X_scaled = StandardScaler().fit_transform(df_esp[predictor_cols].values)

iso = IsolationForest(contamination=0.1, random_state=42)
df_esp["outlier_isoforest"] = iso.fit_predict(X_scaled) == -1
lof = LocalOutlierFactor(n_neighbors=min(10, len(X_scaled)-1), contamination=0.1)
df_esp["outlier_lof"] = lof.fit_predict(X_scaled) == -1
db = DBSCAN(eps=1.5, min_samples=3)
df_esp["outlier_dbscan"] = db.fit_predict(X_scaled) == -1
ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
df_esp["outlier_ocsvm"] = ocsvm.fit_predict(X_scaled) == -1

cols_out = ["outlier_isoforest", "outlier_lof", "outlier_dbscan", "outlier_ocsvm"]
df_esp["n_metodos"] = df_esp[cols_out].sum(axis=1)
df_esp["outlier_consenso"] = df_esp["n_metodos"] >= 2

resumen = pd.DataFrame({"Metodo": ["IsoForest","LOF","DBSCAN","OCSVM","Consenso(>=2)"],
    "N_atipicos": [df_esp[c].sum() for c in cols_out] + [df_esp["outlier_consenso"].sum()]})
resumen.to_csv(os.path.join(OUT_DIR, "Tabla_outliers_metodos.csv"), index=False)
print("\n=== Comparacion de metodos ===")
print(resumen.to_string(index=False))

df_limpio = df_esp[~df_esp["outlier_consenso"]].copy()
df_esp.to_csv(os.path.join(OUT_DIR, "dataset_con_outliers_marcados.csv"), index=False)
df_limpio.to_csv(os.path.expanduser("~/analisis_gulupa/11_dataset_LIMPIO_sin_atipicos.csv"), index=False)
print(f"\nRegistros limpios (sin atipicos consenso): {len(df_limpio)}/{len(df_esp)}")
print("Guardado: 11_dataset_LIMPIO_sin_atipicos.csv  <-- usar este para Fase 5-6")

pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
df_esp["PC1"], df_esp["PC2"] = pcs[:,0], pcs[:,1]
fig, ax = plt.subplots(figsize=(7,6))
norm = df_esp[~df_esp["outlier_consenso"]]; atip = df_esp[df_esp["outlier_consenso"]]
ax.scatter(norm["PC1"], norm["PC2"], color="#2ca02c", s=50, edgecolor="black", alpha=0.8, label="Normal")
ax.scatter(atip["PC1"], atip["PC2"], color="red", s=80, edgecolor="black", marker="X", label="Atipico (consenso)")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_title("Atipicos en firmas espectrales - Lote 5 (Caucho)", fontweight="bold")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_outliers_PCA.png"), dpi=300, facecolor="white")
plt.close()
print(f"\nFinalizado. Salida: {OUT_DIR}")
