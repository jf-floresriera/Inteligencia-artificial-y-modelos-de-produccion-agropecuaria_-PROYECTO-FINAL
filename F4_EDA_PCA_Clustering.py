#!/usr/bin/env python3
"""
FASE 4: EDA + PCA/Clustering en Lotes 1-5 - Cultivo de CAUCHO
Explora si distintas fechas de siembra muestran firmas espectrales diferenciables.
Metodos: PCA, KMeans, Clustering Jerarquico.
Entrada: dataset con indices extraidos para TODOS los lotes (no solo Lote 5).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"]})

DATASET_CSV = os.path.expanduser("~/analisis_gulupa/10_dataset_modelado_FINAL.csv")  # ajustar si Lotes 1-4 estan en otro CSV
OUT_DIR = os.path.expanduser("~/analisis_gulupa/20_fase4_eda_pca_clustering")
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(DATASET_CSV)
predictor_cols = [c for c in df.columns if any(idx in c for idx in ["NDVI_","SAVI_","EVI_","NDRE_","GNDVI_"])]
df_valid = df.dropna(subset=predictor_cols).copy()
print(f"Registros validos: {len(df_valid)}  | Predictores: {predictor_cols}")

X_scaled = StandardScaler().fit_transform(df_valid[predictor_cols].values)

# --- PCA ---
pca = PCA(n_components=min(5, len(predictor_cols)))
pcs = pca.fit_transform(X_scaled)
var_exp = pca.explained_variance_ratio_
print("\nVarianza explicada por componente:", np.round(var_exp*100, 1))

for i in range(min(2, pcs.shape[1])):
    df_valid[f"PC{i+1}"] = pcs[:, i]

# --- KMeans: buscar k optimo con silhouette ---
sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)
k_optimo = max(sil_scores, key=sil_scores.get)
print(f"\nSilhouette por k: {sil_scores}")
print(f"k optimo sugerido: {k_optimo}")

km_final = KMeans(n_clusters=k_optimo, random_state=42, n_init=10)
df_valid["cluster_kmeans"] = km_final.fit_predict(X_scaled)

# --- Clustering Jerarquico ---
Z = linkage(X_scaled, method="ward")
agglo = AgglomerativeClustering(n_clusters=k_optimo, linkage="ward")
df_valid["cluster_jerarquico"] = agglo.fit_predict(X_scaled)

df_valid.to_csv(os.path.join(OUT_DIR, "dataset_con_clusters.csv"), index=False)

# --- Figuras ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sc = axes[0].scatter(df_valid["PC1"], df_valid["PC2"], c=df_valid["cluster_kmeans"], cmap="tab10", s=40, edgecolor="black", linewidth=0.3)
axes[0].set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)"); axes[0].set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
axes[0].set_title(f"KMeans (k={k_optimo}) en espacio PCA", fontsize=10, fontweight="bold")
plt.colorbar(sc, ax=axes[0], label="Cluster")

axes[1].plot(list(sil_scores.keys()), list(sil_scores.values()), marker="o", color="#2ca02c")
axes[1].set_xlabel("Numero de clusters (k)"); axes[1].set_ylabel("Silhouette score")
axes[1].set_title("Seleccion de k optimo", fontsize=10, fontweight="bold")
axes[1].axvline(k_optimo, color="red", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_PCA_KMeans.png"), dpi=300, facecolor="white")
plt.close()

fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(Z, ax=ax, truncate_mode="lastp", p=30, leaf_rotation=90)
ax.set_title("Dendrograma - Clustering Jerarquico (Ward)", fontweight="bold")
ax.set_ylabel("Distancia")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig_dendrograma.png"), dpi=300, facecolor="white")
plt.close()

if "Fecha" in df_valid.columns:
    tabla_cruzada = pd.crosstab(df_valid["cluster_kmeans"], df_valid["Fecha"])
    tabla_cruzada.to_csv(os.path.join(OUT_DIR, "Tabla_cluster_vs_fecha.csv"))
    print("\n=== Cluster vs Fecha de siembra/muestreo ===")
    print(tabla_cruzada.to_string())

print(f"\n=== Fase 4 finalizada. Salida: {OUT_DIR} ===")
