#!/usr/bin/env python3
"""
FASE 7: Extrapolacion del modelo entrenado (Lote 5) a los Lotes 1-4 - CAUCHO
Genera mapas raster de estimacion de LAI/%Cob en zonas sin medicion directa de campo.

Entrada: modelos entrenados (.pkl) de la Fase 5-6, y los rasters de indices
         (09_indices_finca/<fecha>/) recortados para TODA la finca (todos los lotes).
"""
import os
import glob
import re
import numpy as np
import pandas as pd
import rasterio
import joblib
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif"]})

MODELOS_DIR = os.path.expanduser("~/analisis_gulupa/30_fase5_6_modelado")
RASTER_DIR  = os.path.expanduser("~/analisis_gulupa/09_indices_finca")
OUT_DIR     = os.path.expanduser("~/analisis_gulupa/40_fase7_extrapolacion")
os.makedirs(OUT_DIR, exist_ok=True)

FECHAS = ["20160913", "20161222", "20170814"]
INDICES = ["NDVI", "SAVI", "EVI", "NDRE", "GNDVI"]
TARGETS = ["LAI4", "LAI5", "%Cob"]

# --- Cargar todos los rasters de indices y apilarlos como un stack multibanda ---
print("Cargando rasters de indices para toda la finca...")
bandas = {}
perfil_ref = None
for fecha in FECHAS:
    for idx in INDICES:
        path = os.path.join(RASTER_DIR, fecha, f"{idx}_{fecha}.tif")
        if not os.path.exists(path):
            print(f"[AVISO] No encontrado: {path}")
            continue
        with rasterio.open(path) as src:
            bandas[f"{idx}_{fecha}"] = src.read(1).astype("float32")
            if perfil_ref is None:
                perfil_ref = src.profile

predictor_cols = list(bandas.keys())
print(f"Predictores cargados: {len(predictor_cols)} -> {predictor_cols}")

shape_ref = bandas[predictor_cols[0]].shape
stack = np.stack([bandas[c] for c in predictor_cols], axis=-1)  # (H, W, n_predictores)
H, W, n_pred = stack.shape
stack_flat = stack.reshape(-1, n_pred)

mask_valida = np.all(np.isfinite(stack_flat), axis=1)
print(f"Pixeles validos (con todos los indices disponibles): {mask_valida.sum()}/{len(mask_valida)}")

for target in TARGETS:
    nombre_archivo = target.replace("%", "pct")
    modelo_path = os.path.join(MODELOS_DIR, f"modelo_XGBoost_{nombre_archivo}.pkl")
    if not os.path.exists(modelo_path):
        print(f"[SKIP] No se encontro modelo para {target} en {modelo_path}")
        continue

    modelo = joblib.load(modelo_path)
    print(f"\nExtrapolando {target}...")

    pred_flat = np.full(len(stack_flat), np.nan, dtype="float32")
    pred_flat[mask_valida] = modelo.predict(stack_flat[mask_valida])
    pred_map = pred_flat.reshape(H, W)

    out_tif = os.path.join(OUT_DIR, f"mapa_estimado_{nombre_archivo}.tif")
    perfil_ref.update(dtype="float32", count=1, nodata=np.nan)
    with rasterio.open(out_tif, "w", **perfil_ref) as dst:
        dst.write(pred_map, 1)
    print(f"  [GUARDADO] {out_tif}")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.ma.masked_invalid(pred_map), cmap="RdYlGn")
    ax.set_title(f"Estimacion de {target} — toda la finca (extrapolado desde Lote 5)", fontsize=11, fontweight="bold")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.04, label=target)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"Fig_mapa_{nombre_archivo}.png"), dpi=300, facecolor="white")
    plt.close()

print(f"\n=== Fase 7 finalizada. Mapas guardados en: {OUT_DIR} ===")
print("NOTA: revisar visualmente los mapas -> valores fuera de lo agronomicamente esperado")
print("en Lotes 1-4 pueden indicar extrapolacion fuera del rango de entrenamiento (Lote 5).")
