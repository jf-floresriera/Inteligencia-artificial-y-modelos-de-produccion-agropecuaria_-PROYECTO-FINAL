#!/usr/bin/env python3
"""
FASE 3: Extraccion de valores en puntos de campo (Lote 5)
Une los indices espectrales (NDVI, SAVI, EVI, NDRE, GNDVI) por cada
fecha, con las coordenadas exactas de los puntos donde se midio
LAI4, LAI5 y %Cob (base de datos GLA).

Requisitos: rasterio, pandas, geopandas
"""
import os, glob
import numpy as np
import pandas as pd
import rasterio

CSV_CAMPO = os.path.expanduser("~/analisis_gulupa/base_datos_copas_HLD_con_coordenadas.csv")
INDICES_DIR = os.path.expanduser("~/analisis_gulupa/09_indices_lote5")
OUT_CSV = os.path.expanduser("~/analisis_gulupa/10_dataset_modelado.csv")

df_campo = pd.read_csv(CSV_CAMPO)
print(f"Puntos de campo cargados: {len(df_campo)}")
print("Columnas disponibles:", list(df_campo.columns))

# Ajusta estos nombres segun tus columnas reales de lat/lon
COL_LON = "longitud" if "longitud" in df_campo.columns else "lon"
COL_LAT = "latitud" if "latitud" in df_campo.columns else "lat"

INDICES = ["NDVI", "SAVI", "EVI", "NDRE", "GNDVI"]
fechas_dirs = sorted(glob.glob(os.path.join(INDICES_DIR, "*")))

resultados = df_campo.copy()

for fecha_dir in fechas_dirs:
    fecha = os.path.basename(fecha_dir)
    print(f"\\n=== Extrayendo fecha {fecha} ===")

    for indice in INDICES:
        tif_path = os.path.join(fecha_dir, f"{indice}.tif")
        if not os.path.exists(tif_path):
            continue

        with rasterio.open(tif_path) as src:
            valores = []
            for _, row in df_campo.iterrows():
                lon, lat = row[COL_LON], row[COL_LAT]
                try:
                    row_px, col_px = src.index(lon, lat)
                    val = src.read(1)[row_px, col_px]
                    valores.append(val if val != 0 else np.nan)
                except Exception:
                    valores.append(np.nan)

        col_name = f"{indice}_{fecha}"
        resultados[col_name] = valores
        print(f"  {col_name}: {np.nanmean(valores):.3f} (media)")

resultados.to_csv(OUT_CSV, index=False)
print(f"\\n=== Dataset de modelado guardado en: {OUT_CSV} ===")
print(resultados.head())
