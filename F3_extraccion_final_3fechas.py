#!/usr/bin/env python3
"""
FASE 3 (extraccion) + FASE 3c (deteccion de atipicos) - LISTOS PARA CORRER
Usa los indices ya calculados en: ~/analisis_gulupa/09_indices_finca/<fecha>/
para las 3 fechas: 20160913, 20161222, 20170814
"""
import os, glob, re
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from shapely.geometry import Point

CSV_BASE   = os.path.expanduser("~/analisis_gulupa/base_datos_copas_HLD_con_coordenadas.csv")
RASTER_DIR = os.path.expanduser("~/analisis_gulupa/09_indices_finca")   # <- YA ACTUALIZADO
OUT_CSV    = os.path.expanduser("~/analisis_gulupa/10_dataset_modelado_FINAL.csv")
CRS_PUNTOS = "EPSG:4326"

df = pd.read_csv(CSV_BASE)
geometry = [Point(xy) for xy in zip(df["Longitud"], df["Latitud"])]
gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=CRS_PUNTOS)

raster_files = sorted(glob.glob(os.path.join(RASTER_DIR, "**", "*.tif"), recursive=True))
print(f"Rasters encontrados: {len(raster_files)}")

patron = re.compile(r"(NDVI|SAVI|EVI|NDRE|GNDVI)_(\d{8})", re.IGNORECASE)
resumen = []
for path in raster_files:
    nombre = os.path.basename(path)
    m = patron.search(nombre)
    if not m:
        continue
    col_name = f"{m.group(1).upper()}_{m.group(2)}"
    with rasterio.open(path) as src:
        gdf_r = gdf.to_crs(src.crs) if gdf.crs != src.crs else gdf
        coords = [(pt.x, pt.y) for pt in gdf_r.geometry]
        valores = [v[0] for v in src.sample(coords)]
        nodata = src.nodata
        valores = [np.nan if (nodata is not None and v == nodata) or not np.isfinite(v) else v for v in valores]
    df[col_name] = valores
    n_ok = int(np.sum(~pd.isna(valores)))
    resumen.append({"Columna": col_name, "Validos": n_ok, "Total": len(df), "%valido": round(100*n_ok/len(df),1)})
    print(f"[{'OK' if n_ok>0 else 'VACIO'}] {col_name}: {n_ok}/{len(df)} ({round(100*n_ok/len(df),1)}%)")

df.to_csv(OUT_CSV, index=False)
pd.DataFrame(resumen).to_csv(os.path.expanduser("~/analisis_gulupa/resumen_extraccion_FINAL.csv"), index=False)
print(f"\nCSV final guardado en: {OUT_CSV}")
print("Listo para F3c_deteccion_valores_atipicos_v2.py y luego Fase 4.")
