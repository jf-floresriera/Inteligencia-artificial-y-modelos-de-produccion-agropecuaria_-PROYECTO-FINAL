#!/usr/bin/env python3
"""
Analisis de calidad de las 3 imagenes Sentinel-2 BOA recortadas
al area de la finca (6 lotes de gulupa).

Evalua:
1. Cobertura de nubes (usando banda SCL - Scene Classification)
2. Estadisticas de reflectancia por banda (min, max, media, std)
3. Presencia de valores invalidos/saturados
4. Consistencia geometrica (dimensiones, resolucion, CRS)
5. NDVI resumen como control de sanidad de vegetacion

Requisitos:
    pip install rasterio numpy pandas
"""

import os
import glob
import numpy as np
import rasterio
import pandas as pd

BASE_DIR = os.path.expanduser("~/analisis_gulupa/03_BOA_recortado_finca")
OUTPUT_CSV = os.path.expanduser("~/analisis_gulupa/05_reporte_calidad.csv")

# Clases de la banda SCL (Scene Classification Layer) de Sen2Cor
SCL_CLASES = {
    0: "No_data", 1: "Saturado_defectuoso", 2: "Sombra_oscura",
    3: "Sombra_nube", 4: "Vegetacion", 5: "Suelo_desnudo",
    6: "Agua", 7: "Nube_baja_prob", 8: "Nube_media_prob",
    9: "Nube_alta_prob", 10: "Cirros_delgados", 11: "Nieve_hielo"
}

productos = sorted(glob.glob(os.path.join(BASE_DIR, "*MSIL2A*")))
resultados = []

for prod_dir in productos:
    nombre = os.path.basename(prod_dir)
    fecha = nombre.split("_")[2][:8]
    print(f"\n=== Analizando {nombre} (fecha {fecha}) ===")

    fila = {"producto": nombre, "fecha": fecha}

    # --- 1. Cobertura de nubes via SCL (20m) ---
    scl_files = glob.glob(os.path.join(prod_dir, "R20m_*SCL*.tif"))
    if scl_files:
        with rasterio.open(scl_files[0]) as src:
            scl = src.read(1)
            total_px = scl.size
            nubes_px = np.isin(scl, [7,8,9,10]).sum()
            validos_px = np.isin(scl, [4,5,6,11]).sum()
            nodata_px = (scl == 0).sum()

            pct_nubes = 100 * nubes_px / total_px
            pct_validos = 100 * validos_px / total_px
            pct_nodata = 100 * nodata_px / total_px

            fila["pct_nubes"] = round(pct_nubes, 2)
            fila["pct_pixeles_validos"] = round(pct_validos, 2)
            fila["pct_nodata"] = round(pct_nodata, 2)
            fila["total_pixeles_20m"] = total_px

            print(f"  Nubes: {pct_nubes:.2f}% | Validos: {pct_validos:.2f}% | NoData: {pct_nodata:.2f}%")
    else:
        print("  No se encontro banda SCL")
        fila["pct_nubes"] = None

    # --- 2. Estadisticas por banda clave (10m: B02,B03,B04,B08) ---
    bandas_clave = ["B02", "B03", "B04", "B08"]
    for banda in bandas_clave:
        b_files = glob.glob(os.path.join(prod_dir, f"R10m_*_{banda}*.tif"))
        if not b_files:
            continue
        with rasterio.open(b_files[0]) as src:
            arr = src.read(1).astype(float)
            arr_valid = arr[arr > 0]  # excluir nodata=0
            if arr_valid.size > 0:
                fila[f"{banda}_media"] = round(np.mean(arr_valid), 1)
                fila[f"{banda}_std"] = round(np.std(arr_valid), 1)
                fila[f"{banda}_min"] = int(np.min(arr_valid))
                fila[f"{banda}_max"] = int(np.max(arr_valid))
                pct_saturado = 100 * (arr_valid >= 10000).sum() / arr_valid.size
                fila[f"{banda}_pct_saturado"] = round(pct_saturado, 2)

    # --- 3. NDVI resumen (usando B08 y B04) ---
    b04_files = glob.glob(os.path.join(prod_dir, "R10m_*_B04*.tif"))
    b08_files = glob.glob(os.path.join(prod_dir, "R10m_*_B08*.tif"))
    if b04_files and b08_files:
        with rasterio.open(b04_files[0]) as src_r, rasterio.open(b08_files[0]) as src_n:
            red = src_r.read(1).astype(float)
            nir = src_n.read(1).astype(float)
            mask_valid = (red > 0) & (nir > 0)
            ndvi = np.where(mask_valid, (nir - red) / (nir + red + 1e-9), np.nan)
            ndvi_valid = ndvi[~np.isnan(ndvi)]
            if ndvi_valid.size > 0:
                fila["NDVI_medio"] = round(np.nanmean(ndvi_valid), 3)
                fila["NDVI_std"] = round(np.nanstd(ndvi_valid), 3)
                fila["NDVI_min"] = round(np.nanmin(ndvi_valid), 3)
                fila["NDVI_max"] = round(np.nanmax(ndvi_valid), 3)
                print(f"  NDVI medio: {fila['NDVI_medio']}")

    # --- 4. Consistencia geometrica ---
    if b04_files:
        with rasterio.open(b04_files[0]) as src:
            fila["ancho_px"] = src.width
            fila["alto_px"] = src.height
            fila["resolucion_m"] = src.res[0]
            fila["CRS"] = str(src.crs)

    resultados.append(fila)

df = pd.DataFrame(resultados)
df = df.sort_values("fecha")
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n=== Reporte guardado en: {OUTPUT_CSV} ===")
print(df.to_string(index=False))
