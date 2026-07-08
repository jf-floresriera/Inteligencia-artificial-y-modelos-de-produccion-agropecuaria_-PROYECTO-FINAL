#!/usr/bin/env python3
"""
FASE 2c (v2): CALCULO DE INDICES DE VEGETACION - CORREGIDO
Estructura real descubierta (archivos planos, no subcarpetas):

  R10m_T18NXK_<fecha>T<hora>_B02_10m.tif
  R10m_T18NXK_<fecha>T<hora>_B03_10m.tif
  R10m_T18NXK_<fecha>T<hora>_B04_10m.tif
  R10m_T18NXK_<fecha>T<hora>_B08_10m.tif
  R20m_T18NXK_<fecha>T<hora>_B05_20m.tif   <-- OJO: B05 (red edge) solo viene a 20m

Por eso B05 se debe REMUESTREAR (resample) a la grilla de 10m antes de calcular NDRE.
NDVI, GNDVI, EVI, SAVI no necesitan B05, se calculan directo a 10m.
"""
import os
import glob
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

BASE_DIR = os.path.expanduser("~/analisis_gulupa/03_BOA_recortado_finca")
OUT_BASE = os.path.expanduser("~/analisis_gulupa/09_indices_finca")

CARPETAS_FECHAS = {
    "20160913": "S2A_MSIL2A_20160913T151702_N9999_R125_T18NXK_20260707T201721",
    "20161222": "S2A_MSIL2A_20161222T151702_N9999_R125_T18NXK_20260707T204039",
    "20170814": "S2B_MSIL2A_20170814T152119_N0500_R125_T18NXK_20230805T065021",
}

def buscar(carpeta, patron):
    encontrados = glob.glob(os.path.join(carpeta, patron))
    return sorted(encontrados)[0] if encontrados else None

def leer(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
    if np.nanmax(arr) > 10:
        arr = arr / 10000.0
    return arr, profile

def resamplear_a_referencia(path_origen, profile_ref):
    with rasterio.open(path_origen) as src:
        destino = np.zeros((profile_ref["height"], profile_ref["width"]), dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=destino,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=profile_ref["transform"], dst_crs=profile_ref["crs"],
            resampling=Resampling.bilinear,
        )
    if np.nanmax(destino) > 10:
        destino = destino / 10000.0
    return destino

def calcular_indices(fecha, carpeta_fecha):
    print(f"\n===== Procesando fecha {fecha} =====")
    carpeta = os.path.join(BASE_DIR, carpeta_fecha)
    if not os.path.isdir(carpeta):
        print(f"[ERROR] No existe la carpeta: {carpeta}")
        return

    p_b02 = buscar(carpeta, "*_B02_10m.tif")
    p_b03 = buscar(carpeta, "*_B03_10m.tif")
    p_b04 = buscar(carpeta, "*_B04_10m.tif")
    p_b08 = buscar(carpeta, "*_B08_10m.tif")
    p_b05 = buscar(carpeta, "*_B05_20m.tif")

    for nombre, ruta in [("B02_10m", p_b02), ("B03_10m", p_b03), ("B04_10m", p_b04),
                         ("B08_10m", p_b08), ("B05_20m", p_b05)]:
        print(f"  {nombre}: {'OK -> ' + os.path.basename(ruta) if ruta else 'NO ENCONTRADA'}")

    if not all([p_b02, p_b03, p_b04, p_b08]):
        print(f"[ERROR] Faltan bandas basicas de 10m para {fecha}. Se omite esta fecha.")
        return

    B02, profile = leer(p_b02)
    B03, _ = leer(p_b03)
    B04, _ = leer(p_b04)
    B08, _ = leer(p_b08)

    eps = 1e-6
    ndvi  = (B08 - B04) / (B08 + B04 + eps)
    gndvi = (B08 - B03) / (B08 + B03 + eps)
    evi   = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1 + eps)
    savi  = (B08 - B04) / (B08 + B04 + 0.428 + eps) * 1.428

    indices = {"NDVI": ndvi, "GNDVI": gndvi, "EVI": evi, "SAVI": savi}

    if p_b05:
        B05 = resamplear_a_referencia(p_b05, profile)
        ndre = (B08 - B05) / (B08 + B05 + eps)
        indices["NDRE"] = ndre
        print("  [OK] B05 remuestreada de 20m -> 10m para calcular NDRE")
    else:
        print("  [AVISO] No se encontro B05_20m -> NDRE no se calculara para esta fecha")

    out_dir = os.path.join(OUT_BASE, fecha)
    os.makedirs(out_dir, exist_ok=True)
    profile.update(dtype="float32", count=1, nodata=np.nan)

    for nombre, arr in indices.items():
        arr = np.clip(arr, -1, 2).astype("float32")
        out_path = os.path.join(out_dir, f"{nombre}_{fecha}.tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(arr, 1)
        print(f"  [GUARDADO] {out_path}  (min={np.nanmin(arr):.3f}, max={np.nanmax(arr):.3f})")

for fecha, carpeta in CARPETAS_FECHAS.items():
    calcular_indices(fecha, carpeta)

print(f"\n=== Calculo de indices finalizado. Salida en: {OUT_BASE} ===")
