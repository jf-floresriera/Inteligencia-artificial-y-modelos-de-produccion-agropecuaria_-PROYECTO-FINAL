#!/usr/bin/env python3
"""
FASE 1: Recorte jerarquico
  (a) Lote 6 (finca completa) -> contexto visual general
  (b) Lote 5 (ground truth)   -> modelado cuantitativo

Requisitos: rasterio, geopandas, shapely
"""
import os, glob
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

INPUT_DIR = os.path.expanduser("~/analisis_gulupa/02_L2A_BOA_completo")
GEOJSON_LOTES = os.path.expanduser("~/analisis_gulupa/lotes_gulupa.geojson")
OUT_FINCA = os.path.expanduser("~/analisis_gulupa/07_recorte_lote6_finca")
OUT_LOTE5 = os.path.expanduser("~/analisis_gulupa/08_recorte_lote5_campo")

os.makedirs(OUT_FINCA, exist_ok=True)
os.makedirs(OUT_LOTE5, exist_ok=True)

lotes = gpd.read_file(GEOJSON_LOTES)
lote6 = lotes[lotes["lote_id"] == 6].geometry.iloc[0]   # finca completa
lote5 = lotes[lotes["lote_id"] == 5].geometry.iloc[0]   # sublote con muestreo de campo

RESOLUCIONES = ["R10m", "R20m"]

def recortar(safe_dir, geom_wgs84, out_base, buffer_deg=0.0002):
    nombre_producto = os.path.basename(safe_dir).replace(".SAFE", "")
    granule_dirs = glob.glob(os.path.join(safe_dir, "GRANULE", "*"))
    if not granule_dirs:
        return
    granule_dir = granule_dirs[0]
    out_dir_producto = os.path.join(out_base, nombre_producto)
    os.makedirs(out_dir_producto, exist_ok=True)

    geom_buf = geom_wgs84.buffer(buffer_deg)

    for res in RESOLUCIONES:
        img_dir = os.path.join(granule_dir, "IMG_DATA", res)
        if not os.path.isdir(img_dir):
            continue
        for jp2 in glob.glob(os.path.join(img_dir, "*.jp2")):
            banda_nombre = os.path.basename(jp2).replace(".jp2", ".tif")
            out_path = os.path.join(out_dir_producto, f"{res}_{banda_nombre}")
            try:
                with rasterio.open(jp2) as src:
                    geom_reproj = gpd.GeoSeries([geom_buf], crs="EPSG:4326").to_crs(src.crs)
                    geom_json = [mapping(geom_reproj.iloc[0])]
                    out_image, out_transform = mask(src, geom_json, crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({"height": out_image.shape[1], "width": out_image.shape[2],
                                      "transform": out_transform, "driver": "GTiff"})
                    with rasterio.open(out_path, "w", **out_meta) as dst:
                        dst.write(out_image)
            except Exception as e:
                print(f"  ERROR {banda_nombre}: {e}")

    print(f"  -> {out_dir_producto}")

safes = glob.glob(os.path.join(INPUT_DIR, "*MSIL2A*"))
print(f"Productos L2A encontrados: {len(safes)}")

for safe in safes:
    print(f"\\n=== {os.path.basename(safe)} ===")
    print(" Recortando a Lote 6 (finca completa)...")
    recortar(safe, lote6, OUT_FINCA)
    print(" Recortando a Lote 5 (ground truth)...")
    recortar(safe, lote5, OUT_LOTE5)

print(f"\\n=== Recorte jerarquico finalizado ===")
print(f"Finca completa: {OUT_FINCA}")
print(f"Lote 5 (campo): {OUT_LOTE5}")
