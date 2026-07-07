#!/usr/bin/env python3
"""
Recorta las 3 imagenes Sentinel-2 BOA (2016-09-13, 2016-12-22, 2017-08-14)
al area exacta de los 6 lotes de gulupa, usando el poligono real
(no un rectangulo aproximado).

Requisitos:
    pip install rasterio geopandas shapely
"""

import os
import glob
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping

INPUT_DIR = os.path.expanduser("~/analisis_gulupa/02_L2A_BOA_completo")
OUTPUT_DIR = os.path.expanduser("~/analisis_gulupa/03_BOA_recortado_finca")
GEOJSON_LOTES = os.path.expanduser("~/analisis_gulupa/lotes_gulupa.geojson")

os.makedirs(OUTPUT_DIR, exist_ok=True)

lotes = gpd.read_file(GEOJSON_LOTES)
print(f"Lotes cargados: {len(lotes)}")

union_lotes = lotes.union_all()
buffer_deg = 0.0005  # ~50m de buffer
aoi_geom = union_lotes.buffer(buffer_deg)
print("AOI (union de 6 lotes + buffer) generado")

safes = glob.glob(os.path.join(INPUT_DIR, "*MSIL2A*.SAFE"))
print(f"\nCarpetas L2A encontradas: {len(safes)}")
for s in safes:
    print(f"  - {os.path.basename(s)}")

RESOLUCIONES = ["R10m", "R20m"]

for safe in safes:
    nombre_producto = os.path.basename(safe).replace(".SAFE", "")
    print(f"\n=== Procesando {nombre_producto} ===")

    granule_dirs = glob.glob(os.path.join(safe, "GRANULE", "*"))
    if not granule_dirs:
        print("  No se encontro carpeta GRANULE")
        continue
    granule_dir = granule_dirs[0]

    out_dir_producto = os.path.join(OUTPUT_DIR, nombre_producto)
    os.makedirs(out_dir_producto, exist_ok=True)

    for res in RESOLUCIONES:
        img_dir = os.path.join(granule_dir, "IMG_DATA", res)
        if not os.path.isdir(img_dir):
            continue

        jp2_files = glob.glob(os.path.join(img_dir, "*.jp2"))
        print(f"  {res}: {len(jp2_files)} bandas encontradas")

        for jp2 in jp2_files:
            banda_nombre = os.path.basename(jp2).replace(".jp2", ".tif")
            out_path = os.path.join(out_dir_producto, f"{res}_{banda_nombre}")

            try:
                with rasterio.open(jp2) as src:
                    aoi_reproj = gpd.GeoSeries([aoi_geom], crs="EPSG:4326").to_crs(src.crs)
                    geom_reproj = [mapping(aoi_reproj.iloc[0])]

                    out_image, out_transform = mask(src, geom_reproj, crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "height": out_image.shape[1],
                        "width": out_image.shape[2],
                        "transform": out_transform,
                        "driver": "GTiff"
                    })

                    with rasterio.open(out_path, "w", **out_meta) as dst:
                        dst.write(out_image)

            except Exception as e:
                print(f"    ERROR en {banda_nombre}: {e}")

    print(f"  Guardado en: {out_dir_producto}")

print(f"\n=== Recorte finalizado ===")
print(f"Revisa: {OUTPUT_DIR}")
