#!/usr/bin/env python3
"""
Descarga Sentinel-2 L2A (BOA) para 2017-08-14 directamente desde
Google Earth Engine (COPERNICUS/S2_SR_HARMONIZED), recortada al AOI.

Requisitos:
    pip install earthengine-api
    earthengine authenticate --project wide-origin-466923-d8   (ya hecho)

Nota: se exporta a Google Drive porque la API de descarga directa
de GEE tiene limite de tamano para imagenes grandes; si prefieres
evitar Drive, usa geemap.ee_export_image() con region pequena.
"""

import ee
import os

ee.Initialize(project='wide-origin-466923-d8')

AOI_BBOX = [-73.3272, 3.7958, -73.3087, 3.8156]
aoi = ee.Geometry.Rectangle(AOI_BBOX)

fecha = "2017-08-14"
fecha_ee = ee.Date(fecha)
fecha_fin = fecha_ee.advance(1, "day")

coleccion = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
             .filterDate(fecha_ee, fecha_fin)
             .filterBounds(aoi))

n_img = coleccion.size().getInfo()
print(f"Imagenes L2A/BOA encontradas para {fecha}: {n_img}")

if n_img > 0:
    imagen = coleccion.first()
    bandas = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12","SCL"]
    imagen_sel = imagen.select(bandas)

    props = imagen.getInfo()["properties"]
    print("Nubosidad (CLOUDY_PIXEL_PERCENTAGE):", props.get("CLOUDY_PIXEL_PERCENTAGE"))
    print("ID del producto:", props.get("PRODUCT_ID"))

    task = ee.batch.Export.image.toDrive(
        image=imagen_sel,
        description="S2_BOA_20170814",
        folder="S2_BOA_gulupa",
        region=aoi,
        scale=10,
        crs="EPSG:4326",
        maxPixels=1e9
    )
    task.start()
    print("Tarea de exportacion iniciada -> revisa https://code.earthengine.google.com/tasks")
    print("Se guardara en tu Google Drive, carpeta S2_BOA_gulupa")
else:
    print("No hay imagen L2A oficial para esta fecha/AOI. Habria que usar Sen2Cor localmente.")
