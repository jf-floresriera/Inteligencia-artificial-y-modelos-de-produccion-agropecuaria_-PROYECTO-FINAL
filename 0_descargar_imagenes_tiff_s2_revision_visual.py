# -*- coding: utf-8 -*-
"""
DESCARGA DE CANDIDATAS SENTINEL-2 PARA REVISION VISUAL
Proyecto caucho / clones / LAI / CAP

Este script hace SOLO la etapa de inspeccion visual:
1) Busca todas las imagenes Sentinel-2 disponibles en los meses de interes.
2) Descarga un GeoTIFF multibanda FULL, sin enmascarar nubes.
3) Incluye bandas originales, indices y mascaras.
4) Separa cada banda en GeoTIFF individual.
5) Genera paneles PNG RGB, falso color, NDVI, MSAVI, NDMI y CLEAR_MASK.
6) Genera un Excel para marcar manualmente la mejor imagen por mes.

Requisitos:
python -m pip install earthengine-api geemap pandas numpy openpyxl rasterio matplotlib
"""

import math
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import ee
import geemap
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt


# =============================================================================
# 1. PARAMETROS PRINCIPALES
# =============================================================================

GEE_PROJECT = "wide-origin-466923-d8"

ROOT_DIR = Path(r"D:\Enrique_Tapiero")
KML_PATH = ROOT_DIR / "CLONE_LAI_CAP.kml"

OUT_DIR = ROOT_DIR / "S2_CANDIDATAS_REVISION_VISUAL_V3"
OUT_TIFF = OUT_DIR / "01_TIFF_FULL_WITH_CLOUDS"
OUT_BANDS = OUT_DIR / "02_BANDAS_INDIVIDUALES"
OUT_FIG = OUT_DIR / "03_PANELES_VISUALES"
OUT_CONTACT = OUT_DIR / "04_CONTACT_SHEETS"
OUT_CSV = OUT_DIR / "05_CSV"
OUT_EXCEL = OUT_DIR / "06_EXCEL"

for folder in [OUT_DIR, OUT_TIFF, OUT_BANDS, OUT_FIG, OUT_CONTACT, OUT_CSV, OUT_EXCEL]:
    folder.mkdir(parents=True, exist_ok=True)

# Meses exactos de interes
TARGET_PERIODS = [
    {"period_label": "2016_09_septiembre", "start": "2016-09-01", "end": "2016-10-01"},
    {"period_label": "2016_12_diciembre", "start": "2016-12-01", "end": "2017-01-01"},
    {"period_label": "2017_05_mayo", "start": "2017-05-01", "end": "2017-06-01"},
    {"period_label": "2017_07_julio", "start": "2017-07-01", "end": "2017-08-01"},
    {"period_label": "2017_08_agosto", "start": "2017-08-01", "end": "2017-09-01"},
]

# Productos a revisar
# S2_SR_BOA = Sentinel-2 BOA/SR cuando exista en GEE
# S2_TOA_FALLBACK = Sentinel-2 TOA/L1C para inspeccion visual o respaldo
PRODUCTS_TO_SEARCH = ["S2_SR_BOA", "S2_TOA_FALLBACK"]

# En esta etapa NO filtramos por nubosidad, para revisar visualmente todo.
FILTER_BY_CLOUD_TILE = False
MAX_CLOUD_TILE = 100.0

# Exportacion
EXPORT_AS_RECTANGLE = True
BUFFER_METERS = 150
EXPORT_CRS = "EPSG:32618"
EXPORT_SCALE = 10

# Indices
L = 0.5

# Procesos locales
SPLIT_BANDS = True
MAKE_PANELS = True
MAKE_CONTACT_SHEETS = True


# =============================================================================
# 2. INICIALIZAR EARTH ENGINE
# =============================================================================

def initialize_ee():
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"Earth Engine inicializado correctamente con proyecto: {GEE_PROJECT}")
    except Exception as e:
        print("No se pudo inicializar Earth Engine directamente.")
        print(e)
        print("Iniciando autenticacion...")
        ee.Authenticate(force=True)
        ee.Initialize(project=GEE_PROJECT)
        print(f"Earth Engine autenticado con proyecto: {GEE_PROJECT}")


# =============================================================================
# 3. AOI DESDE KML
# =============================================================================

def parse_kml_polygons(kml_path):
    """Lee poligonos simples desde un KML y devuelve ee.FeatureCollection."""
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    tree = ET.parse(kml_path)
    root = tree.getroot()
    placemarks = root.findall(".//kml:Placemark", ns)

    features = []
    for i, pm in enumerate(placemarks, start=1):
        name_el = pm.find("kml:name", ns)
        zone_name = name_el.text.strip() if name_el is not None and name_el.text else f"ZONE_{i:02d}"

        # Toma todos los anillos encontrados dentro del Placemark y usa el primero como poligono principal.
        coords_elements = pm.findall(".//kml:coordinates", ns)
        if not coords_elements:
            continue

        coords_text = coords_elements[0].text.strip() if coords_elements[0].text else ""
        coords = []
        for part in coords_text.split():
            values = part.split(",")
            if len(values) >= 2:
                coords.append([float(values[0]), float(values[1])])

        if len(coords) >= 4:
            geom = ee.Geometry.Polygon([coords])
            features.append(ee.Feature(geom, {"zone_id": zone_name, "source": "KML"}))

    return ee.FeatureCollection(features)


def load_aoi():
    fallback_coords = [
        [-73.322381843292, 3.80929830965208],
        [-73.319322262507, 3.79898368614295],
        [-73.3130944767671, 3.80039269628281],
        [-73.3165070861042, 3.8107073029227],
        [-73.322381843292, 3.80929830965208],
    ]

    if KML_PATH.exists():
        zones = parse_kml_polygons(KML_PATH)
        try:
            n_zones = zones.size().getInfo()
        except Exception:
            n_zones = 0

        if n_zones > 0:
            print("KML cargado:", KML_PATH)
            print("Numero de zonas/clones leidos:", n_zones)
            return zones

        print("El KML existe, pero no se pudieron leer poligonos. Usando fallback.")
    else:
        print("No se encontro KML local. Usando poligono de respaldo.")

    return ee.FeatureCollection([
        ee.Feature(ee.Geometry.Polygon([fallback_coords]), {"zone_id": "AOI_TOTAL", "source": "fallback"})
    ])


# =============================================================================
# 4. UTILIDADES
# =============================================================================

def clean_text(value, max_len=None):
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_\-\.]", "_", value)
    if max_len is not None:
        value = value[:max_len]
    return value


def fmt_pct(value):
    try:
        x = float(value)
        if math.isnan(x):
            return "NA"
        return f"{x:05.1f}".replace(".", "p")
    except Exception:
        return "NA"


def fmt_idx(value):
    try:
        x = float(value)
        if math.isnan(x):
            return "NA"
        prefix = "m" if x < 0 else "p"
        return prefix + f"{abs(x):0.3f}".replace(".", "p")
    except Exception:
        return "NA"


def feature_collection_to_dataframe(fc):
    info = fc.getInfo()
    rows = [f.get("properties", {}) for f in info.get("features", [])]
    return pd.DataFrame(rows)


def collection_to_feature_collection(collection, feature_function):
    n = collection.size().getInfo()
    if n == 0:
        return ee.FeatureCollection([]), 0
    image_list = collection.toList(n)
    fc = ee.FeatureCollection(image_list.map(lambda img: feature_function(ee.Image(img))))
    return fc, n


def get_collection_id(product_type):
    if product_type == "S2_SR_BOA":
        return "COPERNICUS/S2_SR_HARMONIZED"
    if product_type == "S2_TOA_FALLBACK":
        return "COPERNICUS/S2_HARMONIZED"
    raise ValueError(f"Producto no reconocido: {product_type}")


def build_full_gee_image_id(product_type, image_id):
    image_id = str(image_id)
    if image_id.startswith("COPERNICUS/"):
        return image_id
    if product_type == "S2_SR_BOA":
        return "COPERNICUS/S2_SR_HARMONIZED/" + image_id
    if product_type == "S2_TOA_FALLBACK":
        return "COPERNICUS/S2_HARMONIZED/" + image_id
    raise ValueError(f"Producto no reconocido: {product_type}")


# =============================================================================
# 5. BANDAS, MASCARAS E INDICES
# =============================================================================

S2_SR_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"
]

S2_TOA_BANDS = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"
]

INDEX_BANDS = [
    "NDVI", "RVI", "DVI", "SAVI", "OSAVI", "MSAVI", "ARVI", "SARVI", "EVI",
    "GNDVI", "NDRE", "NDMI", "NBR", "BSI", "CI_GREEN", "CI_REDEDGE", "MTCI", "FVC_NDVI_PROXY"
]

QUALITY_BANDS_SR = ["SCL_CLASS", "CLOUD_MASK", "SHADOW_MASK", "CLOUD_SHADOW_MASK", "CLEAR_MASK", "VALID_MASK", "AOI_MASK"]
QUALITY_BANDS_TOA = ["QA60_CLASS", "CLOUD_MASK", "SHADOW_MASK", "CLOUD_SHADOW_MASK", "CLEAR_MASK", "VALID_MASK", "AOI_MASK"]


def s2_sr_masks(image):
    scl = image.select("SCL")
    valid = scl.neq(0)

    cloud = scl.eq(8).Or(scl.eq(9)).Or(scl.eq(10))
    shadow = scl.eq(3)
    cloud_shadow = cloud.Or(shadow)

    clear = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(7))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )
    return valid, cloud, shadow, cloud_shadow, clear


def s2_toa_masks(image):
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11

    cloud = qa.bitwiseAnd(cloud_bit).neq(0).Or(qa.bitwiseAnd(cirrus_bit).neq(0))
    valid = image.select("B2").mask()
    clear = valid.And(cloud.Not())

    shadow = cloud.multiply(0).rename("SHADOW_MASK")
    cloud_shadow = cloud
    return valid, cloud, shadow, cloud_shadow, clear


def add_s2_indices(scaled):
    blue = scaled.select("B2")
    green = scaled.select("B3")
    red = scaled.select("B4")
    re1 = scaled.select("B5")
    re2 = scaled.select("B6")
    nir = scaled.select("B8")
    nirn = scaled.select("B8A")
    swir1 = scaled.select("B11")
    swir2 = scaled.select("B12")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    rvi = nir.divide(red).rename("RVI")
    dvi = nir.subtract(red).rename("DVI")

    savi = nir.subtract(red).multiply(1 + L).divide(nir.add(red).add(L)).rename("SAVI")
    osavi = nir.subtract(red).multiply(1.16).divide(nir.add(red).add(0.16)).rename("OSAVI")

    msavi = (
        nir.multiply(2).add(1)
        .subtract(
            nir.multiply(2).add(1)
            .pow(2)
            .subtract(nir.subtract(red).multiply(8))
            .max(0)
            .sqrt()
        )
        .divide(2)
        .rename("MSAVI")
    )

    rb = red.multiply(2).subtract(blue)
    arvi = nir.subtract(rb).divide(nir.add(rb)).rename("ARVI")
    sarvi = nir.subtract(rb).multiply(1 + L).divide(nir.add(rb).add(L)).rename("SARVI")

    evi = nir.subtract(red).multiply(2.5).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    ).rename("EVI")

    gndvi = nir.subtract(green).divide(nir.add(green)).rename("GNDVI")
    ndre = nirn.subtract(re1).divide(nirn.add(re1)).rename("NDRE")
    ndmi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDMI")
    nbr = nir.subtract(swir2).divide(nir.add(swir2)).rename("NBR")

    bsi = swir1.add(red).subtract(nir.add(blue)).divide(
        swir1.add(red).add(nir).add(blue)
    ).rename("BSI")

    ci_green = nir.divide(green).subtract(1).rename("CI_GREEN")
    ci_rededge = nirn.divide(re1).subtract(1).rename("CI_REDEDGE")
    mtci = re2.subtract(re1).divide(re1.subtract(red)).rename("MTCI")

    fvc_ndvi_proxy = ndvi.subtract(0.2).divide(0.86 - 0.2).clamp(0, 1).rename("FVC_NDVI_PROXY")

    return ee.Image.cat([
        ndvi, rvi, dvi, savi, osavi, msavi, arvi, sarvi, evi, gndvi, ndre,
        ndmi, nbr, bsi, ci_green, ci_rededge, mtci, fvc_ndvi_proxy
    ]).toFloat()


def make_aoi_mask(aoi):
    return ee.Image.constant(1).clip(aoi).unmask(0).rename("AOI_MASK").toFloat()


def prepare_full_image(image, product_type, aoi, clip_region):
    image = ee.Image(image)

    if product_type == "S2_SR_BOA":
        bands = S2_SR_BANDS
        valid, cloud, shadow, cloud_shadow, clear = s2_sr_masks(image)
        scaled = image.select(bands).divide(10000).toFloat()
        classification = image.select("SCL").rename("SCL_CLASS").toFloat()
    else:
        bands = S2_TOA_BANDS
        valid, cloud, shadow, cloud_shadow, clear = s2_toa_masks(image)
        scaled = image.select(bands).divide(10000).toFloat()
        classification = image.select("QA60").rename("QA60_CLASS").toFloat()

    indices = add_s2_indices(scaled)

    out = (
        scaled
        .addBands(indices)
        .addBands(classification)
        .addBands(cloud.rename("CLOUD_MASK").toFloat())
        .addBands(shadow.rename("SHADOW_MASK").toFloat())
        .addBands(cloud_shadow.rename("CLOUD_SHADOW_MASK").toFloat())
        .addBands(clear.rename("CLEAR_MASK").toFloat())
        .addBands(valid.rename("VALID_MASK").toFloat())
        .addBands(make_aoi_mask(aoi))
        .clip(clip_region)
    )
    return out


# =============================================================================
# 6. METADATOS
# =============================================================================

def image_to_metadata_feature_factory(product_type, period_label, start, end, aoi):
    def image_to_metadata_feature(image):
        image = ee.Image(image)
        date = image.date()

        if product_type == "S2_SR_BOA":
            valid, cloud, shadow, cloud_shadow, clear = s2_sr_masks(image)
            scaled_clear = image.select(S2_SR_BANDS).divide(10000).toFloat().updateMask(clear)
        else:
            valid, cloud, shadow, cloud_shadow, clear = s2_toa_masks(image)
            scaled_clear = image.select(S2_TOA_BANDS).divide(10000).toFloat().updateMask(clear)

        pixel_area = ee.Image.pixelArea()
        area_img = ee.Image.cat([
            pixel_area.updateMask(valid).unmask(0).rename("valid_m2"),
            pixel_area.updateMask(cloud).unmask(0).rename("cloud_m2"),
            pixel_area.updateMask(shadow).unmask(0).rename("shadow_m2"),
            pixel_area.updateMask(cloud_shadow).unmask(0).rename("cloud_shadow_m2"),
            pixel_area.updateMask(clear).unmask(0).rename("clear_m2"),
        ])

        area_stats = area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=20,
            maxPixels=1e8,
            tileScale=4,
        )

        valid_area = ee.Number(area_stats.get("valid_m2")).max(1)
        cloud_aoi = ee.Number(area_stats.get("cloud_m2")).divide(valid_area).multiply(100)
        shadow_aoi = ee.Number(area_stats.get("shadow_m2")).divide(valid_area).multiply(100)
        cloud_shadow_aoi = ee.Number(area_stats.get("cloud_shadow_m2")).divide(valid_area).multiply(100)
        clear_aoi = ee.Number(area_stats.get("clear_m2")).divide(valid_area).multiply(100)

        indices_clear = add_s2_indices(scaled_clear)
        index_stats = indices_clear.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=20,
            maxPixels=1e8,
            tileScale=4,
        )

        return ee.Feature(None, {
            "period_label": period_label,
            "product_type": product_type,
            "start": start,
            "end": end,
            "image_id": image.id(),
            "date": date.format("YYYY-MM-dd"),
            "year": date.get("year"),
            "month": date.get("month"),
            "tile_id": image.get("MGRS_TILE"),
            "cloud_tile_pct": image.get("CLOUDY_PIXEL_PERCENTAGE"),
            "cloud_aoi_pct": cloud_aoi,
            "shadow_aoi_pct": shadow_aoi,
            "cloud_shadow_aoi_pct": cloud_shadow_aoi,
            "clear_aoi_pct": clear_aoi,
            "NDVI_AOI": index_stats.get("NDVI"),
            "MSAVI_AOI": index_stats.get("MSAVI"),
            "NDRE_AOI": index_stats.get("NDRE"),
            "NDMI_AOI": index_stats.get("NDMI"),
            "FVC_NDVI_PROXY_AOI": index_stats.get("FVC_NDVI_PROXY"),
            "manual_select": "",
            "manual_notes": "",
        })

    return image_to_metadata_feature


def collect_candidates(period, product_type, aoi):
    period_label = period["period_label"]
    start = period["start"]
    end = period["end"]
    collection_id = get_collection_id(product_type)

    collection = ee.ImageCollection(collection_id).filterBounds(aoi).filterDate(start, end).sort("system:time_start")

    if FILTER_BY_CLOUD_TILE:
        collection = collection.filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD_TILE))

    n = collection.size().getInfo()
    print(f"{period_label} | {product_type} | {start} a {end} | imagenes encontradas: {n}")

    if n == 0:
        return pd.DataFrame()

    feature_fn = image_to_metadata_feature_factory(product_type, period_label, start, end, aoi)
    metadata_fc, _ = collection_to_feature_collection(collection, feature_fn)
    df = feature_collection_to_dataframe(metadata_fc)

    if df.empty:
        return df

    numeric_cols = [
        "cloud_tile_pct", "cloud_aoi_pct", "shadow_aoi_pct", "cloud_shadow_aoi_pct",
        "clear_aoi_pct", "NDVI_AOI", "MSAVI_AOI", "NDRE_AOI", "NDMI_AOI", "FVC_NDVI_PROXY_AOI",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["date", "cloud_shadow_aoi_pct"]).reset_index(drop=True)
    return df


def make_filename(props):
    period = clean_text(props.get("period_label", "period"), 25)
    product = clean_text(props.get("product_type", "product"), 20)
    date = clean_text(props.get("date", "date"), 12)
    image_id = clean_text(props.get("image_id", "image"), 22)
    tile = clean_text(props.get("tile_id", "tile"), 8)

    filename = (
        f"{period}_{product}_{date}_{image_id}_{tile}"
        f"_TC{fmt_pct(props.get('cloud_tile_pct'))}"
        f"_AC{fmt_pct(props.get('cloud_aoi_pct'))}"
        f"_CS{fmt_pct(props.get('cloud_shadow_aoi_pct'))}"
        f"_CL{fmt_pct(props.get('clear_aoi_pct'))}"
        f"_NDVI{fmt_idx(props.get('NDVI_AOI'))}"
        f"_NDMI{fmt_idx(props.get('NDMI_AOI'))}.tif"
    )
    return OUT_TIFF / filename


def download_candidate(props, aoi, export_region, clip_region):
    product_type = props["product_type"]
    image_id_full = build_full_gee_image_id(product_type, props["image_id"])
    out_tif = make_filename(props)

    if out_tif.exists():
        print("Ya existe, se omite:", out_tif.name)
        return True, str(out_tif)

    print("\nDescargando candidata:")
    print(out_tif.name)
    print("ID:", image_id_full)

    try:
        image = ee.Image(image_id_full)
        export_image = prepare_full_image(image, product_type, aoi, clip_region)

        geemap.download_ee_image(
            image=export_image,
            filename=str(out_tif),
            region=export_region,
            scale=EXPORT_SCALE,
            crs=EXPORT_CRS,
            overwrite=True,
        )
        time.sleep(1)
        return True, str(out_tif)
    except Exception as e:
        print("ERROR descargando:", image_id_full)
        print(e)
        return False, ""


# =============================================================================
# 7. PROCESAMIENTO LOCAL DE TIFF: BANDAS Y PANELES
# =============================================================================

SR_LOCAL_BANDS = S2_SR_BANDS + INDEX_BANDS + QUALITY_BANDS_SR
TOA_LOCAL_BANDS = S2_TOA_BANDS + INDEX_BANDS + QUALITY_BANDS_TOA


def get_local_band_names(src, tif_path):
    descriptions = list(src.descriptions)
    if descriptions and len(descriptions) == src.count:
        names = [str(d).strip() for d in descriptions if d is not None and str(d).strip()]
        if len(names) == src.count:
            return names

    upper_name = Path(tif_path).name.upper()
    if "S2_SR_BOA" in upper_name and len(SR_LOCAL_BANDS) == src.count:
        return SR_LOCAL_BANDS
    if "S2_TOA_FALLBACK" in upper_name and len(TOA_LOCAL_BANDS) == src.count:
        return TOA_LOCAL_BANDS
    return [f"band_{i}" for i in range(1, src.count + 1)]


def find_band_index(band_names, target):
    target = target.lower()
    for i, name in enumerate(band_names, start=1):
        if str(name).lower() == target:
            return i
    return None


def read_band(src, band_names, target):
    idx = find_band_index(band_names, target)
    if idx is None:
        return None
    arr = src.read(idx, masked=True).astype("float32").filled(np.nan)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    return arr


def stretch_band(arr, mask=None, p_low=2, p_high=98):
    if arr is None:
        return None

    arr = arr.astype("float32")
    if mask is not None:
        vals = arr[np.isfinite(arr) & mask]
    else:
        vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return np.zeros_like(arr, dtype="float32")

    lo = np.nanpercentile(vals, p_low)
    hi = np.nanpercentile(vals, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = np.nanmin(vals)
        hi = np.nanmax(vals)
    if hi <= lo:
        return np.zeros_like(arr, dtype="float32")

    out = (arr - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out[~np.isfinite(out)] = 0
    return out


def make_rgb(red, green, blue, mask=None):
    r = stretch_band(red, mask)
    g = stretch_band(green, mask)
    b = stretch_band(blue, mask)
    return np.dstack([r, g, b])


def pct_from_mask(mask_arr, aoi_pixels=None):
    if mask_arr is None:
        return np.nan
    vals = mask_arr[aoi_pixels] if aoi_pixels is not None else mask_arr.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(100 * np.mean(vals > 0.5))


def mean_from_arr(arr, aoi_pixels=None):
    if arr is None:
        return np.nan
    vals = arr[aoi_pixels] if aoi_pixels is not None else arr.ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.nanmean(vals))


def split_bands_to_individual_tifs(tif_path):
    tif_path = Path(tif_path)
    short_stem = clean_text(tif_path.stem, 110)
    out_dir = OUT_BANDS / short_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    with rasterio.open(tif_path) as src:
        band_names = get_local_band_names(src, tif_path)
        profile = src.profile.copy()
        profile.update(count=1, compress="deflate")

        for band_number, band_name in enumerate(band_names, start=1):
            out_path = out_dir / f"{short_stem}_B{band_number:02d}_{clean_text(band_name, 30)}.tif"
            if not out_path.exists():
                arr = src.read(band_number)
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(arr, 1)
                    dst.set_band_description(1, band_name)
            records.append({
                "source_tif": str(tif_path),
                "band_number": band_number,
                "band_name": band_name,
                "band_tif": str(out_path),
            })
    return pd.DataFrame(records)


def create_panel_from_tif(tif_path, props=None):
    tif_path = Path(tif_path)
    with rasterio.open(tif_path) as src:
        band_names = get_local_band_names(src, tif_path)

        blue = read_band(src, band_names, "B2")
        green = read_band(src, band_names, "B3")
        red = read_band(src, band_names, "B4")
        nir = read_band(src, band_names, "B8")

        ndvi = read_band(src, band_names, "NDVI")
        msavi = read_band(src, band_names, "MSAVI")
        ndmi = read_band(src, band_names, "NDMI")
        clear_mask = read_band(src, band_names, "CLEAR_MASK")
        cloud_mask = read_band(src, band_names, "CLOUD_MASK")
        aoi_mask = read_band(src, band_names, "AOI_MASK")

        if blue is None or green is None or red is None or nir is None:
            print("No se pudieron leer bandas RGB/NIR:", tif_path.name)
            return None

        if aoi_mask is not None:
            aoi_pixels = np.isfinite(aoi_mask) & (aoi_mask > 0.5)
        else:
            aoi_pixels = np.isfinite(red)

        rgb = make_rgb(red, green, blue, aoi_pixels)
        false_color = make_rgb(nir, red, green, aoi_pixels)

        cloud_pct = pct_from_mask(cloud_mask, aoi_pixels)
        clear_pct = pct_from_mask(clear_mask, aoi_pixels)
        ndvi_mean = mean_from_arr(ndvi, aoi_pixels)
        ndmi_mean = mean_from_arr(ndmi, aoi_pixels)

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()

        if props is None:
            title_left = tif_path.name
        else:
            title_left = f"{props.get('period_label', '')} | {props.get('product_type', '')} | {props.get('date', '')}"

        title = (
            f"{title_left}\n"
            f"Nube mascara AOI: {cloud_pct:.2f}% | Clear mascara AOI: {clear_pct:.2f}% | "
            f"NDVI medio: {ndvi_mean:.3f} | NDMI medio: {ndmi_mean:.3f}"
        )
        fig.suptitle(title, fontsize=10, fontweight="bold")

        axes[0].imshow(rgb)
        axes[0].set_title("RGB B4/B3/B2")
        axes[0].axis("off")

        axes[1].imshow(false_color)
        axes[1].set_title("Falso color B8/B4/B3")
        axes[1].axis("off")

        if ndvi is not None:
            im = axes[2].imshow(np.where(aoi_pixels, ndvi, np.nan), vmin=-0.2, vmax=1.0, cmap="RdYlGn")
            axes[2].set_title("NDVI")
            plt.colorbar(im, ax=axes[2], fraction=0.046)
        else:
            axes[2].text(0.5, 0.5, "NDVI no disponible", ha="center")
        axes[2].axis("off")

        if msavi is not None:
            im = axes[3].imshow(np.where(aoi_pixels, msavi, np.nan), vmin=-0.2, vmax=1.0, cmap="RdYlGn")
            axes[3].set_title("MSAVI")
            plt.colorbar(im, ax=axes[3], fraction=0.046)
        else:
            axes[3].text(0.5, 0.5, "MSAVI no disponible", ha="center")
        axes[3].axis("off")

        if ndmi is not None:
            im = axes[4].imshow(np.where(aoi_pixels, ndmi, np.nan), vmin=-0.5, vmax=0.8, cmap="BrBG")
            axes[4].set_title("NDMI")
            plt.colorbar(im, ax=axes[4], fraction=0.046)
        else:
            axes[4].text(0.5, 0.5, "NDMI no disponible", ha="center")
        axes[4].axis("off")

        if clear_mask is not None:
            im = axes[5].imshow(clear_mask, vmin=0, vmax=1, cmap="gray")
            axes[5].set_title("CLEAR_MASK")
            plt.colorbar(im, ax=axes[5], fraction=0.046)
        else:
            axes[5].text(0.5, 0.5, "CLEAR_MASK no disponible", ha="center")
        axes[5].axis("off")

        plt.tight_layout()
        return fig


def make_contact_sheets(panels_df):
    if panels_df.empty:
        return pd.DataFrame()

    records = []
    for period_label, group in panels_df.groupby("period_label"):
        group = group.sort_values(["product_type", "date", "panel_png"])
        pngs = list(group["panel_png"])
        if not pngs:
            continue

        cols = 3
        rows = int(np.ceil(len(pngs) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        axes = np.array(axes).ravel()

        for ax in axes:
            ax.axis("off")

        for ax, png_path in zip(axes, pngs):
            try:
                img = plt.imread(png_path)
                ax.imshow(img)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error leyendo\n{Path(png_path).name}", ha="center", va="center")
                print("Error en contact sheet:", png_path, e)
            ax.axis("off")

        fig.suptitle(f"Revision visual - {period_label}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        out_png = OUT_CONTACT / f"{clean_text(period_label)}_CONTACT_SHEET.png"
        fig.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig)

        records.append({"period_label": period_label, "contact_sheet": str(out_png), "n_panels": len(pngs)})
        print("Contact sheet guardado:", out_png)

    return pd.DataFrame(records)


# =============================================================================
# 8. PROCESO PRINCIPAL
# =============================================================================

def main():
    initialize_ee()
    zones_fc = load_aoi()
    aoi = zones_fc.geometry()

    if EXPORT_AS_RECTANGLE:
        export_region = aoi.buffer(BUFFER_METERS).bounds()
        clip_region = export_region
        print("Modo exportacion: ventana rectangular con buffer.")
    else:
        export_region = aoi
        clip_region = aoi
        print("Modo exportacion: recorte exacto al AOI.")

    print("Carpeta de salida:", OUT_DIR)

    all_metadata = []
    all_downloads = []
    all_splits = []
    all_panels = []

    for period in TARGET_PERIODS:
        period_label = period["period_label"]
        print("\n" + "=" * 100)
        print("PERIODO:", period_label)
        print("=" * 100)

        for product_type in PRODUCTS_TO_SEARCH:
            df = collect_candidates(period, product_type, aoi)
            if df.empty:
                continue

            metadata_csv = OUT_CSV / f"metadata_{period_label}_{product_type}.csv"
            df.to_csv(metadata_csv, index=False, encoding="utf-8-sig")
            all_metadata.append(df)

            for _, row in df.iterrows():
                props = row.to_dict()
                ok, tif_path = download_candidate(props, aoi, export_region, clip_region)

                record = props.copy()
                record["downloaded"] = ok
                record["tif_path"] = tif_path
                all_downloads.append(record)

                if not ok or not tif_path:
                    continue

                if SPLIT_BANDS:
                    try:
                        split_df = split_bands_to_individual_tifs(tif_path)
                        all_splits.append(split_df)
                    except Exception as e:
                        print("ERROR separando bandas:", tif_path)
                        print(e)

                if MAKE_PANELS:
                    try:
                        fig = create_panel_from_tif(tif_path, props)
                        if fig is not None:
                            out_png = OUT_FIG / f"{clean_text(Path(tif_path).stem, 140)}_PANEL.png"
                            fig.savefig(out_png, dpi=170, bbox_inches="tight")
                            plt.close(fig)

                            panel_record = props.copy()
                            panel_record["tif_path"] = tif_path
                            panel_record["panel_png"] = str(out_png)
                            all_panels.append(panel_record)
                            print("Panel guardado:", out_png)
                    except Exception as e:
                        print("ERROR generando panel:", tif_path)
                        print(e)

    all_metadata_df = pd.concat(all_metadata, ignore_index=True) if all_metadata else pd.DataFrame()
    downloads_df = pd.DataFrame(all_downloads)
    splits_df = pd.concat(all_splits, ignore_index=True) if all_splits else pd.DataFrame()
    panels_df = pd.DataFrame(all_panels)
    contacts_df = make_contact_sheets(panels_df) if MAKE_CONTACT_SHEETS else pd.DataFrame()

    metadata_csv = OUT_CSV / "ALL_metadata_candidatas.csv"
    downloads_csv = OUT_CSV / "ALL_descargas_candidatas.csv"
    splits_csv = OUT_CSV / "ALL_bandas_individuales.csv"
    panels_csv = OUT_CSV / "ALL_paneles_visuales.csv"
    contacts_csv = OUT_CSV / "ALL_contact_sheets.csv"

    all_metadata_df.to_csv(metadata_csv, index=False, encoding="utf-8-sig")
    downloads_df.to_csv(downloads_csv, index=False, encoding="utf-8-sig")
    splits_df.to_csv(splits_csv, index=False, encoding="utf-8-sig")
    panels_df.to_csv(panels_csv, index=False, encoding="utf-8-sig")
    contacts_df.to_csv(contacts_csv, index=False, encoding="utf-8-sig")

    excel_path = OUT_EXCEL / "catalogo_revision_visual_candidatas_s2.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        all_metadata_df.to_excel(writer, sheet_name="metadata_candidatas", index=False)
        downloads_df.to_excel(writer, sheet_name="descargas", index=False)
        panels_df.to_excel(writer, sheet_name="paneles_visuales", index=False)
        contacts_df.to_excel(writer, sheet_name="contact_sheets", index=False)
        splits_df.to_excel(writer, sheet_name="bandas_individuales", index=False)

        if not downloads_df.empty:
            selection_table = downloads_df.copy()
            selection_table["manual_select"] = ""
            selection_table["manual_notes"] = ""

            first_cols = [
                "manual_select", "manual_notes", "period_label", "product_type", "date", "image_id",
                "cloud_tile_pct", "cloud_aoi_pct", "cloud_shadow_aoi_pct", "clear_aoi_pct",
                "NDVI_AOI", "NDMI_AOI", "tif_path",
            ]
            first_cols = [c for c in first_cols if c in selection_table.columns]
            other_cols = [c for c in selection_table.columns if c not in first_cols]
            selection_table = selection_table[first_cols + other_cols]
            selection_table.to_excel(writer, sheet_name="SELECCION_MANUAL", index=False)

    print("\n" + "=" * 100)
    print("DESCARGA DE CANDIDATAS FINALIZADA")
    print("=" * 100)
    print("Carpeta principal:", OUT_DIR)
    print("TIFF multibanda:", OUT_TIFF)
    print("Bandas individuales:", OUT_BANDS)
    print("Paneles visuales:", OUT_FIG)
    print("Contact sheets:", OUT_CONTACT)
    print("Excel de revision:", excel_path)
    print("CSV metadata:", metadata_csv)
    print("\nProximo paso manual:")
    print("1. Abre la carpeta 04_CONTACT_SHEETS.")
    print("2. Revisa visualmente cada panel por mes.")
    print("3. Abre el Excel de revision.")
    print("4. En la hoja SELECCION_MANUAL coloca 1 en manual_select para la mejor imagen de cada mes.")
    print("5. Despues se corre un segundo script solo con las imagenes seleccionadas.")
    print("=" * 100)


if __name__ == "__main__":
    main()
