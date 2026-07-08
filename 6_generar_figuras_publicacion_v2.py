#!/usr/bin/env python3
"""
Figuras estilo IJRS/MDPI para el analisis de caucho (Hevea brasiliensis)
Sentinel-2 BOA multitemporal (2016-09-13, 2016-12-22, 2017-08-14)

Usa fuente serif (Times New Roman si esta disponible, si no
Liberation Serif/DejaVu Serif como fallback automatico de matplotlib)
y paneles rotulados (A), (B), (C)...

Requisitos:
    pip install matplotlib rasterio numpy pandas
    sudo apt install ttf-mscorefonts-installer -y  (opcional, para Times real)
"""

import os
import glob
import string
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.linewidth":     0.7,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.top":          True,
    "ytick.right":        True,
    "figure.dpi":         150,
    "savefig.dpi":        300,
})

BASE_DIR = os.path.expanduser("~/analisis_gulupa/03_BOA_recortado_finca")
OUT_DIR  = os.path.expanduser("~/analisis_gulupa/06_figuras_publicacion")
os.makedirs(OUT_DIR, exist_ok=True)

CULTIVO = "caucho (Hevea brasiliensis)"
fechas_legibles = {"20160913": "13 Sep 2016", "20161222": "22 Dec 2016", "20170814": "14 Aug 2017"}

LETTERS = list(string.ascii_uppercase)

def panel_label(ax, idx, fs=11):
    ax.text(0.02, 0.97, f"({LETTERS[idx]})", transform=ax.transAxes,
            fontsize=fs, fontweight="bold", va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

def cargar_banda(prod_dir, banda, res="R10m"):
    files = glob.glob(os.path.join(prod_dir, f"{res}_*_{banda}*.tif"))
    if not files:
        return None
    with rasterio.open(files[0]) as src:
        arr = src.read(1).astype(float)
        arr[arr == 0] = np.nan
        return arr

def calcular_ndvi(prod_dir):
    red = cargar_banda(prod_dir, "B04")
    nir = cargar_banda(prod_dir, "B08")
    if red is None or nir is None:
        return None
    return (nir - red) / (nir + red)

productos = sorted(glob.glob(os.path.join(BASE_DIR, "*MSIL2A*")))

ndvi_cmap = LinearSegmentedColormap.from_list(
    "ndvi_cmap", ["#8B4513", "#D2B48C", "#FFFF99", "#7FBF7F", "#228B22", "#006400"]
)

# ============================================================
# FIGURA 1: Mapas NDVI multitemporales (A, B, C)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
ndvi_arrays = {}

for i, prod_dir in enumerate(productos):
    nombre = os.path.basename(prod_dir)
    fecha_key = nombre.split("_")[2][:8]
    ndvi = calcular_ndvi(prod_dir)
    ndvi_arrays[fecha_key] = ndvi

    im = axes[i].imshow(ndvi, cmap=ndvi_cmap, vmin=0, vmax=0.8)
    axes[i].set_title(fechas_legibles.get(fecha_key, fecha_key), fontsize=11, fontweight="bold")
    axes[i].set_xlabel("Columna (píxel)", fontsize=9)
    if i == 0:
        axes[i].set_ylabel("Fila (píxel)", fontsize=9)
    axes[i].tick_params(labelsize=7)
    panel_label(axes[i], i)

cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.18, aspect=40)
cbar.set_label("NDVI", fontsize=10)
fig.suptitle(f"Distribución espacial del NDVI en el área de cultivo de {CULTIVO}",
             fontsize=12, y=1.02)
plt.savefig(os.path.join(OUT_DIR, "Fig1_NDVI_mapas_multitemporal.png"), bbox_inches="tight", facecolor="white")
plt.close()
print("Fig1 guardada")

# ============================================================
# FIGURA 2: Boxplot de reflectancia por banda y fecha (A-D)
# ============================================================
bandas = ["B02", "B03", "B04", "B08"]
nombres_banda = {"B02": "Azul (490 nm)", "B03": "Verde (560 nm)", "B04": "Rojo (665 nm)", "B08": "NIR (842 nm)"}

fig, axes = plt.subplots(1, 4, figsize=(13, 4.2))
colores = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd"]

for j, banda in enumerate(bandas):
    datos_banda, etiquetas = [], []
    for prod_dir in productos:
        nombre = os.path.basename(prod_dir)
        fecha_key = nombre.split("_")[2][:8]
        arr = cargar_banda(prod_dir, banda)
        if arr is not None:
            valid = arr[~np.isnan(arr)]
            datos_banda.append(valid)
            etiquetas.append(fechas_legibles.get(fecha_key, fecha_key))

    bp = axes[j].boxplot(datos_banda, labels=etiquetas, patch_artist=True,
                          widths=0.5, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(colores[j])
        patch.set_alpha(0.6)
    axes[j].set_title(nombres_banda[banda], fontsize=10)
    axes[j].set_ylabel("Reflectancia BOA (x10\u2074)" if j == 0 else "", fontsize=9)
    axes[j].tick_params(axis="x", rotation=30, labelsize=7)
    axes[j].tick_params(axis="y", labelsize=8)
    panel_label(axes[j], j, fs=10)

fig.suptitle(f"Distribución de reflectancia de superficie (BOA) por banda espectral — {CULTIVO}", fontsize=12, y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig2_boxplot_reflectancia.png"), bbox_inches="tight", facecolor="white")
plt.close()
print("Fig2 guardada")

# ============================================================
# FIGURA 3: Serie temporal de NDVI (panel unico A)
# ============================================================
fechas_orden = sorted(ndvi_arrays.keys())
medias = [np.nanmean(ndvi_arrays[f]) for f in fechas_orden]
stds = [np.nanstd(ndvi_arrays[f]) for f in fechas_orden]
labels_x = [fechas_legibles.get(f, f) for f in fechas_orden]

fig, ax = plt.subplots(figsize=(7, 4.8))
ax.errorbar(labels_x, medias, yerr=stds, fmt="o-", color="#228B22",
            ecolor="#8FBC8F", elinewidth=1.5, capsize=4, markersize=7,
            markerfacecolor="#006400", linewidth=1.5)
ax.set_ylabel("NDVI (media ± DE)", fontsize=10)
ax.set_xlabel("Fecha de adquisición", fontsize=10)
ax.set_title(f"Evolución temporal del NDVI en el cultivo de {CULTIVO}", fontsize=11, fontweight="bold")
ax.set_ylim(0.2, 0.75)
ax.grid(True, linestyle="--", alpha=0.4)
panel_label(ax, 0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig3_serie_temporal_NDVI.png"), bbox_inches="tight", facecolor="white")
plt.close()
print("Fig3 guardada")

# ============================================================
# FIGURA 4: Histogramas comparativos de NDVI (panel unico A)
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.8))
colores_hist = ["#1f77b4", "#ff7f0e", "#2ca02c"]

for k, f in enumerate(fechas_orden):
    valid = ndvi_arrays[f][~np.isnan(ndvi_arrays[f])]
    ax.hist(valid, bins=30, alpha=0.5, label=fechas_legibles.get(f, f),
            color=colores_hist[k], edgecolor="black", linewidth=0.3)

ax.set_xlabel("NDVI", fontsize=10)
ax.set_ylabel("Frecuencia (n\u00b0 de píxeles)", fontsize=10)
ax.set_title(f"Distribución de frecuencias del NDVI por fecha — {CULTIVO}", fontsize=11, fontweight="bold")
ax.legend(frameon=False, fontsize=8)
panel_label(ax, 0)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Fig4_histograma_NDVI.png"), bbox_inches="tight", facecolor="white")
plt.close()
print("Fig4 guardada")

print(f"\\n=== Todas las figuras guardadas en: {OUT_DIR} ===")
