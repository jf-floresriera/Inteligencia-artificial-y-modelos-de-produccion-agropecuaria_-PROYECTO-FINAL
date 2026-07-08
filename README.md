# Proyecto Final · IA y Producción Agropecuaria

<div align="center">

<img src="https://img.shields.io/badge/IA-Aplicada-6f42c1?style=for-the-badge" alt="IA Aplicada">
<img src="https://img.shields.io/badge/Agro-Datos-2ea44f?style=for-the-badge" alt="Agro Datos">
<img src="https://img.shields.io/badge/Remote%20Sensing-Sentinel--2-0366d6?style=for-the-badge" alt="Remote Sensing">
<img src="https://img.shields.io/badge/Machine%20Learning-Modelado-f39c12?style=for-the-badge" alt="Machine Learning">

</div>

> Repositorio del proyecto final de la asignatura **Inteligencia Artificial y Modelos de Producción Agropecuaria**. Integra procesamiento geoespacial, análisis exploratorio, detección de atípicos, generación de índices y modelado predictivo con datos del sector agropecuario.

## Objetivo

Construir un flujo de trabajo reproducible para analizar información agropecuaria a partir de imágenes, variables derivadas e indicadores espectrales, y luego evaluar modelos de inteligencia artificial para apoyar la interpretación del sistema productivo.

## Alcance del repositorio

- Descarga, preparación y recorte de imágenes.
- Conversión y procesamiento de productos Sentinel-2.
- Cálculo de índices y extracción de variables de interés.
- Análisis exploratorio, PCA y clustering.
- Detección de valores atípicos.
- Modelado con Ridge, Lasso, Random Forest y XGBoost.
- Generación de mapas y figuras para publicación.

## Estructura principal

- `05_reporte_calidad.csv`
- `06_figuras_publicacion/`
- `09_indices_finca/`
- `0_AREA.py`
- `0_descargar_2_imagenes_v2.py`
- `0_descargar_imagenes_tiff_s2_revision_visual.py`
- `10_dataset_modelado_FINAL.csv`
- `10_dataset_modelado_POR_MES.csv`
- `11_dataset_LIMPIO_sin_atipicos.csv`
- `16_deteccion_atipicos/`
- `1_procesar_toa_a_boa_sen2cor_v2.py`
- `2_descargar_BOA_2017_GEE.py`
- `3_descargar_L2A_2017_local.py`
- `4_recortar_finca.py`
- `5_analisis_calidad.py`
- `6_generar_figuras_publicacion_v2.py`
- `7_mapa_ubicacion_caucho.py`
- `F1_recorte_jerarquico.py`
- `F2c_calcular_indices_v2_estructura_real.py`
- `F3_extraccion_final_3fechas.py`
- `F3_extraccion_puntos_campo.py`
- `F3c_outliers_FINAL.py`
- `F4_EDA_PCA_Clustering.py`
- `F5_F6_OPCION_A_v2.py`
- `F5_F6_Ridge_Lasso_RF_XGBoost.py`
- `F5_F6_SLOO_TUNEO_COMPLETO.py`
- `F7_extrapolacion_lotes.py`
- `lotes_gulupa.geojson`

## Flujo metodológico

1. **Adquisición de datos**: scripts para descarga y preparación inicial de imágenes y capas espaciales.
2. **Preprocesamiento**: correcciones, recortes y organización de insumos para análisis.
3. **Construcción de variables**: cálculo de índices, extracción de valores y armado del dataset final.
4. **Análisis exploratorio**: visualización, PCA, clustering y revisión de patrones.
5. **Control de calidad**: detección y análisis de atípicos.
6. **Modelado**: comparación de modelos regularizados y de ensamble.
7. **Salida final**: mapas, tablas y figuras listas para reporte o publicación.

## Scripts clave

| Archivo | Rol dentro del proyecto |
|---|---|
| `5_analisis_calidad.py` | Revisión de calidad de datos e insumos. |
| `6_generar_figuras_publicacion_v2.py` | Construcción de figuras finales del proyecto. |
| `F3c_outliers_FINAL.py` | Detección de valores atípicos. |
| `F4_EDA_PCA_Clustering.py` | Análisis exploratorio, PCA y clustering. |
| `F5_F6_Ridge_Lasso_RF_XGBoost.py` | Modelado con algoritmos supervisados. |
| `F5_F6_SLOO_TUNEO_COMPLETO.py` | Validación y ajuste de modelos. |
| `F7_extrapolacion_lotes.py` | Extrapolación de resultados a lotes. |

## Librerías usadas en este repositorio

Las siguientes librerías fueron detectadas directamente en los scripts del proyecto:

| Librería | Uso principal | Versión recomendada |
|---|---|---|
| `geopandas` | Manejo de capas vectoriales | `0.14+` |
| `joblib` | Serialización y apoyo en pipelines | `1.3+` |
| `matplotlib` | Gráficas base y figuras | `3.8+` |
| `numpy` | Cómputo numérico y arreglos | `1.26+` |
| `pandas` | Manejo de tablas y datasets | `2.x` |
| `rasterio` | Lectura y procesamiento raster | `1.3+` |
| `requests` | Descarga de recursos | `2.31+` |
| `scipy` | Funciones científicas y apoyo estadístico | `1.11+` |
| `shapely` | Geometrías espaciales | `2.x` |
| `scikit-learn` | PCA, clustering y modelos ML | `1.4+` |
| `xgboost` | Modelado boosting | `2.x` |

> Nota: las versiones se proponen como una base compatible para replicar el entorno. Si ya tienes un ambiente funcional del curso, puedes fijarlas en un `requirements.txt` o `environment.yml`.

## Entorno sugerido

```bash
python -m venv .venv
source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn geopandas rasterio shapely scipy xgboost opencv-python joblib requests folium
```

## Figuras del proyecto

Estas imágenes están enlazadas desde las subcarpetas reales del repositorio para que el README se vea visual en GitHub.

### `06_figuras_publicacion/Fig1_NDVI_mapas_multitemporal.png`

![Figura del proyecto](06_figuras_publicacion/Fig1_NDVI_mapas_multitemporal.png)

### `06_figuras_publicacion/Fig2_boxplot_reflectancia.png`

![Figura del proyecto](06_figuras_publicacion/Fig2_boxplot_reflectancia.png)

### `06_figuras_publicacion/Fig3_serie_temporal_NDVI.png`

![Figura del proyecto](06_figuras_publicacion/Fig3_serie_temporal_NDVI.png)

### `06_figuras_publicacion/Fig4_histograma_NDVI.png`

![Figura del proyecto](06_figuras_publicacion/Fig4_histograma_NDVI.png)

### `06_figuras_publicacion/FigLoc_location_map_caucho.png`

![Figura del proyecto](06_figuras_publicacion/FigLoc_location_map_caucho.png)

### `16_deteccion_atipicos/Fig_outliers_PCA.png`

![Figura del proyecto](16_deteccion_atipicos/Fig_outliers_PCA.png)

## Resultados y productos detectados

- `10_dataset_modelado_FINAL.csv`
- `10_dataset_modelado_POR_MES.csv`
- `11_dataset_LIMPIO_sin_atipicos.csv`
- `05_reporte_calidad.csv`
- Figuras finales en `06_figuras_publicacion/`
- Resultados de atípicos en `16_deteccion_atipicos/`

## Cómo clonar y revisar

```bash
git clone https://github.com/jf-floresriera/Inteligencia-artificial-y-modelos-de-produccion-agropecuaria_-PROYECTO-FINAL.git
cd Inteligencia-artificial-y-modelos-de-produccion-agropecuaria_-PROYECTO-FINAL
```

## Contacto

> Agrega aquí tus correos de contacto para que queden visibles en GitHub.
>
> - `tu_correo_1@ejemplo.com`
> - `tu_correo_2@ejemplo.com`

## Mejoras futuras

- Añadir `requirements.txt` o `environment.yml`.
- Incluir métricas comparativas de modelos en una tabla.
- Agregar GIF o portada de presentación.
- Integrar una sección de conclusiones principales del proyecto.
