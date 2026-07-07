#!/usr/bin/env python3
"""
Descarga la imagen Sentinel-2 L2A (BOA oficial) del 2017-08-14
directamente a disco local, usando el mismo metodo que ya
funciono para las otras 2 imagenes (Copernicus Data Space).

Requisitos:
    pip install requests
"""

import requests
import os
import getpass

OUTPUT_DIR = os.path.expanduser("~/analisis_gulupa/02_L2A_BOA_completo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

AOI_WKT = "POLYGON((-73.3272 3.7958, -73.3087 3.7958, -73.3087 3.8156, -73.3272 3.8156, -73.3272 3.7958))"
FECHA_INI = "2017-08-13T00:00:00.000Z"
FECHA_FIN = "2017-08-15T00:00:00.000Z"

def obtener_token(username, password):
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }
    r = requests.post(url, data=data)
    print("Respuesta autenticacion:", r.status_code)
    if r.status_code != 200:
        print(r.text)
        return None
    return r.json()["access_token"]

def buscar_producto_l2a():
    url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
    filtro = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"contains(Name,'MSIL2A') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{AOI_WKT}') and "
        f"ContentDate/Start gt {FECHA_INI} and ContentDate/Start lt {FECHA_FIN}"
    )
    r = requests.get(url, params={"$filter": filtro, "$top": 5})
    print("Respuesta busqueda:", r.status_code)
    if r.status_code != 200:
        print(r.text)
        return []
    return r.json().get("value", [])

def descargar_producto(product_id, nombre_archivo, token):
    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    r = requests.get(url, headers=headers, stream=True, allow_redirects=True)
    print("Respuesta descarga:", r.status_code)
    if r.status_code != 200:
        print(r.text[:500])
        return False
    ruta = os.path.join(OUTPUT_DIR, nombre_archivo)
    total = 0
    with open(ruta, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
            total += len(chunk)
            print(f"\r  Descargando... {total/1024/1024:.1f} MB", end="")
    print(f"\n  Descargado: {ruta}")
    return True

if __name__ == "__main__":
    print("=== Buscando producto L2A (BOA) para 2017-08-14 ===")
    productos = buscar_producto_l2a()
    print(f"Encontrados: {len(productos)}")
    for p in productos:
        print(f"  - {p['Name']} (Id: {p['Id']})")

    if not productos:
        print("No se encontro producto L2A oficial para esta fecha/AOI.")
        exit(0)

    print("\nIngresa tus credenciales de Copernicus Data Space")
    username = input("Usuario/Email: ")
    password = getpass.getpass("Password: ")

    token = obtener_token(username, password)
    if not token:
        print("No se pudo obtener el token.")
        exit(1)

    p = productos[0]
    nombre_zip = p["Name"] + ".zip"
    print(f"\nDescargando: {nombre_zip}")
    descargar_producto(p["Id"], nombre_zip, token)
