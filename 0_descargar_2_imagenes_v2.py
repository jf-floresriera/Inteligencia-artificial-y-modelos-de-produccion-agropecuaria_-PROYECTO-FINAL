#!/usr/bin/env python3
"""
Descarga las 2 imagenes .SAFE de Sentinel-2 L1C (2016-09-13 y 2016-12-22)
directamente a tu disco local.

CORRECCION: usa autenticacion con usuario/contrasena (cliente publico
cdse-public), que es el que funciona con el endpoint de descarga OData.
El Client ID/Secret de Sentinel Hub (sh-...) NO sirve para este endpoint.

Requisitos:
    pip install requests
"""

import requests
import os
import getpass

OUTPUT_DIR = os.path.expanduser("~/analisis_gulupa/01_L1C_TOA_completo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# IDs de los productos ya encontrados en el paso anterior
PRODUCTOS = {
    "2016-09-13": {
        "id": "f42def81-fdc9-4f34-a369-4766ff6dc6bc",
        "nombre": "S2A_MSIL1C_20160913T151702_N0500_R125_T18NXK_20230930T171916.SAFE.zip",
    },
    "2016-12-22": {
        "id": "3b81d020-8ea7-4425-889c-0159f6360746",
        "nombre": "S2A_MSIL1C_20161222T151702_N0500_R125_T18NXK_20230924T054704.SAFE.zip",
    },
}

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
    print("Ingresa tus credenciales de Copernicus Data Space (dataspace.copernicus.eu)")
    username = input("Usuario/Email: ")
    password = getpass.getpass("Password: ")

    token = obtener_token(username, password)
    if not token:
        print("No se pudo obtener el token. Revisa usuario/contrasena.")
        exit(1)
    print("Token obtenido correctamente\n")

    for fecha, info in PRODUCTOS.items():
        print(f"=== Descargando {fecha} ===")
        descargar_producto(info["id"], info["nombre"], token)
        print()
