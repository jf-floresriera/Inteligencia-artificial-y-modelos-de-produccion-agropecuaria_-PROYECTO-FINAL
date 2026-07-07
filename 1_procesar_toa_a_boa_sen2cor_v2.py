#!/usr/bin/env python3
"""
Procesa productos Sentinel-2 L1C (.SAFE, en TOA) a Nivel 2A (BOA)
usando Sen2Cor 2.11.00, para las fechas de interes:
    2016-09-13, 2016-12-22
"""

import os
import subprocess
import argparse
import glob

SEN2COR_BIN = os.path.expanduser("~/Sen2Cor-02.10.01-Linux64/bin/L2A_Process")

FECHAS_INTERES = ["20160913", "20161222"]

def encontrar_safe_por_fecha(input_dir, fecha):
    patron = os.path.join(input_dir, f"*MSIL1C*{fecha}*.SAFE")
    return glob.glob(patron)

def procesar_sen2cor(ruta_safe, output_dir, resolucion=10):
    cmd = [
        SEN2COR_BIN,
        ruta_safe,
        "--output_dir", output_dir,
        "--resolution", str(resolucion),
    ]
    comando_str = " ".join(cmd)
    print(f"Ejecutando: {comando_str}")
    resultado = subprocess.run(cmd, capture_output=True, text=True)

    if resultado.returncode == 0:
        print(f"  OK -> Procesado correctamente: {ruta_safe}")
    else:
        print(f"  ERROR procesando {ruta_safe}")
        print("STDOUT:", resultado.stdout[-2000:])
        print("STDERR:", resultado.stderr[-2000:])

    return resultado.returncode == 0

def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for fecha in FECHAS_INTERES:
        print(f"\n=== Buscando producto L1C para fecha {fecha} ===")
        safes = encontrar_safe_por_fecha(input_dir, fecha)

        if not safes:
            print(f"  No se encontro ningun .SAFE para {fecha} en {input_dir}")
            continue

        for safe in safes:
            procesar_sen2cor(safe, output_dir)

    print("\n=== Proceso finalizado ===")
    print(f"Revisa las carpetas *_MSIL2A_*.SAFE dentro de: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
