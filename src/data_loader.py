"""Ingesta y limpieza de los flujos de red utilizados en el proyecto.

Dominio IoMT: CICIoMT2024 (benigno, DDoS SYN y DDoS UDP).
Dominio tradicional (opcional, etapa III): CSE-CIC-IDS2018.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
CARPETA_DATOS = Path(os.environ.get("NIDS_DATA_DIR", RAIZ_PROYECTO / "data"))
CARPETA_RAW = CARPETA_DATOS / "raw" / "CICIoMT2024"
CARPETA_IDS2018 = CARPETA_DATOS / "raw" / "CSE-CIC-IDS2018"
CARPETA_PROCESADOS = CARPETA_DATOS / "processed"
CARPETA_RESULTADOS = RAIZ_PROYECTO / "deliverables" / "resultados"
CARPETA_MODELOS = RAIZ_PROYECTO / "models"

# 0 = benigno; las clases 1 y 2 son variantes de DDoS
CLASES = {0: "Benigno", 1: "DDoS SYN", 2: "DDoS UDP"}
COLUMNAS_OBJETIVO = ["clase", "etiqueta"]

ARCHIVOS = {
    "train": {
        0: "Benign_train.pcap.csv",
        1: "TCP_IP-DDoS-SYN1_train.pcap.csv",
        2: "TCP_IP-DDoS-UDP1_train.pcap.csv",
    },
    "test": {
        0: "Benign_test.pcap.csv",
        1: "TCP_IP-DDoS-SYN_test.pcap.csv",
        2: "TCP_IP-DDoS-UDP1_test.pcap.csv",
    },
}

ARCHIVO_IDS2018 = "Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv"


def ruta_archivo(particion, clase):
    return CARPETA_RAW / particion / ARCHIVOS[particion][clase]


def verificar_archivos():
    """Indica qué archivos de CICIoMT2024 están disponibles en data/raw/."""
    filas = [
        {
            "Partición": particion,
            "Clase": CLASES[clase],
            "Archivo": archivo,
            "Disponible": ruta_archivo(particion, clase).exists(),
        }
        for particion, archivos in ARCHIVOS.items()
        for clase, archivo in archivos.items()
    ]
    estado = pd.DataFrame(filas)

    if not estado["Disponible"].all():
        faltantes = estado.loc[~estado["Disponible"], "Archivo"].tolist()
        raise FileNotFoundError(
            f"Faltan archivos en {CARPETA_RAW}: {faltantes}. "
            "Revisar DESCARGA_DATOS.txt."
        )
    return estado


def cargar_particion(particion, clases=None, nrows=None):
    """Lee los CSV de una partición ("train" o "test") y agrega las etiquetas.

    `clase` es la etiqueta multiclase y `etiqueta` la binaria (1 = DDoS).
    """
    partes = []
    for clase, archivo in ARCHIVOS[particion].items():
        if clases is not None and clase not in clases:
            continue
        df = pd.read_csv(ruta_archivo(particion, clase), nrows=nrows)
        df.columns = df.columns.str.strip()
        partes.append(df.assign(clase=clase))

    datos = pd.concat(partes, ignore_index=True)
    datos["etiqueta"] = (datos["clase"] > 0).astype(int)
    return datos


def columnas_caracteristicas(datos):
    return [c for c in datos.columns if c not in COLUMNAS_OBJETIVO]


def resumen_calidad(datos):
    """Faltantes, infinitos, duplicados y filas con etiquetas contradictorias."""
    columnas = columnas_caracteristicas(datos)
    numericas = datos[columnas].select_dtypes(include="number")
    contradictorias = (
        datos.groupby(columnas, dropna=False)["clase"].transform("nunique") > 1
    )
    return pd.Series({
        "Registros": len(datos),
        "Valores faltantes": int(datos[columnas].isna().sum().sum()),
        "Valores infinitos": int(np.isinf(numericas).sum().sum()),
        "Filas duplicadas": int(datos.duplicated().sum()),
        "Filas contradictorias": int(contradictorias.sum()),
    })


def limpiar(datos, eliminar_filas=True):
    """Prepara una copia de los datos; los CSV originales no se modifican.

    - Reemplaza infinitos por NaN (se imputan luego dentro del pipeline).
    - Si `eliminar_filas` es True, elimina filas con características
      idénticas y clases distintas, y duplicados exactos, para que una
      misma fila no quede a la vez en entrenamiento y validación.
      En la prueba oficial se usa False para evaluar el tráfico tal cual.
    """
    base = datos.replace([np.inf, -np.inf], np.nan)
    columnas = columnas_caracteristicas(base)

    contradictorias = (
        base.groupby(columnas, dropna=False)["clase"].transform("nunique") > 1
    )
    if not eliminar_filas:
        contradictorias[:] = False
    base = base.loc[~contradictorias]

    antes = len(base)
    if eliminar_filas:
        base = base.drop_duplicates()

    informe = {
        "Registros iniciales": len(datos),
        "Contradictorias eliminadas": int(contradictorias.sum()),
        "Duplicados eliminados": antes - len(base),
        "Registros finales": len(base),
    }
    return base.reset_index(drop=True), informe


def guardar_procesado(datos, nombre):
    CARPETA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_PROCESADOS / f"{nombre}.parquet"
    datos.to_parquet(ruta, index=False)
    return ruta


def cargar_procesado(nombre):
    ruta = CARPETA_PROCESADOS / f"{nombre}.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecutar antes los notebooks 01 y 02."
        )
    return pd.read_parquet(ruta)


def guardar_variables(config):
    CARPETA_PROCESADOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_PROCESADOS / "variables.json"
    ruta.write_text(json.dumps(config, indent=2, ensure_ascii=False))
    return ruta


def cargar_variables():
    """Configuración de variables definida en el notebook 02."""
    ruta = CARPETA_PROCESADOS / "variables.json"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe {ruta}. Ejecutar antes el notebook 02.")
    return json.loads(ruta.read_text())


def cargar_ids2018(archivo=ARCHIVO_IDS2018, nrows=None):
    """Lee un CSV de CSE-CIC-IDS2018 (Processed Traffic Data for ML Algorithms).

    Devuelve None si el archivo no fue descargado: el dominio tradicional
    es opcional en el código actual.
    """
    ruta = CARPETA_IDS2018 / archivo
    if not ruta.exists():
        return None

    datos = pd.read_csv(ruta, nrows=nrows, low_memory=False)
    datos.columns = datos.columns.str.strip()
    # Algunos CSV de 2018 repiten la fila de encabezado en medio del archivo
    datos = datos.loc[datos["Label"] != "Label"].copy()
    datos["etiqueta"] = (datos["Label"].str.lower() != "benign").astype(int)
    return datos
