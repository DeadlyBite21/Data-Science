"""Métricas y visualizaciones para clasificación binaria y multiclase."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import CARPETA_RESULTADOS, CLASES


def puntaje_ataque(modelo, X):
    """Probabilidad de DDoS, si el modelo la entrega (para PR-AUC)."""
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return 1 - proba[:, 0]  # multiclase: probabilidad de no ser benigno
    return None


def metricas_binarias(y_true, y_pred, puntaje=None, clase_real=None):
    """Métricas para benigno (0) frente a DDoS (1).

    `clase_real` (etiqueta multiclase) permite ver qué variante de DDoS
    se detecta y cuál no.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metricas = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision DDoS": precision_score(y_true, y_pred, zero_division=0),
        "Recall DDoS": recall_score(y_true, y_pred, zero_division=0),
        "F1 DDoS": f1_score(y_true, y_pred, zero_division=0),
        "F1 macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Tasa falsos positivos": fp / (fp + tn) if fp + tn else np.nan,
    }
    if puntaje is not None and len(np.unique(y_true)) == 2:
        metricas["PR-AUC"] = average_precision_score(y_true, puntaje)

    if clase_real is not None:
        clase_real = np.asarray(clase_real)
        for clase, nombre in CLASES.items():
            if clase > 0 and (clase_real == clase).any():
                metricas[f"Recall {nombre.split()[-1]}"] = (
                    y_pred[clase_real == clase] == 1
                ).mean()

    metricas["Falsas alarmas"] = int(fp)
    metricas["Ataques no detectados"] = int(fn)
    return metricas


def metricas_multiclase(y_true, y_pred, clases=CLASES):
    """Métricas macro, recall por clase y tasa de falsos positivos.

    La tasa de falsos positivos considera como alarma cualquier benigno
    clasificado como alguna variante de DDoS.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    etiquetas = list(clases)
    benignos = y_true == 0

    metricas = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision macro": precision_score(
            y_true, y_pred, labels=etiquetas, average="macro", zero_division=0
        ),
        "Recall macro": recall_score(
            y_true, y_pred, labels=etiquetas, average="macro", zero_division=0
        ),
        "F1 macro": f1_score(
            y_true, y_pred, labels=etiquetas, average="macro", zero_division=0
        ),
    }
    recalls = recall_score(
        y_true, y_pred, labels=etiquetas, average=None, zero_division=0
    )
    for clase, valor in zip(etiquetas, recalls):
        metricas[f"Recall {clases[clase]}"] = valor
    metricas["Tasa falsos positivos"] = (
        (y_pred[benignos] != 0).mean() if benignos.any() else np.nan
    )
    return metricas


def evaluar(modelo, X, y, tipo="binaria", clase_real=None):
    pred = modelo.predict(X)
    if tipo == "binaria":
        return metricas_binarias(y, pred, puntaje_ataque(modelo, X), clase_real)
    return metricas_multiclase(y, pred)


def matriz_confusion(y_true, y_pred, tipo="binaria", titulo="", ax=None,
                     normalizar=None):
    """Dibuja la matriz de confusión con nombres de clase legibles."""
    if tipo == "binaria":
        etiquetas, nombres = [0, 1], ["Benigno", "DDoS"]
    else:
        etiquetas, nombres = list(CLASES), list(CLASES.values())

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        labels=etiquetas,
        display_labels=nombres,
        normalize=normalizar,
        cmap="Blues",
        values_format=".3f" if normalizar else "d",
        colorbar=False,
        ax=ax,
    )
    (ax or plt.gca()).set_title(titulo)


def guardar_tabla(tabla, nombre):
    CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_RESULTADOS / f"{nombre}.csv"
    tabla.to_csv(ruta)
    return ruta
