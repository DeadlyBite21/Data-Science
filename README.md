# Sistema Inteligente de Detección y Clasificación de Intrusiones en Redes (NIDS)

Este repositorio contiene el desarrollo del proyecto semestral de **Ciencia de Datos**, enfocado en la construcción de una solución integral de Machine Learning para la detección temprana y clasificación multiclase de tráfico de red malicioso (Network Intrusion Detection System).

El proyecto sigue la metodología estándar **CRISP-DM** (Cross-Industry Standard Process for Data Mining), abordando desde el entendimiento y preprocesamiento de datos de tráfico hasta la validación y despliegue de modelos predictivos.

---

## 📌 Descripción del Problema y Objetivos

Los centros de operaciones de seguridad (SOC) se enfrentan a un volumen masivo de alertas de red generadas por sistemas de firmas estáticas, lo que produce una alta tasa de falsos positivos y dificulta la detección de ataques complejos o desconocidos. 

* **Objetivo General:** Desarrollar y evaluar un pipeline de Ciencia de Datos capaz de clasificar flujos de red en tiempo real, diferenciando tráfico benigno de amenazas específicas con alto desempeño y explicabilidad técnica.
* **Objetivos Específicos:**
  1. Realizar la limpieza, normalización y tratamiento de desbalance en flujos de red de alta dimensionalidad.
  2. Implementar y comparar múltiples algoritmos de clasificación (árboles basados en ensambles, boosting y redes neuronales).
  3. Evaluar el rendimiento utilizando métricas robustas para datos desbalanceados ($F_1\text{-Score}$ Macro, Precision-Recall AUC y FPR).
  4. Interpretar los patrones del tráfico mediante técnicas de explicabilidad (SHAP / Feature Importance)[cite: 1].
  5. Desplegar un prototipo interactivo (dashboard) para la visualización de alertas y métricas clave[cite: 1].

---

## 📊 Dataset Utilizado

El proyecto utiliza como referencia los conjuntos de datos de ciberseguridad del **Canadian Institute for Cybersecurity (CIC)**:
* **CSE-CIC-IDS2018 / CIC-IDS2017:** Metadatos de flujos de red con más de 70 características extraídas (duración de flujos, tasas de paquetes inter-arribo, banderas TCP, bytes/segundo) y etiquetas de diversas clases de ataque (DDoS, DoS, PortScan, Brute Force, Infiltration, Botnets).

> **Nota sobre los datos:** Siguiendo las directrices del curso respecto a la sensibilidad y tamaño de los datasets[cite: 1], los archivos CSV sin procesar no están almacenados directamente en este repositorio. Puedes descargarlos desde el portal oficial de UNB/CIC y colocarlos en la carpeta `data/raw/`.

---

## 🛠️ Estructura del Repositorio

```text
├── data/
│   ├── raw/                 # Datos originales (no versionados)
│   └── processed/           # Datos procesados y limpios listos para modelado
├── notebooks/
│   ├── 01_eda_and_cleaning.ipynb    # Limpieza, tratamiento de nulos/infinitos y EDA
│   ├── 02_preprocessing_fe.ipynb    # Selección de variables y balanceo
│   ├── 03_baseline_models.ipynb     # Primeros modelos y línea base (Etapa II)
│   └── 04_final_models_eval.ipynb   # Comparación, optimización y validación (Etapa III)
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Scripts para ingesta y limpieza de flujos
│   ├── features.py          # Pipelines de transformación y escalado
│   └── evaluate.py          # Funciones de cálculo de métricas y visualización
├── app/
│   └── app.py               # Aplicación interactiva / Dashboard (Streamlit)
├── deliverables/
│   ├── posters/             # Póster oficial para la Feria de Proyectos TIC
│   └── presentations/       # Diapositivas de las etapas evaluadas
├── requirements.txt         # Dependencias del entorno de ejecución
├── .gitignore
└── README.md
