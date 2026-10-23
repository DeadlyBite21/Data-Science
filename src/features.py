"""Selección de variables, transformaciones y homologación entre dominios."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# Variables de CICIoMT2024 que pueden reflejar condiciones de la captura
# (temporización o tamaño de ventana) más que el comportamiento del ataque.
# Se excluyen solo si el notebook 02 lo indica.
VARIABLES_SOSPECHOSAS = ["IAT", "Number", "Weight"]


def columnas_constantes(X):
    return [c for c in X.columns if X[c].nunique(dropna=False) <= 1]


def columnas_correlacionadas(X, umbral=0.98, muestra=200_000, random_state=42):
    """Columnas cuya correlación absoluta con otra anterior supera el umbral.

    De cada par se conserva la primera columna en el orden original.
    """
    if len(X) > muestra:
        X = X.sample(muestra, random_state=random_state)
    corr = X.corr().abs()
    superior = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

    eliminar = {}
    for columna in superior.columns:
        pares = superior.index[superior[columna] > umbral].tolist()
        conservadas = [p for p in pares if p not in eliminar]
        if conservadas:
            eliminar[columna] = conservadas[0]
    return eliminar


def seleccionar_variables(X_train, umbral_corr=0.98, excluir=()):
    """Decide qué variables usar, ajustando solo con entrenamiento.

    Devuelve la lista seleccionada y una tabla con el motivo de cada exclusión.
    """
    motivos = {c: "Excluida manualmente" for c in excluir if c in X_train}
    restantes = X_train.drop(columns=list(motivos))

    for c in columnas_constantes(restantes):
        motivos[c] = "Constante en entrenamiento"
    restantes = restantes.drop(columns=columnas_constantes(restantes))

    for c, par in columnas_correlacionadas(restantes, umbral_corr).items():
        motivos[c] = f"Correlación > {umbral_corr} con {par}"

    seleccionadas = [c for c in X_train.columns if c not in motivos]
    informe = pd.DataFrame(
        {"Motivo": pd.Series(motivos, dtype="object")}
    ).rename_axis("Variable")
    return seleccionadas, informe


def log_signo(X):
    """log(1 + |x|) conservando el signo; reduce colas muy largas (p. ej. Rate)."""
    return np.sign(X) * np.log1p(np.abs(X))


def preprocesador(escalar=False):
    """Imputación por mediana y, para modelos lineales, log + estandarización.

    Los árboles no necesitan escalar porque solo comparan umbrales.
    """
    pasos = [("imputar", SimpleImputer(strategy="median"))]
    if escalar:
        pasos += [
            ("log", FunctionTransformer(log_signo, feature_names_out="one-to-one")),
            ("escalar", StandardScaler()),
        ]
    return Pipeline(pasos)


# ---------------------------------------------------------------------------
# Estabilidad de las variables (Domain Shift / Concept Drift)
# ---------------------------------------------------------------------------

def psi(referencia, actual, bins=10):
    """Population Stability Index entre dos muestras de una variable.

    Las variables con pocos valores distintos (p. ej. banderas 0/1) se
    comparan por valor; las continuas, por cuantiles de la referencia.
    Guía habitual: < 0,1 estable; 0,1–0,25 cambio moderado; > 0,25 cambio alto.
    """
    referencia = pd.Series(referencia).dropna()
    actual = pd.Series(actual).dropna()
    if referencia.empty or actual.empty:
        return np.nan

    if referencia.nunique() <= bins:
        valores = sorted(set(referencia) | set(actual))
        p = referencia.value_counts(normalize=True).reindex(valores, fill_value=0).to_numpy()
        q = actual.value_counts(normalize=True).reindex(valores, fill_value=0).to_numpy()
    else:
        interiores = np.unique(np.quantile(referencia, np.linspace(0, 1, bins + 1))[1:-1])
        cortes = np.concatenate([[-np.inf], interiores, [np.inf]])
        p = np.histogram(referencia, cortes)[0] / len(referencia)
        q = np.histogram(actual, cortes)[0] / len(actual)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def estabilidad_variables(A, B, variables, muestra=100_000, random_state=42):
    """Compara la distribución de cada variable entre dos conjuntos.

    Usa la estadística de Kolmogorov-Smirnov (0 = idénticas, 1 = disjuntas)
    y el PSI. Se calcula sobre una muestra para acotar el tiempo.
    """
    if len(A) > muestra:
        A = A.sample(muestra, random_state=random_state)
    if len(B) > muestra:
        B = B.sample(muestra, random_state=random_state)

    filas = []
    for v in variables:
        a, b = A[v].dropna(), B[v].dropna()
        filas.append({
            "Variable": v,
            "KS": ks_2samp(a, b).statistic if len(a) and len(b) else np.nan,
            "PSI": psi(a, b),
        })
    return pd.DataFrame(filas).set_index("Variable").sort_values("KS")


# ---------------------------------------------------------------------------
# Homologación entre CICIoMT2024 y redes tradicionales (CICFlowMeter)
# ---------------------------------------------------------------------------
# Ambos datasets usan extractores distintos: CICIoMT2024 resume ventanas de
# paquetes y CICFlowMeter resume flujos completos. Solo se conservan
# variables con un significado equivalente, expresadas en la misma escala
# (proporciones de banderas, longitudes de paquete y tasa de paquetes).
#
# Diferencias detectadas al comparar los datos reales:
# - Longitud: CICIoMT2024 mide el paquete completo (un SYN pesa 54 bytes:
#   14 Ethernet + 20 IP + 20 TCP); CICFlowMeter mide solo la carga útil.
#   Con `alinear=True` se suman las cabeceras a CICFlowMeter.
# - Banderas SYN y ACK: en CIC-DDoS2019 el SYN flood tiene SYN Flag Count = 0
#   y ACK Flag Count = 1 en todos los flujos (error conocido de
#   CICFlowMeter-V3), por lo que no son comparables entre dominios.
# - Tasa: `Rate` (IoMT) y `Flow Pkts/s` (CICFlowMeter) tienen escalas muy
#   distintas; `normalizar_por_dominio` lleva cada dominio a percentiles.

BANDERAS = ["fin", "syn", "rst", "psh", "ack", "ece", "cwr"]
BANDERAS_IDS2018 = {
    "fin": "FIN Flag Cnt", "syn": "SYN Flag Cnt", "rst": "RST Flag Cnt",
    "psh": "PSH Flag Cnt", "ack": "ACK Flag Cnt", "ece": "ECE Flag Cnt",
    "cwr": "CWE Flag Count",
}
LONGITUDES = ["long_min", "long_max", "long_media", "long_std"]
VARIABLES_COMUNES = (
    ["prop_tcp", "prop_udp"]
    + [f"bandera_{b}" for b in BANDERAS]
    + LONGITUDES + ["tasa_paquetes"]
)
# Variables comunes sin las banderas que CICFlowMeter-V3 mide mal
VARIABLES_COMUNES_FIABLES = [
    v for v in VARIABLES_COMUNES if v not in ("bandera_syn", "bandera_ack")
]
# Cabeceras Ethernet (14) + IP (20) que CICFlowMeter no incluye
CABECERA_ETHERNET_IP = 34


def homologar_iomt(datos):
    comun = pd.DataFrame({
        "prop_tcp": datos["TCP"],
        "prop_udp": datos["UDP"],
        **{f"bandera_{b}": datos[f"{b}_flag_number"] for b in BANDERAS},
        "long_min": datos["Min"],
        "long_max": datos["Max"],
        "long_media": datos["AVG"],
        "long_std": datos["Std"],
        "tasa_paquetes": datos["Rate"],
    })
    return comun.join(datos[["clase", "etiqueta"]])


def homologar_cicflowmeter(datos, alinear=False):
    """Homologa CSE-CIC-IDS2018 o CIC-DDoS2019 (nombres cortos de 2018).

    Con `alinear=True` las longitudes pasan de carga útil a paquete completo,
    sumando la cabecera de transporte media del flujo y Ethernet + IP.
    """
    numerico = lambda c: pd.to_numeric(datos[c], errors="coerce")
    protocolo = numerico("Protocol")
    paquetes = (numerico("Tot Fwd Pkts") + numerico("Tot Bwd Pkts")).clip(lower=1)

    cabecera = 0
    if alinear:
        transporte = (numerico("Fwd Header Len") + numerico("Bwd Header Len")) / paquetes
        cabecera = transporte.clip(lower=0) + CABECERA_ETHERNET_IP

    comun = pd.DataFrame({
        "prop_tcp": (protocolo == 6).astype(float),
        "prop_udp": (protocolo == 17).astype(float),
        **{
            f"bandera_{b}": (numerico(col) / paquetes).clip(0, 1)
            for b, col in BANDERAS_IDS2018.items()
        },
        "long_min": numerico("Pkt Len Min") + cabecera,
        "long_max": numerico("Pkt Len Max") + cabecera,
        "long_media": numerico("Pkt Len Mean") + cabecera,
        "long_std": numerico("Pkt Len Std"),
        "tasa_paquetes": numerico("Flow Pkts/s"),
    })
    comun = comun.replace([np.inf, -np.inf], np.nan)
    extra = [c for c in ["clase", "etiqueta", "Label"] if c in datos]
    return comun.join(datos[extra])


def normalizar_por_dominio(datos, variables, columnas=None, n_cuantiles=1000):
    """Reemplaza longitudes y tasa por su percentil dentro del dominio.

    Elimina diferencias de unidades y escala entre extractores. Solo usa los
    valores de las variables (no las etiquetas), por lo que puede aplicarse
    al dominio de destino sin filtrar información de clase. Es una forma
    simple de adaptación de dominio no supervisada.
    """
    from sklearn.preprocessing import QuantileTransformer

    continuas = LONGITUDES + ["tasa_paquetes"]
    columnas = columnas or [v for v in variables if v in continuas]
    resultado = datos.copy()
    transformador = QuantileTransformer(
        n_quantiles=min(n_cuantiles, len(datos)), random_state=42
    )
    resultado[columnas] = transformador.fit_transform(
        datos[columnas].fillna(datos[columnas].median())
    )
    return resultado
