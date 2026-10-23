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
# Homologación entre CICIoMT2024 y CSE-CIC-IDS2018
# ---------------------------------------------------------------------------
# Ambos datasets usan extractores distintos: CICIoMT2024 resume ventanas de
# paquetes y CIC-IDS2018 resume flujos completos (CICFlowMeter). Solo se
# conservan variables con un significado equivalente, expresadas en la misma
# escala (proporciones de banderas, longitudes de paquete y tasa de paquetes).
# Es una aproximación y debe reportarse como limitación.

BANDERAS = ["fin", "syn", "rst", "psh", "ack", "ece", "cwr"]
BANDERAS_IDS2018 = {
    "fin": "FIN Flag Cnt", "syn": "SYN Flag Cnt", "rst": "RST Flag Cnt",
    "psh": "PSH Flag Cnt", "ack": "ACK Flag Cnt", "ece": "ECE Flag Cnt",
    "cwr": "CWE Flag Count",
}
VARIABLES_COMUNES = (
    ["prop_tcp", "prop_udp"]
    + [f"bandera_{b}" for b in BANDERAS]
    + ["long_min", "long_max", "long_media", "long_std", "tasa_paquetes"]
)


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
    return comun.join(datos[["etiqueta"]])


def homologar_ids2018(datos):
    numerico = lambda c: pd.to_numeric(datos[c], errors="coerce")
    protocolo = numerico("Protocol")
    paquetes = (numerico("Tot Fwd Pkts") + numerico("Tot Bwd Pkts")).clip(lower=1)

    comun = pd.DataFrame({
        "prop_tcp": (protocolo == 6).astype(float),
        "prop_udp": (protocolo == 17).astype(float),
        **{
            f"bandera_{b}": (numerico(col) / paquetes).clip(0, 1)
            for b, col in BANDERAS_IDS2018.items()
        },
        "long_min": numerico("Pkt Len Min"),
        "long_max": numerico("Pkt Len Max"),
        "long_media": numerico("Pkt Len Mean"),
        "long_std": numerico("Pkt Len Std"),
        "tasa_paquetes": numerico("Flow Pkts/s"),
    })
    comun = comun.replace([np.inf, -np.inf], np.nan)
    return comun.join(datos[["etiqueta", "Label"]])
