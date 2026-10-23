"""
CardioPredict AI - Tamizaje de riesgo de cardiopatía
Proyecto ACA 1 - Aprendizaje Automático (CUN)

Arquitectura:
  - modelo_cardiaco.pkl   : pipeline completo (imputación + escalado + One-Hot + Regresión Logística)
  - metricas_modelo.json  : métricas, umbral, curva ROC, matriz de confusión y coeficientes
                            generados por el notebook. La app NO contiene números escritos a mano.
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="CardioPredict AI | Tamizaje de Riesgo Cardíaco",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_RIESGO = "#e74c3c"
COLOR_SANO = "#2ecc71"
COLOR_PRINCIPAL = "#1B5E8C"


# ==============================================================================
# 2. CARGA DE ARTEFACTOS (en caché: se leen una sola vez)
# ==============================================================================
COLUMNAS = [
    "edad", "sexo", "tipo_dolor_pecho", "presion_reposo", "colesterol",
    "azucar_ayunas", "electro_reposo", "frecuencia_max", "angina_ejercicio",
    "depresion_st", "pendiente_st", "num_vasos_principales", "talasemia", "objetivo",
]


def reentrenar_pipeline(hiperparametros):
    """
    Reconstruye el pipeline con EXACTAMENTE la misma configuración del notebook
    (misma partición, preprocesamiento e hiperparámetros, random_state=42).
    Se usa solo si el .pkl no puede abrirse porque el servidor tiene una versión
    de scikit-learn distinta a la de Colab (los .pkl no son portables entre versiones).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv("processed.cleveland.data", names=COLUMNAS, na_values="?")
    df["objetivo"] = (df["objetivo"] > 0).astype(int)
    X, y = df.drop("objetivo", axis=1), df["objetivo"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    continuas = ["edad", "presion_reposo", "colesterol", "frecuencia_max", "depresion_st"]
    ordinales = ["num_vasos_principales"]
    binarias = ["sexo", "azucar_ayunas", "angina_ejercicio"]
    nominales = ["tipo_dolor_pecho", "electro_reposo", "pendiente_st", "talasemia"]

    preprocesador = ColumnTransformer(transformers=[
        ("cont", Pipeline([("imputar", SimpleImputer(strategy="median")),
                           ("escalar", StandardScaler())]), continuas),
        ("ord", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          ("escalar", StandardScaler())]), ordinales),
        ("bin", "passthrough", binarias),
        ("nom", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), nominales),
    ])
    pipeline = Pipeline([
        ("prep", preprocesador),
        ("clf", LogisticRegression(C=hiperparametros["C"], max_iter=1000, random_state=42)),
    ])
    return pipeline.fit(X_train, y_train)


def coeficientes_del_pipeline(pipeline):
    """Devuelve {característica: coeficiente} con los mismos nombres del notebook."""
    nombres = [n.split("__", 1)[1] for n in
               pipeline.named_steps["prep"].get_feature_names_out().astype(str)]
    return dict(zip(nombres, pipeline.named_steps["clf"].coef_[0]))


@st.cache_resource
def cargar_modelo(hiperparametros, coeficientes_notebook):
    """
    1) Intenta abrir el pipeline exportado desde Colab (modelo_cardiaco.pkl).
    2) Si la versión de scikit-learn del servidor no permite abrirlo, lo reconstruye
       con la misma configuración y VERIFICA que sus coeficientes coincidan con los
       del notebook (metricas_modelo.json). Si no coinciden, no se usa.
    Devuelve (modelo, origen) o (None, mensaje_de_error).
    """
    try:
        return joblib.load("modelo_cardiaco.pkl"), "pkl"
    except Exception:
        pass

    try:
        modelo = reentrenar_pipeline(hiperparametros)
    except Exception as error:
        return None, f"No fue posible reconstruir el modelo: {error}"

    coef_app = coeficientes_del_pipeline(modelo)
    iguales = all(abs(coef_app[k] - v) < 1e-3 for k, v in coeficientes_notebook.items())
    if not iguales:
        return None, "El modelo reconstruido no coincide con el del notebook."
    return modelo, "reconstruido"


@st.cache_data
def cargar_metricas():
    """Carga las evidencias exportadas por el notebook. Devuelve None si no existe."""
    try:
        with open("metricas_modelo.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


evidencias = cargar_metricas()
if evidencias is None:
    st.error("⚠️ No se encontró 'metricas_modelo.json' en la misma carpeta que 'app.py'.")
    st.stop()

modelo, ORIGEN_MODELO = cargar_modelo(evidencias["hiperparametros"], evidencias["coeficientes"])
if modelo is None:
    st.error(f"⚠️ {ORIGEN_MODELO}")
    st.stop()

UMBRAL = evidencias["umbral"]
CV = evidencias["metricas_cv"]
TEST = evidencias["metricas_test"]
RANGOS = evidencias["rangos_entrenamiento"]


def mostrar_figura(fig):
    """Muestra un gráfico centrado y con ancho moderado (evita que ocupe toda la pantalla)."""
    _, centro, _ = st.columns([1, 2, 1])
    with centro:
        st.pyplot(fig)
    plt.close(fig)


def pct(valor):
    """Convierte una proporción (0-1) en texto de porcentaje."""
    return f"{valor * 100:.1f}%"


# ==============================================================================
# 3. BARRA LATERAL: DATOS DEL PACIENTE
# ==============================================================================
st.sidebar.header("📋 Parámetros Clínicos del Paciente")
st.sidebar.caption("Ingrese los valores observados en la consulta.")


def recopilar_entradas():
    st.sidebar.subheader("Datos generales")
    edad = st.sidebar.number_input("Edad (años)", min_value=18, max_value=100, value=55, step=1)
    sexo = st.sidebar.selectbox(
        "Sexo biológico", options=[1.0, 0.0],
        format_func=lambda x: "Masculino" if x == 1.0 else "Femenino",
    )

    st.sidebar.subheader("Pruebas cardiovasculares")
    tipo_dolor_pecho = st.sidebar.selectbox(
        "Tipo de dolor torácico", options=[1.0, 2.0, 3.0, 4.0],
        format_func=lambda x: {1.0: "1: Angina típica", 2.0: "2: Angina atípica",
                               3.0: "3: Dolor no anginoso", 4.0: "4: Asintomático"}[x],
    )
    presion_reposo = st.sidebar.number_input(
        "Presión arterial en reposo (mm Hg)", min_value=70, max_value=250, value=130, step=1)
    colesterol = st.sidebar.number_input(
        "Colesterol sérico (mg/dl)", min_value=80, max_value=700, value=240, step=1)
    azucar_ayunas = st.sidebar.selectbox(
        "Glucemia en ayunas > 120 mg/dl", options=[0.0, 1.0],
        format_func=lambda x: "Sí (> 120 mg/dl)" if x == 1.0 else "No",
    )
    electro_reposo = st.sidebar.selectbox(
        "Electrocardiograma en reposo", options=[0.0, 1.0, 2.0],
        format_func=lambda x: {0.0: "0: Normal", 1.0: "1: Anomalía de onda ST-T",
                               2.0: "2: Hipertrofia ventricular izquierda"}[x],
    )

    st.sidebar.subheader("Prueba de esfuerzo")
    frecuencia_max = st.sidebar.number_input(
        "Frecuencia cardíaca máxima alcanzada (lpm)", min_value=50, max_value=230, value=150, step=1)
    angina_ejercicio = st.sidebar.selectbox(
        "Angina inducida por ejercicio", options=[0.0, 1.0],
        format_func=lambda x: "Sí" if x == 1.0 else "No",
    )
    depresion_st = st.sidebar.number_input(
        "Depresión del segmento ST", min_value=0.0, max_value=8.0, value=1.0, step=0.1)
    pendiente_st = st.sidebar.selectbox(
        "Pendiente del segmento ST", options=[1.0, 2.0, 3.0],
        format_func=lambda x: {1.0: "1: Ascendente", 2.0: "2: Plana", 3.0: "3: Descendente"}[x],
    )

    st.sidebar.subheader("Estudios avanzados")
    num_vasos_principales = st.sidebar.selectbox(
        "Vasos principales coloreados por fluoroscopia", options=[0.0, 1.0, 2.0, 3.0],
        format_func=lambda x: f"{int(x)}",
    )
    talasemia = st.sidebar.selectbox(
        "Prueba de perfusión (talasemia)", options=[3.0, 6.0, 7.0],
        format_func=lambda x: {3.0: "Normal", 6.0: "Defecto fijo", 7.0: "Defecto reversible"}[x],
    )

    # Mismo nombre y orden de columnas que en el entrenamiento
    return pd.DataFrame([{
        "edad": float(edad), "sexo": sexo, "tipo_dolor_pecho": tipo_dolor_pecho,
        "presion_reposo": float(presion_reposo), "colesterol": float(colesterol),
        "azucar_ayunas": azucar_ayunas, "electro_reposo": electro_reposo,
        "frecuencia_max": float(frecuencia_max), "angina_ejercicio": angina_ejercicio,
        "depresion_st": float(depresion_st), "pendiente_st": pendiente_st,
        "num_vasos_principales": num_vasos_principales, "talasemia": talasemia,
    }])


datos_paciente = recopilar_entradas()


def valores_fuera_de_rango(datos):
    """Lista las variables continuas fuera del rango visto durante el entrenamiento."""
    avisos = []
    for variable, (minimo, maximo) in RANGOS.items():
        valor = datos[variable].iloc[0]
        if valor < minimo or valor > maximo:
            avisos.append(f"**{variable}** = {valor:g} (rango de entrenamiento: {minimo:g} a {maximo:g})")
    return avisos


# ==============================================================================
# 4. ENCABEZADO
# ==============================================================================
st.markdown(f"""
<div style="background: linear-gradient(90deg,#0B3C5D,{COLOR_PRINCIPAL});
            padding:25px; border-radius:12px; color:white; text-align:center;">
  <h1 style="color:white; margin-bottom:0;">🩺 CardioPredict AI</h1>
  <h3 style="color:white; margin-top:4px;">Tamizaje de riesgo de cardiopatía con Machine Learning clásico</h3>
  <p style="font-size:16px; margin-bottom:0;">
    Especialización en Inteligencia Artificial – CUN · Proyecto ACA 1 – Aprendizaje Automático<br>
    <b>Integrantes:</b> Claudia Teresa Aranguren Peña · Juan Sebastián Jaramillo ·
    Weimar Rolando Ramírez · Alexandra Rippe Abril · Jayr Antonio Fandiño<br>
    <b>Docente:</b> Dr. Hamilton Rivera Flor
  </p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.warning(
    "**Herramienta académica de tamizaje.** Estima el riesgo para priorizar exámenes "
    "confirmatorios. **No reemplaza el diagnóstico de un profesional de la salud.**"
)

# ==============================================================================
# 5. EVALUACIÓN DEL PACIENTE
# ==============================================================================
st.subheader("🔍 Evaluación del paciente")

col_datos, col_resultado = st.columns([1.2, 1])

with col_datos:
    st.markdown("**Mediciones ingresadas** (panel lateral)")
    st.dataframe(datos_paciente.T.rename(columns={0: "Valor"}))
    analizar = st.button("Analizar riesgo cardíaco", type="primary")

with col_resultado:
    if analizar:
        # El pipeline aplica imputación, escalado y One-Hot antes de predecir
        probabilidad = float(modelo.predict_proba(datos_paciente)[0, 1])
        positivo = probabilidad >= UMBRAL

        if positivo:
            st.error("### ⚠️ Tamizaje POSITIVO: remitir a exámenes confirmatorios")
        else:
            st.success("### ✅ Tamizaje NEGATIVO: riesgo bajo según el modelo")

        st.metric("Probabilidad estimada de cardiopatía", pct(probabilidad))
        st.progress(min(max(probabilidad, 0.0), 1.0))
        st.caption(
            f"Umbral de decisión: **{pct(UMBRAL)}**. Se eligió bajo (en lugar de 50%) para "
            f"priorizar la detección de enfermos: en validación cruzada detecta ≈"
            f"{pct(evidencias['umbral_cv']['recall'])} de los pacientes con cardiopatía, "
            f"a costa de ≈3 falsas alarmas por cada 10 alertas."
        )

        if positivo:
            st.markdown("**Recomendación:** valoración por cardiología y estudios de confirmación.")
        else:
            st.markdown(
                "**Recomendación:** control médico periódico. Un resultado negativo no descarta "
                "la enfermedad si persisten síntomas."
            )

        avisos = valores_fuera_de_rango(datos_paciente)
        if avisos:
            st.warning(
                "Algunos valores están fuera del rango observado en el entrenamiento; "
                "la predicción es menos confiable:\n\n- " + "\n- ".join(avisos)
            )
    else:
        st.info("👈 Ajuste los datos en el panel lateral y pulse **Analizar riesgo cardíaco**.")

st.divider()

# ==============================================================================
# 6. DESEMPEÑO DEL MODELO (todo se lee del JSON)
# ==============================================================================
st.subheader("📈 Desempeño del modelo")
st.caption(
    f"Modelo: **{evidencias['modelo']}** (C = {evidencias['hiperparametros']['C']}) · "
    f"Entrenamiento: {evidencias['n_entrenamiento']} pacientes · "
    f"Prueba: {evidencias['n_prueba']} pacientes"
)

st.markdown("**Validación cruzada (5 particiones, umbral 0.5)** — estimación más representativa")
c1, c2, c3, c4 = st.columns(4)
def media_desv(nombre):
    """Texto 'media ± desviación' sin flechas (no es una variación sino una dispersión)."""
    return f"{CV[nombre]['media']:.3f} ± {CV[nombre]['desviacion']:.3f}"


c1.metric("ROC-AUC", media_desv("roc_auc"))
c2.metric("Accuracy", media_desv("accuracy"))
c3.metric("Recall", media_desv("recall"))
c4.metric("Precision", media_desv("precision"))

st.markdown(f"**Conjunto de prueba (umbral {UMBRAL})** — evaluación única sobre pacientes no vistos")
t1, t2, t3, t4 = st.columns(4)
t1.metric("Recall (sensibilidad)", f"{TEST['Recall (sensibilidad)']:.3f}")
t2.metric("Especificidad", f"{TEST['Especificidad']:.3f}")
t3.metric("Precision", f"{TEST['Precision']:.3f}")
t4.metric("ROC-AUC", f"{TEST['ROC-AUC']:.3f}")
st.caption(
    "Con solo 28 pacientes enfermos en la prueba, un Recall de 1.000 tiene un intervalo de "
    "confianza al 95% aproximado de 0.88 a 1.00: no implica que el modelo nunca falle."
)

tab_coef, tab_roc, tab_matriz, tab_comp = st.tabs(
    ["🧠 Coeficientes", "📉 Curva ROC", "🎯 Matriz de confusión", "⚖️ Comparación de modelos"]
)

with tab_coef:
    coef = pd.Series(evidencias["coeficientes"]).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(coef.index, coef.values,
            color=[COLOR_RIESGO if v > 0 else COLOR_SANO for v in coef.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coeficiente (log-odds)   ← reduce el riesgo | aumenta el riesgo →")
    ax.set_title("Coeficientes de la Regresión Logística")
    fig.tight_layout()
    mostrar_figura(fig)
    st.caption(
        "Rojo: aumenta el riesgo; verde: lo reduce, manteniendo constantes las demás variables. "
        "Las variables continuas están escaladas (efecto por desviación estándar) y las categorías "
        "One-Hot se comparan con las demás categorías de su misma variable. Son asociaciones, no causas."
    )

with tab_roc:
    roc = evidencias["roc"]
    vn, fp = evidencias["matriz_confusion"][0]
    fn, vp = evidencias["matriz_confusion"][1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(roc["fpr"], roc["tpr"], linewidth=2, color=COLOR_PRINCIPAL,
            label=f"Regresión Logística (AUC = {roc['auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Clasificador aleatorio")
    ax.scatter(fp / (fp + vn), vp / (vp + fn), color=COLOR_RIESGO, s=60, zorder=3,
               label=f"Punto de operación (umbral {UMBRAL})")
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax.set_title("Curva ROC en el conjunto de prueba")
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    mostrar_figura(fig)

with tab_matriz:
    vn, fp = evidencias["matriz_confusion"][0]
    fn, vp = evidencias["matriz_confusion"][1]
    tabla = pd.DataFrame(
        [[vn, fp], [fn, vp]],
        index=["Real: Sano", "Real: Cardiopatía"],
        columns=["Predicho: Sano", "Predicho: Cardiopatía"],
    )
    st.table(tabla)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verdaderos negativos", vn)
    m2.metric("Falsos positivos", fp)
    m3.metric("Falsos negativos", fn)
    m4.metric("Verdaderos positivos", vp)
    st.caption(f"Conjunto de prueba ({evidencias['n_prueba']} pacientes), umbral {UMBRAL}.")

with tab_comp:
    comparacion = (pd.Series(evidencias["comparacion_modelos_cv"], name="ROC-AUC (validación cruzada)")
                   .sort_values(ascending=False).map(lambda v: f"{v:.3f}"))
    st.table(comparacion.to_frame())
    st.caption(
        "Criterio de selección definido antes de experimentar: mayor ROC-AUC medio en validación "
        "cruzada. Las diferencias son pequeñas; ante el empate se prefirió el modelo más simple e "
        "interpretable, que además mostró el menor sobreajuste."
    )

# ==============================================================================
# 7. INFORMACIÓN METODOLÓGICA
# ==============================================================================
st.divider()
with st.expander("ℹ️ Metodología, dataset y limitaciones"):
    st.markdown(f"""
**Dataset:** UCI Heart Disease (Cleveland), 303 pacientes, 13 biomarcadores clínicos.
Objetivo binario: 0 = sano, 1 = cardiopatía (54.1% / 45.9%).

**Flujo:** partición estratificada 80/20 → pipeline de preprocesamiento (imputación por
mediana/moda, escalado estándar y One-Hot Encoding de variables nominales) → comparación de
Regresión Logística, Random Forest y KNN con validación cruzada y `GridSearchCV` → ajuste del
umbral de decisión con validación cruzada (Recall objetivo ≥ 0.90) → evaluación única en el
conjunto de prueba.

**Limitaciones:** muestra pequeña y de un solo centro, lo que limita la generalización. Las
relaciones encontradas son asociaciones, no causas. Herramienta académica: no reemplaza el
criterio médico.

**Versión de scikit-learn del modelo:** {evidencias['version_sklearn']}.
{"Modelo cargado desde `modelo_cardiaco.pkl`." if ORIGEN_MODELO == "pkl" else
 "El servidor usa otra versión de scikit-learn, por lo que el modelo se reconstruyó con la misma "
 "configuración del notebook y se verificó que sus coeficientes coinciden con los exportados."}

**Referencia del dataset:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988).
*Heart disease* [Conjunto de datos]. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X
""")
