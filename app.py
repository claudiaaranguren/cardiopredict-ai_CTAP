"""
CardioPredict AI - Tamizaje de riesgo de cardiopatía
Proyecto ACA 1 - Aprendizaje Automático (CUN)

Arquitectura:
  - modelo_cardiaco.pkl   : pipeline completo (imputación + escalado + One-Hot + Regresión Logística)
  - metricas_modelo.json  : métricas, umbral, curva ROC, matriz de confusión y coeficientes
                            generados por el notebook. La app NO contiene números escritos a mano.
"""

import io
import json
import os

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

# ------------------------------------------------------------------------------
# ENLACES DEL PROYECTO: pegue aquí cada enlace cuando lo tenga (entre comillas).
# Si un enlace queda vacío (""), la app muestra "Disponible pronto".
# ------------------------------------------------------------------------------
ENLACES = {
    "📓 Notebook (Google Colab)": "https://colab.research.google.com/drive/1Xge3m_tg7hq6MR-TQzmT5tycTc0Kv-MM?usp=sharing",
    "💻 Repositorio (GitHub)": "https://github.com/claudiaaranguren/cardiopredict-ai_CTAP",
    "🎥 Video de sustentación": "https://youtu.be/FbgWG-W6Ia4",
    "📄 Póster (infografía PDF)": "https://github.com/claudiaaranguren/cardiopredict-ai_CTAP/blob/main/P%C3%B3ster_ACA_1_A3.pdf",
}
ARCHIVO_POSTER = "infografia.png"   # imagen del póster en el repositorio (opcional)

# Paleta sobria de salud: azules y verde azulado clínico; coral solo para alertas.
MARINO = "#0B3C5D"
AZUL = "#1B6CA8"
TEAL = "#0B7A6B"
VERDE = "#26784A"
CORAL = "#C53030"
AMBAR = "#9C5D06"
PURPURA = "#5B4B8A"
COLOR_RIESGO = CORAL
COLOR_SANO = VERDE
COLOR_PRINCIPAL = AZUL

ESTILOS = f"""
<style>
html {{ font-size: 18px; }}
.tarjeta {{ border-radius: 16px; padding: 18px 14px; color: #FFFFFF; text-align: center;
           box-shadow: 0 6px 16px rgba(11, 60, 93, 0.22); margin-bottom: 12px; }}
.tarjeta .etq {{ font-size: 0.95rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }}
.tarjeta .val {{ font-size: 2.3rem; font-weight: 800; line-height: 1.15; margin: 6px 0 4px; }}
.tarjeta .det {{ font-size: 0.95rem; }}
.titulo {{ background: linear-gradient(90deg, {MARINO}, {AZUL}); color: #FFFFFF; padding: 12px 20px;
          border-radius: 12px; border-left: 10px solid {TEAL}; font-size: 1.45rem; font-weight: 800;
          margin: 26px 0 16px; }}
.banner {{ border-radius: 16px; padding: 20px; color: #FFFFFF; font-size: 1.45rem; font-weight: 800;
          text-align: center; margin-bottom: 14px; box-shadow: 0 6px 16px rgba(0, 0, 0, 0.18); }}
.enlace {{ display: block; text-align: center; padding: 14px 10px; border-radius: 14px; color: #FFFFFF !important;
          text-decoration: none !important; font-weight: 700; font-size: 1.05rem;
          box-shadow: 0 4px 12px rgba(11, 60, 93, 0.2); }}
.enlace:hover {{ filter: brightness(1.12); }}
.enlace.inactivo {{ background: #66757F; cursor: default; }}
.aviso {{ background: #FFF4E0; border-left: 8px solid {AMBAR}; color: #4A3200; padding: 14px 18px;
         border-radius: 10px; font-size: 1.05rem; }}
button[kind="primary"] {{ background: {TEAL} !important; border-color: {TEAL} !important;
                          font-size: 1.15rem !important; font-weight: 700 !important; }}
.stTabs [data-baseweb="tab"] p {{ font-size: 1.1rem; font-weight: 700; }}
</style>
"""


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


def nota(texto):
    """Texto explicativo en tamaño y color normales (más legible que st.caption)."""
    st.markdown(texto)


def tabla_md(encabezados, filas):
    """Tabla en markdown: hereda el color de texto del tema y se lee con claridad."""
    lineas = ["| " + " | ".join(encabezados) + " |",
              "|" + "|".join([":---"] + [":---:"] * (len(encabezados) - 1)) + "|"]
    lineas += ["| " + " | ".join(str(c) for c in fila) + " |" for fila in filas]
    st.markdown("\n".join(lineas))


def tarjeta(etiqueta, valor, color, detalle=""):
    """Tarjeta de color intenso con número grande en blanco."""
    det = f'<div class="det">{detalle}</div>' if detalle else ""
    st.markdown(f'<div class="tarjeta" style="background:{color};"><div class="etq">{etiqueta}</div>'
                f'<div class="val">{valor}</div>{det}</div>', unsafe_allow_html=True)


def titulo_seccion(texto):
    """Título de sección en franja azul con borde verde azulado."""
    st.markdown(f'<div class="titulo">{texto}</div>', unsafe_allow_html=True)


plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5})


def imagen_figura(fig, ancho):
    """Convierte la figura en imagen de ancho fijo (px): así la letra del gráfico
    queda proporcional al texto de la app y no crece con el ancho de la pantalla."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    st.image(buffer, width=ancho)


def mostrar_figura(fig, ancho=620):
    """Muestra un gráfico centrado y de ancho moderado."""
    _, centro, _ = st.columns([1, 3, 1])
    with centro:
        imagen_figura(fig, ancho)


# Nombres legibles para los coeficientes (códigos del dataset UCI → español)
NOMBRES_VARIABLES = {
    "sexo": "Sexo masculino", "edad": "Edad", "presion_reposo": "Presión en reposo",
    "colesterol": "Colesterol", "frecuencia_max": "Frecuencia cardíaca máx.",
    "depresion_st": "Depresión del ST", "azucar_ayunas": "Glucemia > 120 mg/dl",
    "angina_ejercicio": "Angina con ejercicio", "num_vasos_principales": "N.º de vasos principales",
    "tipo_dolor_pecho_1.0": "Dolor: angina típica", "tipo_dolor_pecho_2.0": "Dolor: angina atípica",
    "tipo_dolor_pecho_3.0": "Dolor: no anginoso", "tipo_dolor_pecho_4.0": "Dolor: asintomático",
    "electro_reposo_0.0": "ECG: normal", "electro_reposo_1.0": "ECG: anomalía ST-T",
    "electro_reposo_2.0": "ECG: hipertrofia ventricular",
    "pendiente_st_1.0": "Pendiente ST ascendente", "pendiente_st_2.0": "Pendiente ST plana",
    "pendiente_st_3.0": "Pendiente ST descendente",
    "talasemia_3.0": "Perfusión: normal", "talasemia_6.0": "Perfusión: defecto fijo",
    "talasemia_7.0": "Perfusión: defecto reversible",
}


def pct(valor):
    """Convierte una proporción (0-1) en texto de porcentaje."""
    return f"{valor * 100:.1f}%"


# ==============================================================================
# 3. BARRA LATERAL: DATOS DEL PACIENTE
# ==============================================================================
st.markdown(ESTILOS, unsafe_allow_html=True)
st.sidebar.header("📋 Parámetros Clínicos del Paciente")
st.sidebar.markdown("Ingrese los valores observados en la consulta.")


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
  <h1 style="color:white; margin-bottom:0; font-size:2.8rem;">🩺 CardioPredict AI</h1>
  <h3 style="color:white; margin-top:4px;">Tamizaje de riesgo de cardiopatía con Machine Learning clásico</h3>
  <p style="font-size:1.05rem; margin-bottom:0; line-height:1.6;">
    Especialización en Inteligencia Artificial – CUN · Proyecto ACA 1 – Aprendizaje Automático<br>
    <b>Curso:</b> APRENDIZAJE AUTOMÁTICO/54ES2/SEGUNDO BLOQUE/26ES4<br>
    <b>Autora:</b> Claudia Teresa Aranguren Peña<br>
    <b>Docente:</b> Dr. Hamilton Rivera Flor
  </p>
</div>
""", unsafe_allow_html=True)

st.write("")
st.markdown(
    '<div class="aviso"><b>⚕️ Herramienta académica de tamizaje.</b> Estima el riesgo para priorizar '
    'exámenes confirmatorios. <b>No reemplaza el diagnóstico de un profesional de la salud.</b></div>',
    unsafe_allow_html=True,
)

titulo_seccion("📌 Acceso al proyecto")
colores_enlaces = [AZUL, MARINO, TEAL, PURPURA]
columnas_enlaces = st.columns(len(ENLACES))
for columna, color, (nombre, url) in zip(columnas_enlaces, colores_enlaces, ENLACES.items()):
    with columna:
        if url:
            st.markdown(f'<a class="enlace" style="background:{color};" href="{url}" target="_blank">'
                        f'{nombre}</a>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="enlace inactivo">{nombre}<br>'
                        f'<small>Disponible pronto</small></span>', unsafe_allow_html=True)

if os.path.exists(ARCHIVO_POSTER):
    with st.expander("🖼️ Ver póster del proyecto"):
        st.image(ARCHIVO_POSTER)

st.divider()

# ==============================================================================
# 5. EVALUACIÓN DEL PACIENTE
# ==============================================================================
titulo_seccion("🔍 Evaluación del paciente")

col_datos, col_resultado = st.columns([1.2, 1])

with col_datos:
    st.markdown("**Mediciones ingresadas** (panel lateral)")
    d = datos_paciente.iloc[0]
    si_no = {0.0: "No", 1.0: "Sí"}
    tabla_md(["Variable", "Valor"], [
        ["Edad", f"{d['edad']:g} años"],
        ["Sexo", "Masculino" if d["sexo"] == 1.0 else "Femenino"],
        ["Tipo de dolor torácico", {1.0: "Angina típica", 2.0: "Angina atípica",
                                    3.0: "Dolor no anginoso", 4.0: "Asintomático"}[d["tipo_dolor_pecho"]]],
        ["Presión arterial en reposo", f"{d['presion_reposo']:g} mm Hg"],
        ["Colesterol sérico", f"{d['colesterol']:g} mg/dl"],
        ["Glucemia en ayunas > 120 mg/dl", si_no[d["azucar_ayunas"]]],
        ["Electrocardiograma en reposo", {0.0: "Normal", 1.0: "Anomalía ST-T",
                                          2.0: "Hipertrofia ventricular izq."}[d["electro_reposo"]]],
        ["Frecuencia cardíaca máxima", f"{d['frecuencia_max']:g} lpm"],
        ["Angina inducida por ejercicio", si_no[d["angina_ejercicio"]]],
        ["Depresión del segmento ST", f"{d['depresion_st']:g}"],
        ["Pendiente del segmento ST", {1.0: "Ascendente", 2.0: "Plana",
                                       3.0: "Descendente"}[d["pendiente_st"]]],
        ["Vasos principales (fluoroscopia)", f"{d['num_vasos_principales']:g}"],
        ["Prueba de perfusión (talasemia)", {3.0: "Normal", 6.0: "Defecto fijo",
                                             7.0: "Defecto reversible"}[d["talasemia"]]],
    ])
    analizar = st.button("Analizar riesgo cardíaco", type="primary")

with col_resultado:
    if analizar:
        # El pipeline aplica imputación, escalado y One-Hot antes de predecir
        probabilidad = float(modelo.predict_proba(datos_paciente)[0, 1])
        positivo = probabilidad >= UMBRAL

        if positivo:
            st.markdown(f'<div class="banner" style="background:{CORAL};">⚠️ Tamizaje POSITIVO<br>'
                        f'<span style="font-size:1.1rem;">Remitir a exámenes confirmatorios</span></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="banner" style="background:{VERDE};">✅ Tamizaje NEGATIVO<br>'
                        f'<span style="font-size:1.1rem;">Riesgo bajo según el modelo</span></div>',
                        unsafe_allow_html=True)

        tarjeta("Probabilidad estimada de cardiopatía", pct(probabilidad),
                CORAL if positivo else VERDE, f"Umbral de decisión: {pct(UMBRAL)}")
        st.progress(min(max(probabilidad, 0.0), 1.0))
        nota(
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
                "⚠️ Algunos valores están fuera del rango observado en el entrenamiento; "
                "la predicción es menos confiable:\n\n- " + "\n- ".join(avisos)
            )
    else:
        st.info("👈 Ajuste los datos en el panel lateral y pulse **Analizar riesgo cardíaco**.")

st.divider()

# ==============================================================================
# 6. CONOCE LOS DATOS (EDA resumido: solo las gráficas que explican el dataset)
# ==============================================================================
@st.cache_data
def cargar_datos():
    """Lee el dataset original y traduce las categorías para las gráficas descriptivas."""
    df = pd.read_csv("processed.cleveland.data", names=COLUMNAS, na_values="?")
    df["Diagnóstico"] = (df["objetivo"] > 0).map({False: "Sano", True: "Cardiopatía"})
    df["Tipo de dolor"] = df["tipo_dolor_pecho"].map(
        {1: "Angina típica", 2: "Angina atípica", 3: "No anginoso", 4: "Asintomático"})
    return df


titulo_seccion("📊 Conoce los datos")
datos = cargar_datos()
nota(
    f"UCI Heart Disease (Cleveland): **{len(datos)} pacientes** y 13 variables clínicas. "
    "Estas gráficas describen el dataset completo; el modelo se entrenó solo con el 80 % de los pacientes."
)
g1, g2 = st.columns(2)
with g1:
    conteo = datos["Diagnóstico"].value_counts().reindex(["Sano", "Cardiopatía"])
    fig, ax = plt.subplots(figsize=(5, 3.4))
    barras = ax.bar(conteo.index, conteo.values, color=[COLOR_PRINCIPAL, COLOR_RIESGO])
    ax.bar_label(barras, labels=[f"{v} ({v / len(datos):.0%})" for v in conteo.values])
    ax.set_ylabel("Pacientes")
    ax.set_title("Diagnóstico: clases casi balanceadas")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, conteo.max() * 1.18)
    imagen_figura(fig, 480)
with g2:
    orden = ["Angina típica", "Angina atípica", "No anginoso", "Asintomático"]
    tasa = (datos.groupby("Tipo de dolor")["Diagnóstico"].apply(lambda s: (s == "Cardiopatía").mean())
            .reindex(orden))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    barras = ax.barh(tasa.index, tasa.values * 100,
                     color=[COLOR_RIESGO if v > 0.5 else COLOR_PRINCIPAL for v in tasa.values])
    ax.bar_label(barras, labels=[f"{v:.0%}" for v in tasa.values], padding=3)
    ax.set_xlabel("% de pacientes con cardiopatía")
    ax.set_xlim(0, 100)
    ax.set_title("Cardiopatía según el tipo de dolor de pecho")
    ax.spines[["top", "right"]].set_visible(False)
    imagen_figura(fig, 480)
nota(
    "Las clases están casi balanceadas, por eso la exactitud es interpretable, aunque en salud se "
    "prioriza el Recall. El dolor **asintomático** concentra la mayor proporción de enfermos: "
    "un hallazgo que el modelo también aprendió (ver coeficientes)."
)

# ==============================================================================
# 7. DESEMPEÑO DEL MODELO (todo se lee del JSON)
# ==============================================================================
titulo_seccion("📈 Desempeño del modelo")
nota(
    f"Modelo: **{evidencias['modelo']}** (C = {evidencias['hiperparametros']['C']}) · "
    f"Entrenamiento: {evidencias['n_entrenamiento']} pacientes · "
    f"Prueba: {evidencias['n_prueba']} pacientes"
)

st.markdown("**Validación cruzada (5 particiones, umbral 0.5)** — estimación más representativa")
c1, c2, c3, c4 = st.columns(4)
for col, nombre, etiqueta, color in [(c1, "roc_auc", "ROC-AUC", MARINO), (c2, "accuracy", "Accuracy", AZUL),
                                    (c3, "recall", "Recall", TEAL), (c4, "precision", "Precision", PURPURA)]:
    with col:
        tarjeta(etiqueta, f"{CV[nombre]['media']:.3f}", color, f"± {CV[nombre]['desviacion']:.3f}")

st.markdown(f"**Conjunto de prueba (umbral {UMBRAL})** — evaluación única sobre pacientes no vistos")
t1, t2, t3, t4 = st.columns(4)
for col, clave, etiqueta, color in [(t1, "Recall (sensibilidad)", "Recall", TEAL),
                                   (t2, "Especificidad", "Especificidad", AZUL),
                                   (t3, "Precision", "Precision", PURPURA), (t4, "ROC-AUC", "ROC-AUC", MARINO)]:
    with col:
        tarjeta(etiqueta, f"{TEST[clave]:.3f}", color)
nota(
    "Con solo 28 pacientes enfermos en la prueba, un Recall de 1.000 tiene un intervalo de "
    "confianza al 95% aproximado de 0.88 a 1.00: no implica que el modelo nunca falle."
)

tab_coef, tab_roc, tab_matriz, tab_comp = st.tabs(
    ["🧠 Coeficientes", "📉 Curva ROC", "🎯 Matriz de confusión", "⚖️ Comparación de modelos"]
)

with tab_coef:
    coef = pd.Series(evidencias["coeficientes"]).sort_values()
    coef.index = [NOMBRES_VARIABLES.get(n, n) for n in coef.index]
    fig, ax = plt.subplots(figsize=(7, 6.2))
    ax.barh(coef.index, coef.values,
            color=[COLOR_RIESGO if v > 0 else COLOR_SANO for v in coef.values])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(coef.min() * 1.12, coef.max() * 1.12)
    ax.set_xlabel("Coeficiente (log-odds)   ← reduce el riesgo | aumenta el riesgo →")
    ax.set_title("Coeficientes de la Regresión Logística")
    ax.spines[["top", "right"]].set_visible(False)
    mostrar_figura(fig, 640)
    nota(
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
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.4)
    mostrar_figura(fig, 560)

with tab_matriz:
    vn, fp = evidencias["matriz_confusion"][0]
    fn, vp = evidencias["matriz_confusion"][1]
    tabla_md(["", "Predicho: Sano", "Predicho: Cardiopatía"], [
        ["**Real: Sano**", vn, fp],
        ["**Real: Cardiopatía**", fn, vp],
    ])
    m1, m2, m3, m4 = st.columns(4)
    for col, etiqueta, valor, color in [(m1, "Verdaderos negativos", vn, AZUL), (m2, "Falsos positivos", fp, AMBAR),
                                        (m3, "Falsos negativos", fn, CORAL), (m4, "Verdaderos positivos", vp, VERDE)]:
        with col:
            tarjeta(etiqueta, valor, color)
    nota(f"Conjunto de prueba ({evidencias['n_prueba']} pacientes), umbral {UMBRAL}.")

with tab_comp:
    ranking = sorted(evidencias["comparacion_modelos_cv"].items(), key=lambda kv: -kv[1])
    tabla_md(["Modelo", "ROC-AUC (validación cruzada)"], [
        [f"**{nombre}** ✅ seleccionado" if nombre == evidencias["modelo"] else nombre, f"{valor:.3f}"]
        for nombre, valor in ranking
    ])
    nota(
        "Criterio de selección definido antes de experimentar: mayor ROC-AUC medio en validación "
        "cruzada. Las diferencias son pequeñas; ante el empate se prefirió el modelo más simple e "
        "interpretable, que además mostró el menor sobreajuste."
    )

# ==============================================================================
# 8. INFORMACIÓN METODOLÓGICA
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
