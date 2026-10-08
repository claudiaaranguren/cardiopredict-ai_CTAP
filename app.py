import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="CardioPredict AI | Asistencia Diagnóstica",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 2. CARGA DE ARTEFACTOS SERIALIZADOS
# ==============================================================================
@st.cache_resource
def cargar_artefactos():
    """Carga en caché el modelo y el escalador preentrenados."""
    try:
        modelo = joblib.load('modelo_cardiaco.pkl')
        scaler = joblib.load('scaler_cardiaco.pkl')
        return modelo, scaler
    except FileNotFoundError:
        return None, None

modelo, scaler = cargar_artefactos()

# ==============================================================================
# 3. BARRA LATERAL: ENTRADA DE DATOS DEL PACIENTE
# ==============================================================================
st.sidebar.header("📋 Parámetros Clínicos del Paciente")
st.sidebar.markdown("Ingrese los valores observados en la consulta médica:")

def recopilar_entradas():
    st.sidebar.subheader("Biomarcadores Generales")
    edad = st.sidebar.number_input("Edad (años)", min_value=18, max_value=100, value=55, step=1)
    sexo = st.sidebar.selectbox("Sexo Biológico", options=[1, 0], format_func=lambda x: "Masculino" if x == 1 else "Femenino")
    
    st.sidebar.subheader("Pruebas Cardiovasculares")
    tipo_dolor_pecho = st.sidebar.selectbox(
        "Tipo de Dolor Torácico",
        options=[1.0, 2.0, 3.0, 4.0],
        format_func=lambda x: {
            1.0: "1: Angina típica",
            2.0: "2: Angina atípica",
            3.0: "3: Dolor no anginoso",
            4.0: "4: Asintomático"
        }[x]
    )
    presion_reposo = st.sidebar.slider("Presión Arterial en Reposo (mm Hg)", min_value=80, max_value=210, value=130)
    colesterol = st.sidebar.slider("Colesterol Sérico (mg/dl)", min_value=100, max_value=600, value=240)
    azucar_ayunas = st.sidebar.selectbox(
        "Glucemia en Ayunas > 120 mg/dl",
        options=[0.0, 1.0],
        format_func=lambda x: "Sí (> 120 mg/dl)" if x == 1.0 else "No (Normal)"
    )
    electro_reposo = st.sidebar.selectbox(
        "Electrocardiograma en Reposo",
        options=[0.0, 1.0, 2.0],
        format_func=lambda x: {
            0.0: "0: Normal",
            1.0: "1: Anomalía de onda ST-T",
            2.0: "2: Hipertrofia ventricular izquierda"
        }[x]
    )

    st.sidebar.subheader("Prueba de Esfuerzo Físico")
    frecuencia_max = st.sidebar.slider("Frecuencia Cardíaca Máxima Alcanzada", min_value=60, max_value=220, value=150)
    angina_ejercicio = st.sidebar.selectbox(
        "Angina Inducida por Ejercicio",
        options=[0.0, 1.0],
        format_func=lambda x: "Sí" if x == 1.0 else "No"
    )
    depresion_st = st.sidebar.slider("Depresión del Segmento ST (Prueba de Esfuerzo)", min_value=0.0, max_value=7.0, value=1.0, step=0.1)
    pendiente_st = st.sidebar.selectbox(
        "Pendiente del Segmento ST en Esfuerzo Máximo",
        options=[1.0, 2.0, 3.0],
        format_func=lambda x: {
            1.0: "1: Ascendente",
            2.0: "2: Plana",
            3.0: "3: Descendente"
        }[x]
    )

    st.sidebar.subheader("Estudios Avanzados")
    num_vasos_principales = st.sidebar.selectbox("Vasos Principales Coloreados por Fluoroscopia", options=[0.0, 1.0, 2.0, 3.0])
    talasemia = st.sidebar.selectbox(
        "Evaluación de Perfusión / Talasemia",
        options=[3.0, 6.0, 7.0],
        format_func=lambda x: {
            3.0: "3.0: Normal",
            6.0: "6.0: Defecto fijo",
            7.0: "7.0: Defecto reversible"
        }[x]
    )

    # Estructura del vector con el mismo orden del entrenamiento
    columnas = [
        'edad', 'sexo', 'tipo_dolor_pecho', 'presion_reposo', 'colesterol',
        'azucar_ayunas', 'electro_reposo', 'frecuencia_max', 'angina_ejercicio',
        'depresion_st', 'pendiente_st', 'num_vasos_principales', 'talasemia'
    ]
    
    valores = [
        edad, sexo, tipo_dolor_pecho, presion_reposo, colesterol,
        azucar_ayunas, electro_reposo, frecuencia_max, angina_ejercicio,
        depresion_st, pendiente_st, num_vasos_principales, talasemia
    ]
    
    return pd.DataFrame([valores], columns=columnas)

datos_paciente = recopilar_entradas()

# ==============================================================================
# 4. CUERPO PRINCIPAL DE LA INTERFAZ
# ==============================================================================

st.markdown("""
<div style="
background: linear-gradient(90deg,#0B3C5D,#1B5E8C);
padding:25px;
border-radius:12px;
color:white;
text-align:center;
">

<h1>🩺 CardioPredict AI</h1>
<h3>Sistema Predictivo de Riesgo Cardíaco mediante Machine Learning</h3>
<p style="font-size:18px;">
Especialización en Inteligencia Artificial – CUN<br>
Proyecto ACA 1 - Aprendizaje Automático<br>

<b>Integrantes:</b><br>

Claudia Teresa Aranguren Peña | Juan Sebastián Jaramillo<br>
 
Weimar Rolando Ramírez | Alexandra Rippe Abril | Jayr Antonio Fandiño 

Docente: Dr. Hamilton Rivera Flor
</p>

</div>
""", unsafe_allow_html=True)

st.markdown("")

st.info("""
📚 Proyecto académico desarrollado utilizando el dataset UCI Heart Disease Cleveland.

✅ Modelo seleccionado: Random Forest

✅ Accuracy: 88.52%

✅ Recall: 96.43%

✅ ROC-AUC: 95.13%

✅ Despliegue: Streamlit Community Cloud
""")
# ==============================================================================
# INFORMACIÓN DEL DATASET
# ==============================================================================

with st.expander("📊 Información del Dataset"):

    st.markdown("""
### Dataset utilizado

**Fuente:** UCI Heart Disease Cleveland

**Total de registros:** 303 pacientes

**Variables predictoras:** 13 biomarcadores clínicos

**Variable objetivo**

- 0 = Sano
- 1 = Cardiopatía

**Distribución de clases**

- Pacientes sanos: 54.1%
- Pacientes con cardiopatía: 45.9%
""")

# ==============================================================================
# MODELO DE IA
# ==============================================================================

with st.expander("🤖 Modelo de Inteligencia Artificial"):

    st.markdown("""
### Algoritmo seleccionado

**Random Forest Classifier**

El modelo fue seleccionado después de comparar:

- Regresión Logística
- Random Forest
- K-Nearest Neighbors (KNN)

### Resultados obtenidos

✅ Accuracy: 88.52%

✅ Recall: 96.43%

✅ ROC-AUC: 95.13%

El modelo Random Forest presentó el mejor equilibrio general de desempeño para el problema de clasificación de cardiopatías.
""")

# ==============================================================================
# INTERPRETACIÓN DEL MODELO
# ==============================================================================

with st.expander("🧠 Interpretación del Modelo"):

    st.markdown("""
### Biomarcadores más importantes

1. Frecuencia cardíaca máxima

2. Tipo de dolor torácico

3. Talasemia

4. Número de vasos coronarios principales

5. Edad

Estas variables aportan la mayor cantidad de información para la clasificación de pacientes según el modelo Random Forest.
""")

# ==============================================================================
# ¿CÓMO FUNCIONA LA IA?
# ==============================================================================

with st.expander("⚙️ ¿Cómo funciona CardioPredict AI?"):

    st.markdown("""
### Flujo de Machine Learning

1️⃣ Se ingresan los biomarcadores clínicos del paciente.

2️⃣ Los datos son transformados mediante StandardScaler.

3️⃣ El modelo Random Forest analiza los patrones clínicos.

4️⃣ Se calcula la probabilidad de cardiopatía.

5️⃣ El sistema genera una clasificación.

6️⃣ Se presenta una recomendación orientativa al usuario.
""")

# ==============================================================================
# DASHBOARD DE DESEMPEÑO DEL MODELO
# ==============================================================================

st.subheader("📈 Dashboard de Desempeño del Modelo")

m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        label="Accuracy",
        value="88.52%"
    )

with m2:
    st.metric(
        label="Recall",
        value="96.43%"
    )

with m3:
    st.metric(
        label="ROC-AUC",
        value="95.13%"
    )

st.info("""
Este dashboard resume las métricas obtenidas por el modelo Random Forest
durante la fase de evaluación sobre el conjunto de prueba.
""")

st.subheader("📈 Centro Analítico del Modelo")
st.caption(
    "Explore las visualizaciones que respaldan el desempeño y la interpretabilidad del modelo Random Forest."
)

opcion = st.radio(
    "Seleccione una visualización",
    [
        "🔬 Importancia de Biomarcadores",
        "📉 Curva ROC",
        "🎯 Matriz de Confusión"
    ]
)

# ==============================================================================
# VISUALIZACIONES DEL CENTRO ANALÍTICO
# ==============================================================================

if opcion == "🔬 Importancia de Biomarcadores":

    st.markdown("### 🔬 Biomarcadores más influyentes según Random Forest")

    biomarcadores = [
        "frecuencia_max",
        "tipo_dolor_pecho",
        "talasemia",
        "num_vasos_principales",
        "edad",
        "depresion_st",
        "colesterol",
        "presion_reposo",
        "angina_ejercicio",
        "pendiente_st",
        "sexo",
        "electro_reposo",
        "azucar_ayunas"
    ]

    importancia = [
        0.135404,
        0.127163,
        0.122940,
        0.100811,
        0.091327,
        0.089358,
        0.088681,
        0.080716,
        0.050730,
        0.046626,
        0.035947,
        0.018389,
        0.011908
    ]

    fig, ax = plt.subplots(figsize=(4,2))

    sns.barplot(
        x=importancia,
        y=biomarcadores,
        palette="viridis",
        ax=ax
    )

    ax.set_title("Importancia Biomarcadores", fontsize=9)

    ax.set_xlabel("Importancia", fontsize=7)

    ax.set_ylabel("")

    ax.tick_params(axis='both', labelsize=7)

    plt.tight_layout()

    c1, c2, c3 = st.columns([2,1,2])

    with c2:
    	st.pyplot(fig)

    st.info("""
Principales biomarcadores identificados por Random Forest:

• Frecuencia cardíaca máxima

• Tipo de dolor torácico

• Talasemia

• Número de vasos coronarios principales

• Edad
""")

elif opcion == "📉 Curva ROC":

    st.markdown("### 📉 Curva ROC del Modelo Random Forest")

    fig, ax = plt.subplots(figsize=(3,2))

    fpr = [0.0, 0.03, 0.03, 0.06, 0.08, 0.12, 0.18, 0.45, 1.0]

    tpr = [0.0, 0.28, 0.72, 0.90, 0.93, 0.96, 0.98, 1.0, 1.0]

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label="AUC=0.951"
    )

    ax.plot(
        [0,1],
        [0,1],
        linestyle="--",
        color="gray"
    )

    ax.set_title("Curva ROC", fontsize=9)

    ax.set_xlabel("FPR", fontsize=7)

    ax.set_ylabel("TPR", fontsize=7)

    ax.tick_params(axis='both', labelsize=7)

    ax.legend(fontsize=7)

    ax.grid(True)

    plt.tight_layout()

    c1, c2, c3 = st.columns([2,1,2])

    with c2:
    	st.pyplot(fig)

    st.metric(
        "ROC-AUC",
        "0.951"
    )

# ==============================================================================

elif opcion == "🎯 Matriz de Confusión":

    st.markdown("### 🎯 Matriz de Confusión del Modelo")

    matriz = np.array([
        [27, 6],
        [1, 27]
    ])

    fig, ax = plt.subplots(figsize=(2.5,2))

    sns.heatmap(
        matriz,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Sano","Cardio"],
        yticklabels=["Sano","Cardio"],
        annot_kws={"size":8},
        ax=ax
    )

    ax.set_title(
        "Matriz de Confusión",
        fontsize=9
    )

    ax.set_xlabel(
        "Predicción",
        fontsize=7
    )

    ax.set_ylabel(
        "Real",
        fontsize=7
    )

    ax.tick_params(axis='both', labelsize=7)

    plt.tight_layout()

    c1, c2, c3 = st.columns([2,1,2])

    with c2:
    	st.pyplot(fig)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("TN", "27")
    c2.metric("FP", "6")
    c3.metric("FN", "1")
    c4.metric("TP", "27")

st.write("---")

if modelo is None or scaler is None:
    st.error(
        "⚠️ No se encontraron los archivos 'modelo_cardiaco.pkl' y/o "
        "'scaler_cardiaco.pkl' en el directorio. "
        "Asegúrese de colocarlos junto a 'app.py'."
    )

else:

    col1, col2 = st.columns([1.2, 1])

    with col1:

        st.subheader("🔍 Resumen de Mediciones Ingresadas")

        st.dataframe(
            datos_paciente,
            use_container_width=True
        )

        boton_diagnostico = st.button(
            "Analizar Riesgo Cardíaco",
            type="primary",
            use_container_width=True
        )

    with col2:

        st.subheader("📊 Resultado del Diagnóstico")

        if boton_diagnostico:

            datos_escalados = scaler.transform(datos_paciente)

            prediccion = modelo.predict(datos_escalados)[0]

            probabilidad = (
                modelo.predict_proba(datos_escalados)[0][1] * 100
            )

            if prediccion == 1:

                st.error(
                    "### ⚠️ Riesgo Alto: Probable Cardiopatía"
                )

                st.metric(
                    label="Probabilidad estimada de afección coronaria",
                    value=f"{probabilidad:.1f}%"
                )

                st.progress(int(probabilidad))

                st.warning("""
**Criterio Clínico:**

El algoritmo detectó patrones concordantes con enfermedad cardíaca.

**Recomendación:**
Se sugiere valoración prioritaria por cardiología y estudios de confirmación.
""")

            else:

                st.success(
                    "### ✅ Riesgo Bajo: Patrón Normal / Sano"
                )

                st.metric(
                    label="Probabilidad estimada de afección coronaria",
                    value=f"{probabilidad:.1f}%"
                )

                st.progress(int(probabilidad))

                st.info("""
**Criterio Clínico:**

Los biomarcadores evaluados se ubican dentro de parámetros de bajo riesgo.

**Recomendación:**
Mantener hábitos de vida saludable y control médico periódico.
""")

        else:

            st.info(
                "👈 Ajuste las variables clínicas en el panel lateral "
                "y haga clic en 'Analizar Riesgo Cardíaco' para obtener el dictamen."
            )

# ==============================================================================
# 5. PIE DE PÁGINA METODOLÓGICO
# ==============================================================================
st.write("---")
with st.expander("ℹ️ Información Metodológica y Ética del Modelo"):
    st.markdown("""
    * **Algoritmo Base:** Random Forest Classifier (100 estimadores).
    * **Métrica Principal:** Recall / Sensibilidad del **96.43%** en el conjunto ciego de prueba, minimizando el riesgo de falsos negativos médicos.
    * **Limitación y Ética:** Este aplicativo constituye una herramienta de asistencia diagnóstica académica y no reemplaza el criterio ni las pruebas confirmatorias de un profesional médico certificado.
    """)