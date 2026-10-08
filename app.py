import streamlit as st
import pandas as pd
import numpy as np
import joblib

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
st.title("🩺 Sistema Predictivo de Riesgo Cardíaco")
st.markdown("""
**Especialización en Inteligencia Artificial — CUN | Proyecto ACA 1**  
Esta herramienta clínica utiliza un modelo clásico de **Random Forest** optimizado para maximizar la **Sensibilidad (Recall)**, 
permitiendo evaluar la probabilidad de afección coronaria a partir de biomarcadores fisiológicos estándar (Dataset UCI Cleveland).
""")

st.write("---")

if modelo is None or scaler is None:
    st.error("⚠️ No se encontraron los archivos `modelo_cardiaco.pkl` y/o `scaler_cardiaco.pkl` en el directorio. Asegúrese de colocarlos junto a `app.py`.")
else:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🔍 Resumen de Mediciones Ingresadas")
        st.dataframe(datos_paciente, use_container_width=True)

        # Botón para detonar la inferencia
        boton_diagnostico = st.button("Analizar Riesgo Cardíaco", type="primary", use_container_width=True)

    with col2:
        st.subheader("📊 Resultado del Diagnóstico")
        
        if boton_diagnostico:
            # 1. Escalado estandarizado
            datos_escalados = scaler.transform(datos_paciente)
            
            # 2. Inferencia y cálculo de probabilidad
            prediccion = modelo.predict(datos_escalados)[0]
            probabilidad = modelo.predict_proba(datos_escalados)[0][1] * 100

            # 3. Presentación médica del resultado
            if prediccion == 1:
                st.error(f"### ⚠️ Riesgo Alto: Probable Cardiopatía")
                st.metric(label="Probabilidad estimada de afección coronaria", value=f"{probabilidad:.1f}%")
                st.progress(int(probabilidad))
                st.warning("""
                **Criterio Clínico:**  
                El algoritmo detectó patrones concordantes con enfermedad cardíaca.  
                *Recomendación:* Se sugiere valoración prioritaria por cardiología y estudios de confirmación (ecocardiograma o angiografía).
                """)
            else:
                st.success(f"### ✅ Riesgo Bajo: Patrón Normal / Sano")
                st.metric(label="Probabilidad estimada de afección coronaria", value=f"{probabilidad:.1f}%")
                st.progress(int(probabilidad))
                st.info("""
                **Criterio Clínico:**  
                Los biomarcadores evaluados se ubican dentro de los parámetros habituales de bajo riesgo.  
                *Recomendación:* Mantener hábitos de vida saludable y control periódico rutinario.
                """)
        else:
            st.info("👈 Ajuste las variables clínicas en el panel lateral y haga clic en **'Analizar Riesgo Cardíaco'** para obtener el dictamen.")

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