# CardioPredict AI · Tamizaje de riesgo de cardiopatía

**Proyecto ACA 1 · Aprendizaje Automático** · Especialización en Inteligencia Artificial (CUN)

- **Autora:** Claudia Teresa Aranguren Peña
- **Curso:** APRENDIZAJE AUTOMÁTICO/54ES2/SEGUNDO BLOQUE/26ES4
- **Docente:** Dr. Hamilton Rivera Flor

## Enlaces

- Aplicación (Streamlit Community Cloud): https://cardiopredict-ai-ctap.streamlit.app
- Notebook (Google Colab): https://colab.research.google.com/drive/1Xge3m_tg7hq6MR-TQzmT5tycTc0Kv-MM?usp=sharing
- Video de sustentación: https://youtu.be/FbgWG-W6Ia4

## Qué hace

Estima la probabilidad de cardiopatía a partir de 13 variables clínicas de rutina, con una
Regresión Logística entrenada sobre el dataset UCI Heart Disease (Cleveland, 303 pacientes).
El modelo se eligió por validación cruzada frente a Random Forest y KNN, y opera con un umbral
de 0.25 para priorizar la detección de enfermos (Recall). Es una herramienta académica de
tamizaje: no reemplaza el diagnóstico médico.

## Archivos

| Archivo | Contenido |
| --- | --- |
| `heart_disease.ipynb` | Notebook completo: EDA, limpieza, modelado, evaluación y exportación |
| `app.py` | Aplicación de Streamlit |
| `requirements.txt` | Librerías que instala el servidor |
| `modelo_cardiaco.pkl` | Pipeline entrenado (preprocesamiento + Regresión Logística) |
| `metricas_modelo.json` | Métricas, umbral, matriz de confusión, curva ROC y coeficientes |
| `processed.cleveland.data` | Dataset original (UCI) |

## Ejecutar en local

```
pip install -r requirements.txt
streamlit run app.py
```

## Referencias

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1988). *Heart disease* [Conjunto de datos].
UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python.
*Journal of Machine Learning Research, 12*, 2825–2830.
