import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from logs import configurar_logger


# ============================================================
# CONFIGURACIÓN DE RUTAS Y LOGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELO_PATH = BASE_DIR / "modelo_fraude_final.pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"
HISTORIAL_PATH = BASE_DIR / "historial_predicciones.csv"

# En app.py NO reiniciamos logs, solo agregamos registros
logger = configurar_logger("APP", reiniciar=False)


# ============================================================
# CONFIGURACIÓN DE LA APP
# ============================================================

st.set_page_config(
    page_title="Detección de Fraude",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def calcular_distancia_km(lat1, lon1, lat2, lon2):
    """
    Calcula distancia entre cliente y comercio usando fórmula de Haversine.
    """

    radio_tierra = 6371

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    diferencia_lat = lat2 - lat1
    diferencia_lon = lon2 - lon1

    a = (
        np.sin(diferencia_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(diferencia_lon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return radio_tierra * c


@st.cache_resource
def cargar_modelo_y_preprocesador():
    """
    Carga el modelo final y el preprocesador.
    """

    try:
        logger.info("Iniciando carga de modelo y preprocesador desde app.py")

        if not MODELO_PATH.exists():
            logger.error(f"No se encontró el modelo en la ruta: {MODELO_PATH}")
            st.error("No se encontró modelo_fraude_final.pkl. Primero entrena y evalúa el modelo.")
            st.stop()

        if not PREPROCESSOR_PATH.exists():
            logger.error(f"No se encontró el preprocesador en la ruta: {PREPROCESSOR_PATH}")
            st.error("No se encontró preprocessor.pkl. Primero guarda el preprocesador.")
            st.stop()

        modelo = joblib.load(MODELO_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        logger.info("Modelo y preprocesador cargados correctamente en app.py")

        return modelo, preprocessor

    except Exception as error:
        logger.exception(f"Error al cargar modelo o preprocesador en app.py: {error}")
        st.error("Ocurrió un error al cargar el modelo o el preprocesador.")
        st.stop()


def predecir_fraude(transaccion):
    """
    Recibe una transacción, la procesa y devuelve predicción.
    """

    try:
        logger.info("Iniciando predicción desde app.py")
        logger.info(f"Datos de transacción recibidos: {transaccion}")

        modelo, preprocessor = cargar_modelo_y_preprocesador()

        df_transaccion = pd.DataFrame([transaccion])

        df_procesado = preprocessor.transform(df_transaccion)

        prediccion = modelo.predict(df_procesado)[0]

        if hasattr(modelo, "predict_proba"):
            probabilidad = modelo.predict_proba(df_procesado)[0][1]
        else:
            probabilidad = float(prediccion)

        resultado = "FRAUDE" if prediccion == 1 else "NORMAL"

        logger.info(
            f"Predicción realizada en app.py. "
            f"Resultado: {resultado}, Probabilidad: {probabilidad:.4f}"
        )

        return prediccion, probabilidad

    except Exception as error:
        logger.exception(f"Error durante la predicción en app.py: {error}")
        st.error("Ocurrió un error al realizar la predicción.")
        st.stop()


def guardar_historial(transaccion, resultado, probabilidad):
    """
    Guarda cada predicción en un archivo CSV.
    """

    try:
        logger.info("Guardando predicción en historial desde app.py")

        registro = transaccion.copy()
        registro["resultado"] = resultado
        registro["probabilidad_fraude"] = round(probabilidad, 4)
        registro["fecha_prediccion"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        df_registro = pd.DataFrame([registro])

        if HISTORIAL_PATH.exists():
            df_registro.to_csv(HISTORIAL_PATH, mode="a", header=False, index=False)
        else:
            df_registro.to_csv(HISTORIAL_PATH, index=False)

        logger.info(f"Predicción guardada correctamente en: {HISTORIAL_PATH}")

    except Exception as error:
        logger.exception(f"Error al guardar historial en app.py: {error}")
        st.error("La predicción se realizó, pero ocurrió un error al guardar el historial.")


# ============================================================
# INTERFAZ PRINCIPAL
# ============================================================

logger.info("Aplicación Streamlit iniciada")

st.title("💳 Sistema de Detección de Fraude en Transacciones")
st.write(
    "Esta aplicación permite ingresar una transacción nueva y estimar "
    "si puede ser fraudulenta usando un modelo de Machine Learning."
)

st.divider()

modelo, preprocessor = cargar_modelo_y_preprocesador()

st.success("Modelo y preprocesador cargados correctamente.")


# ============================================================
# FORMULARIO DE TRANSACCIÓN
# ============================================================

st.header("Ingresar nueva transacción")

col1, col2, col3 = st.columns(3)

with col1:
    amt = st.number_input(
        "Monto de la transacción",
        min_value=0.0,
        value=250.0,
        step=10.0
    )

    category = st.selectbox(
        "Categoría del comercio",
        [
            "shopping_net",
            "shopping_pos",
            "grocery_pos",
            "grocery_net",
            "gas_transport",
            "food_dining",
            "entertainment",
            "health_fitness",
            "home",
            "kids_pets",
            "misc_net",
            "misc_pos",
            "personal_care",
            "travel"
        ]
    )

    gender = st.selectbox(
        "Género",
        ["M", "F"]
    )

    age = st.number_input(
        "Edad del cliente",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:
    state = st.selectbox(
        "Estado",
        [
            "CA", "TX", "NY", "FL", "IL", "PA", "OH", "GA", "NC", "MI",
            "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"
        ]
    )

    city_pop = st.number_input(
        "Población de la ciudad",
        min_value=0,
        value=100000
    )

    hour = st.slider(
        "Hora de la transacción",
        min_value=0,
        max_value=23,
        value=14
    )

    day = st.slider(
        "Día del mes",
        min_value=1,
        max_value=31,
        value=15
    )

with col3:
    month = st.slider(
        "Mes",
        min_value=1,
        max_value=12,
        value=5
    )

    day_of_week = st.selectbox(
        "Día de la semana",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: [
            "Lunes",
            "Martes",
            "Miércoles",
            "Jueves",
            "Viernes",
            "Sábado",
            "Domingo"
        ][x]
    )

    lat = st.number_input(
        "Latitud del cliente",
        value=34.0522,
        format="%.6f"
    )

    long = st.number_input(
        "Longitud del cliente",
        value=-118.2437,
        format="%.6f"
    )

st.subheader("Ubicación del comercio")

col4, col5 = st.columns(2)

with col4:
    merch_lat = st.number_input(
        "Latitud del comercio",
        value=36.1699,
        format="%.6f"
    )

with col5:
    merch_long = st.number_input(
        "Longitud del comercio",
        value=-115.1398,
        format="%.6f"
    )


# ============================================================
# CÁLCULO DE DISTANCIA
# ============================================================

distance_km = calcular_distancia_km(lat, long, merch_lat, merch_long)

st.info(f"Distancia aproximada entre cliente y comercio: {distance_km:.2f} km")


# ============================================================
# PREDICCIÓN
# ============================================================

if st.button("Predecir fraude", type="primary"):

    logger.info("Usuario presionó el botón de predicción en app.py")

    transaccion = {
        "amt": amt,
        "lat": lat,
        "long": long,
        "city_pop": city_pop,
        "merch_lat": merch_lat,
        "merch_long": merch_long,
        "hour": hour,
        "day": day,
        "month": month,
        "day_of_week": day_of_week,
        "age": age,
        "distance_km": distance_km,
        "category": category,
        "gender": gender,
        "state": state
    }

    prediccion, probabilidad = predecir_fraude(transaccion)

    st.divider()
    st.header("Resultado de la predicción")

    probabilidad_porcentaje = probabilidad * 100

    col_resultado1, col_resultado2 = st.columns(2)

    with col_resultado1:
        st.metric(
            label="Probabilidad de fraude",
            value=f"{probabilidad_porcentaje:.2f}%"
        )

    with col_resultado2:
        if prediccion == 1:
            st.error("Resultado: posible transacción fraudulenta")
            resultado_texto = "FRAUDE"
        else:
            st.success("Resultado: transacción normal")
            resultado_texto = "NORMAL"

    logger.info(
        f"Resultado mostrado en app.py. "
        f"Resultado: {resultado_texto}, "
        f"Probabilidad: {probabilidad:.4f}, "
        f"Monto: {amt}, "
        f"Categoría: {category}, "
        f"Hora: {hour}"
    )

    guardar_historial(transaccion, resultado_texto, probabilidad)

    st.write("La transacción fue guardada en `historial_predicciones.csv`.")


# ============================================================
# HISTORIAL
# ============================================================

st.divider()
st.header("Historial de predicciones")

if HISTORIAL_PATH.exists():
    historial = pd.read_csv(HISTORIAL_PATH)
    st.dataframe(historial.tail(20), width="stretch")
else:
    st.write("Todavía no hay predicciones guardadas.")