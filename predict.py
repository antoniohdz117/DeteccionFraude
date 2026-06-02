import os
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path

from logs import configurar_logger


# ============================================================
# CONFIGURACIÓN DE RUTAS Y LOGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODELO_PATH = BASE_DIR / "modelo_fraude_final.pkl"
PREPROCESSOR_PATH = BASE_DIR / "preprocessor.pkl"
HISTORIAL_PATH = BASE_DIR / "historial_predicciones.csv"

# En predict.py NO reiniciamos el archivo de logs
logger = configurar_logger("PREDICT", reiniciar=False)


# ============================================================
# CARGA DE MODELO
# ============================================================

def cargar_modelo():
    """
    Carga el modelo final y el preprocesador guardado.
    """

    try:
        logger.info("Iniciando carga del modelo y preprocesador")

        if not MODELO_PATH.exists():
            logger.error(f"No se encontró el modelo en la ruta: {MODELO_PATH}")
            raise FileNotFoundError(
                "No se encontró modelo_fraude_final.pkl. Primero debes entrenar y evaluar el modelo."
            )

        if not PREPROCESSOR_PATH.exists():
            logger.error(f"No se encontró el preprocesador en la ruta: {PREPROCESSOR_PATH}")
            raise FileNotFoundError(
                "No se encontró preprocessor.pkl. Primero debes guardar el preprocesador."
            )

        modelo = joblib.load(MODELO_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        logger.info("Modelo y preprocesador cargados correctamente")

        return modelo, preprocessor

    except Exception as error:
        logger.exception(f"Error al cargar modelo o preprocesador: {error}")
        raise


# ============================================================
# PREDICCIÓN
# ============================================================

def predecir_transaccion(datos_transaccion):
    """
    Recibe una transacción nueva, la transforma y predice si es fraude.
    """

    try:
        logger.info("Iniciando predicción de nueva transacción")
        logger.info(f"Datos recibidos: {datos_transaccion}")

        modelo, preprocessor = cargar_modelo()

        df_nuevo = pd.DataFrame([datos_transaccion])

        datos_procesados = preprocessor.transform(df_nuevo)

        prediccion = modelo.predict(datos_procesados)[0]

        if hasattr(modelo, "predict_proba"):
            probabilidad = modelo.predict_proba(datos_procesados)[0][1]
        else:
            probabilidad = float(prediccion)

        if prediccion == 1:
            resultado = "FRAUDE"
        else:
            resultado = "NORMAL"

        logger.info(
            f"Predicción realizada correctamente. "
            f"Resultado: {resultado}, Probabilidad: {probabilidad:.4f}"
        )

        return resultado, probabilidad

    except Exception as error:
        logger.exception(f"Error durante la predicción: {error}")
        raise


# ============================================================
# GUARDAR HISTORIAL
# ============================================================

def guardar_prediccion(datos_transaccion, resultado, probabilidad):
    """
    Guarda la transacción nueva y el resultado en un CSV.
    """

    try:
        logger.info("Guardando predicción en historial_predicciones.csv")

        registro = datos_transaccion.copy()
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
        logger.exception(f"Error al guardar la predicción: {error}")
        raise


# ============================================================
# EJEMPLO DE PREDICCIÓN
# ============================================================

def ejemplo_prediccion():
    """
    Ejemplo de una transacción nueva.
    Estos campos deben coincidir con los usados en preprocessing.py.
    """

    try:
        logger.info("Iniciando ejemplo de predicción desde predict.py")

        transaccion_nueva = {
            "amt": 2500.00,
            "lat": 19.4326,
            "long": -99.1332,
            "city_pop": 9200000,
            "merch_lat": 20.6597,
            "merch_long": -103.3496,
            "hour": 23,
            "day": 15,
            "month": 5,
            "day_of_week": 5,
            "age": 32,
            "distance_km": 460.0,
            "category": "shopping_net",
            "gender": "M",
            "state": "MX"
        }

        resultado, probabilidad = predecir_transaccion(transaccion_nueva)

        print("\n========== PREDICCIÓN DE TRANSACCIÓN ==========")
        print("Resultado:", resultado)
        print("Probabilidad de fraude:", round(probabilidad * 100, 2), "%")

        guardar_prediccion(transaccion_nueva, resultado, probabilidad)

        print("\nPredicción guardada en historial_predicciones.csv")

        logger.info(
            f"Ejemplo de predicción finalizado. "
            f"Resultado: {resultado}, Probabilidad: {probabilidad:.4f}"
        )

    except Exception as error:
        logger.exception(f"Error en ejemplo_prediccion: {error}")

        print("\nOcurrió un error al ejecutar predict.py")
        print("Revisa el archivo de logs para más detalles.")
        print(error)


# ============================================================
# EJECUCIÓN
# ============================================================

if __name__ == "__main__":
    ejemplo_prediccion()