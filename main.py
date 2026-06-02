import joblib

from data_loader import cargar_datos
from preprocessing import preparar_datos
from evaluation import evaluar_modelos
from models import entrenamientoModelos

# Documentación de logs
from logs import configurar_logger


# En main.py usamos reiniciar=True para crear el archivo de logs desde cero
logger = configurar_logger("MAIN", reiniciar=True)


def main():
    try:
        logger.info("========================================")
        logger.info("INICIO DEL SISTEMA DE DETECCIÓN DE FRAUDE")
        logger.info("========================================")

        print("\n========================================")
        print(" SISTEMA DE DETECCIÓN DE FRAUDE")
        print("========================================")

        # ============================================================
        # 1. CARGA DE DATOS
        # ============================================================

        logger.info("[1/5] Iniciando carga de datos")
        print("\n[1/5] Cargando datos...")

        train, test = cargar_datos()

        logger.info(f"Datos cargados correctamente. Train: {train.shape}, Test: {test.shape}")
        logger.info(f"Columnas train: {list(train.columns)}")

        print("\nDatos cargados correctamente.")

        # ============================================================
        # 2. PREPROCESAMIENTO
        # ============================================================

        logger.info("[2/5] Iniciando preprocesamiento de datos")
        print("\n[2/5] Preprocesando datos...")

        X_train, X_test, y_train, y_test, preprocessor = preparar_datos(train, test)

        logger.info("Preprocesamiento finalizado correctamente")
        logger.info(f"X_train: {X_train.shape}")
        logger.info(f"X_test: {X_test.shape}")
        logger.info(f"y_train: {y_train.shape}")
        logger.info(f"y_test: {y_test.shape}")

        print("\nDatos preprocesados correctamente.")

        # Guardar preprocesador
        joblib.dump(preprocessor, "preprocessor.pkl")

        logger.info("Preprocesador guardado como preprocessor.pkl")
        print("Preprocesador guardado como: preprocessor.pkl")

        # ============================================================
        # 3. ENTRENAMIENTO DE MODELOS
        # ============================================================

        logger.info("[3/5] Iniciando entrenamiento de modelos")
        print("\n[3/5] Entrenando modelos...")

        modelos_entrenados = entrenamientoModelos(X_train, y_train)

        logger.info("Entrenamiento de modelos finalizado correctamente")
        logger.info(f"Modelos entrenados: {list(modelos_entrenados.keys())}")

        print("\nModelos entrenados correctamente.")

        # ============================================================
        # 4. EVALUACIÓN DE MODELOS
        # ============================================================

        logger.info("[4/5] Iniciando evaluación de modelos")
        print("\n[4/5] Evaluando modelos...")

        resultados, mejor_modelo, mejor_nombre = evaluar_modelos(
            modelos_entrenados,
            X_test,
            y_test
        )

        logger.info("Evaluación finalizada correctamente")
        logger.info(f"Mejor modelo seleccionado: {mejor_nombre}")
        logger.info("\nResultados de modelos:\n" + str(resultados))

        print("\nEvaluación finalizada correctamente.")

        # ============================================================
        # 5. RESUMEN FINAL
        # ============================================================

        logger.info("[5/5] Proceso terminado correctamente")

        print("\n[5/5] Proceso terminado.")

        print("\n========================================")
        print(" RESUMEN DEL PROYECTO")
        print("========================================")

        print("\nModelos evaluados:")
        print(resultados)

        print("\nMejor modelo seleccionado:")
        print(mejor_nombre)

        print("\nArchivos generados:")
        print("- preprocessor.pkl")
        print("- modelo_fraude_final.pkl")
        print("- resultados_modelos.csv")
        print("- modelos individuales .pkl")

        print("\nAhora puedes ejecutar:")
        print("1. predict.py para probar una transacción nueva")
        print("2. app.py para abrir la aplicación web con Streamlit")

        print("\nProceso completado exitosamente.")

        logger.info("Archivos generados:")
        logger.info("- preprocessor.pkl")
        logger.info("- modelo_fraude_final.pkl")
        logger.info("- resultados_modelos.csv")
        logger.info("- modelos individuales .pkl")
        logger.info("FIN DEL SISTEMA DE DETECCIÓN DE FRAUDE")

    except Exception as error:
        logger.exception(f"Ocurrió un error durante la ejecución de main.py: {error}")

        print("\nOcurrió un error durante la ejecución.")
        print("Revisa el archivo de logs para más detalles.")
        print(error)


if __name__ == "__main__":
    main()