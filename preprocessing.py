#Arhcivo para hacer un preprocesamiento de los datos, estandarizar las variables numericas y codificar las categoricas
import numpy as np
import pandas as pd

# Preprocesamiento de datos
from sklearn.compose import ColumnTransformer
#standarScaler para estandarizar las variables numericas y OneHotEncoder para codificar las variables categoricas pasarlo a 0 y 1
from sklearn.preprocessing import StandardScaler, OneHotEncoder




# esto puede ser un indicador importante de fraude si la distancia es muy grande entre el cliente y el comercio, 
# especialmente si la transacción ocurre en un corto período de tiempo después de una transacción anterior. 
def calcular_distancia_km(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia aproximada en kilómetros entre el cliente y el comercio.
    Usa la fórmula de Haversine.
    """

    # Radio promedio de la Tierra en kilómetros
    radioTierra = 6371  # kilómetros

    # Convertir grados a radianes   
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)


    # Calcular diferencias
    diferenciaLat = lat2 - lat1
    diferenciaLon = lon2 - lon1

    a = (
        np.sin(diferenciaLat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(diferenciaLon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return radioTierra * c



#Creamos nuevas variables a partir de las existentes para ayudar a detectar 
# patrones de fraude.
def crear_variables(df):
    """
    Crea nuevas variables útiles para detectar patrones de fraude.
    """
    
    #copy del dataframe para no modificar el original
    df = df.copy()

    # Convertir fecha de transacción
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"],
        errors="coerce"
    )

    # Variables de tiempo
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek

    # Convertir fecha de nacimiento
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce")

    # Calcular edad aproximada del cliente
    df["age"] = (
        (df["trans_date_trans_time"] - df["dob"]).dt.days / 365
    ).astype(float)

    # Calcular distancia entre cliente y comercio
    df["distance_km"] = calcular_distancia_km(
        df["lat"], 
        df["long"],
        df["merch_lat"],
        df["merch_long"]
    )

    # Rellenar posibles valores nulos generados
    df["age"] = df["age"].fillna(df["age"].median())
    df["hour"] = df["hour"].fillna(0)
    df["day"] = df["day"].fillna(0)
    df["month"] = df["month"].fillna(0)
    df["day_of_week"] = df["day_of_week"].fillna(0)
    df["distance_km"] = df["distance_km"].fillna(df["distance_km"].median())

    return df


#preparamos los datos para el entrenamiento de modelos, 
def preparar_datos(train, test):
    """
    Prepara train y test para entrenamiento de modelos.
    """

    print("\n========== INICIANDO PREPROCESAMIENTO ==========")

    train = crear_variables(train)
    test = crear_variables(test)

    # Variable objetivo
    columna_objetivo = "is_fraud"

    y_train = train[columna_objetivo]
    y_test = test[columna_objetivo]

    X_train = train.drop(columns=[columna_objetivo])
    X_test = test.drop(columns=[columna_objetivo])

    # Columnas que no usaremos directamente
    columnas_eliminar = [
        "cc_num",
        "merchant",
        "first",
        "last",
        "street",
        "city",
        "zip",
        "job",
        "trans_num",
        "unix_time",
        "dob",
        "trans_date_trans_time"
    ]

    columnas_eliminar = [col for col in columnas_eliminar if col in X_train.columns]

    X_train = X_train.drop(columns=columnas_eliminar)
    X_test = X_test.drop(columns=columnas_eliminar)

    # Variables numéricas
    columnas_numericas = [
        "amt",
        "lat",
        "long",
        "city_pop",
        "merch_lat",
        "merch_long",
        "hour",
        "day",
        "month",
        "day_of_week",
        "age",
        "distance_km"
    ]

    columnas_numericas = [col for col in columnas_numericas if col in X_train.columns]

    # Variables categóricas
    columnas_categoricas = [
        "category",
        "gender",
        "state"
    ]

    columnas_categoricas = [col for col in columnas_categoricas if col in X_train.columns]

    print("\nColumnas numéricas usadas:")
    print(columnas_numericas)

    print("\nColumnas categóricas usadas:")
    print(columnas_categoricas)

    # Transformador de datos
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), columnas_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas)
        ]
    )

    # Ajustar con train y transformar train/test
    X_train_procesado = preprocessor.fit_transform(X_train)
    X_test_procesado = preprocessor.transform(X_test)

    print("\n========== PREPROCESAMIENTO FINALIZADO ==========")
    print("X_train procesado:", X_train_procesado.shape)
    print("X_test procesado:", X_test_procesado.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    return X_train_procesado, X_test_procesado, y_train, y_test, preprocessor


# #evaluamos el modelo utilizando las métricas de clasificación
# def evaluar_modelo(modelo, X_test, y_test):
#     """
#     Evalúa el modelo utilizando métricas de clasificación.
#     """
#     from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#     y_pred = modelo.predict(X_test)

#     print("Accuracy:", accuracy_score(y_test, y_pred))
#     print(confusion_matrix(y_test, y_pred))
#     print(classification_report(y_test, y_pred))
    
# #evaluamos las fucniones
# print("================================================================= ==========")
# data =  calcular_distancia_km(40.7128, -74.0060, 34.0522, -118.2437)  # Distancia entre NYC y LA
# print("Distancia entre NYC y LA (km):", data)
# print("================================================================= ==========")
# print("================================================================= ==========")
# data = crear_variables(df=pd.DataFrame({
#     "trans_date_trans_time": ["2021-01-01 12:00:00", "2021-01-02 15:30:00"],
#     "dob": ["1990-01-01", "1985-05-15"],
#     "lat": [40.7128, 34.0522],
#     "long": [-74.0060, -118.2437],
#     "merch_lat": [40.730610, 34.052235],
#     "merch_long": [-73.935242, -118.243683]
# }))

# print(data)
# print("================================================================= ==========")
# #Prepacion de datos
# print("================================================================= ==========")
# X_train_procesado, X_test_procesado, y_train, y_test, preprocessor = preparar_datos(
#     train=pd.DataFrame({
#         "trans_date_trans_time": ["2021-01-01 12:00:00", "2021-01-02 15:30:00"],
#         "dob": ["1990-01-01", "1985-05-15"],
#         "lat": [40.7128, 34.0522],
#         "long": [-74.0060, -118.2437],
#         "merch_lat": [40.730610, 34.052235],
#         "merch_long": [-73.935242, -118.243683],
#         "is_fraud": [0, 1]
#     }),
#     test=pd.DataFrame({
#         "trans_date_trans_time": ["2021-01-03 10:00:00", "2021-01-04 18:45:00"],
#         "dob": ["1992-02-20", "1988-08-30"],
#         "lat": [41.8781, 29.7604],
#         "long": [-87.6298, -95.3698],
#         "merch_lat": [41.881832, 29.749907],
#         "merch_long": [-87.623177, -95.358421],
#         "is_fraud": [0, 1]
#     })
# )

# print("X_train procesado:", X_train_procesado)
# print("X_test procesado:", X_test_procesado)    
# print("================================================================= ==========")