# bibliotecas utilizadas
import pandas as pd
#pathlib para manejar rutas de archivos
import pathlib as Path


#funcion para cargar el dataset
def cargar_datos():
    """
    Carga los archivos fraudTrain.csv y fraudTest.csv.
    Valida que existan, elimina columnas innecesarias y muestra información básica.
    """
    
    ruta_train = "data/fraudTrain.csv"
    ruta_test = "data/fraudTest.csv"
    
    #primero validamos que existan los archivos
    if not Path.Path("data/fraudTrain.csv").exists():
        raise FileNotFoundError("El archivo fraudTrain.csv no se encuentra en la carpeta data.")
    
    if not Path.Path("data/fraudTest.csv").exists():
        raise FileNotFoundError("El archivo fraudTest.csv no se encuentra en la carpeta data.")
    
    #CARGAR AR CHIVOS
    train = pd.read_csv(ruta_train)
    test = pd.read_csv(ruta_test)
    
    #ELIMINAR COLUMNAS INNECESARIAS
    if "Unnamed: 0" in train.columns:
        train = train.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in test.columns:
        test = test.drop(columns=["Unnamed: 0"])


    #validamos que exista una variable objetivo llamada "is_fraud"
    if "Unnamed: 0" in train.columns:
        train = train.drop(columns=["Unnamed: 0"])

    if "Unnamed: 0" in test.columns:
        test = test.drop(columns=["Unnamed: 0"])

    # Validar que exista la variable objetivo
    if "is_fraud" not in train.columns:
        raise ValueError("El archivo fraudTrain.csv no tiene la columna 'is_fraud'.")

    if "is_fraud" not in test.columns:
        raise ValueError("El archivo fraudTest.csv no tiene la columna 'is_fraud'.")
    
    print("\n========== DATOS CARGADOS CORRECTAMENTE ==========")
    print(f"Train: {train.shape[0]} filas y {train.shape[1]} columnas")
    print(f"Test:  {test.shape[0]} filas y {test.shape[1]} columnas")

    print("\n========== COLUMNAS DEL DATASET ==========")
    print(train.columns.tolist())

    print("\n========== PRIMERAS FILAS DE TRAIN ==========")
    print(train.head())

    print("\n========== DISTRIBUCIÓN DE FRAUDE EN TRAIN ==========")
    print(train["is_fraud"].value_counts())

    print("\n========== PORCENTAJE DE FRAUDE EN TRAIN ==========")
    print(train["is_fraud"].value_counts(normalize=True) * 100)

    print("\n========== VALORES NULOS EN TRAIN ==========")
    print(train.isnull().sum())

    return train, test



