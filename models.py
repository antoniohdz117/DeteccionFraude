#biblioetcas utilizadas para el entrenamiento de los modelos
import joblib
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def entrenamientoModelos(X_train, y_train):
    
    

    # Cálculo para manejar desbalanceo en XGBoost
    cantidad_no_fraude = (y_train == 0).sum()
    cantidad_fraude = (y_train == 1).sum()

    scale_pos_weight = cantidad_no_fraude / cantidad_fraude

    modelos = {
        "Regresion_Logistica": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Random_Forest": RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1
        )
    }

    modelos_entrenados = {}

    for nombre, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre}")

        inicio = time.time()

        modelo.fit(X_train, y_train)

        fin = time.time()
        tiempo = round(fin - inicio, 2)

        modelos_entrenados[nombre] = modelo

        

        nombre_archivo = f"{nombre}.pkl"
        joblib.dump(modelo, nombre_archivo)

        



    return modelos_entrenados