import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


def evaluar_modelos(modelos, X_test, y_test):
    """
    Evalúa los modelos entrenados y selecciona el mejor usando F1-Score.
    """

    print("\n========== EVALUACIÓN DE MODELOS ==========")

    resultados = []

    mejor_modelo = None
    mejor_nombre = None
    mejor_f1 = 0

    for nombre, modelo in modelos.items():
        print(f"\n========== MODELO: {nombre} ==========")

        y_pred = modelo.predict(X_test)

        if hasattr(modelo, "predict_proba"):
            y_prob = modelo.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)

        print("\nMatriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

        print("\nReporte de clasificación:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\nMétricas:")
        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1-Score:", round(f1, 4))
        print("ROC-AUC:", round(roc_auc, 4))

        resultados.append({
            "Modelo": nombre,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC-AUC": roc_auc
        })

        if f1 > mejor_f1:
            mejor_f1 = f1
            mejor_modelo = modelo
            mejor_nombre = nombre

    df_resultados = pd.DataFrame(resultados)

    print("\n========== COMPARACIÓN DE MODELOS ==========")
    print(df_resultados)

    df_resultados.to_csv("resultados_modelos.csv", index=False)
    joblib.dump(mejor_modelo, "modelo_fraude_final.pkl")

    print("\n========== MEJOR MODELO ==========")
    print("Mejor modelo:", mejor_nombre)
    print("F1-Score:", round(mejor_f1, 4))
    print("Modelo guardado como: modelo_fraude_final.pkl")
    print("Resultados guardados como: resultados_modelos.csv")

    return df_resultados, mejor_modelo, mejor_nombre