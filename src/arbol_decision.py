import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

def entrenar_modelo(df):
    # variables predictoras
    X = df.drop("riesgo_cardiaco", axis=1)

    # variable objetivo
    y = df["riesgo_cardiaco"]

    # dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

    # crear modelo
    modelo = DecisionTreeClassifier(criterion="gini",max_depth=3,random_state=42)

    # entrenar
    modelo.fit(X_train, y_train)

    # predicciones
    y_pred = modelo.predict(X_test)

    print("=== ACCURACY ===")
    print(accuracy_score(y_test, y_pred))

    print("\n=== REPORTE DE CLASIFICACIÓN ===")
    print(classification_report(y_test, y_pred))

    return modelo, X, y