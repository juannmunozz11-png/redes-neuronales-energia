import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ---------- 1. Datos ----------
df = pd.read_excel("ENB2012_data.xlsx")
X = df[["X1","X2","X3","X4","X5","X6","X7","X8"]]
y = df["Y1"]

# Particion 60% entrenamiento / 20% validacion / 20% prueba
X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=42)

print(f"Entrenamiento: {len(X_train)}  Validacion: {len(X_val)}  Prueba: {len(X_test)}")

# Escalado (ajustado solo con entrenamiento)
esc = StandardScaler()
X_train_e = esc.fit_transform(X_train)
X_val_e   = esc.transform(X_val)
X_test_e  = esc.transform(X_test)

# ---------- 2. Configuraciones ----------
topologias = {
    "1 capa (10)":        (10,),
    "1 capa (30)":        (30,),
    "2 capas (20,10)":    (20, 10),
    "3 capas (30,20,10)": (30, 20, 10),
}

algoritmos = {
    "Adam":   dict(solver="adam"),
    "L-BFGS": dict(solver="lbfgs"),
    "SGD":    dict(solver="sgd", momentum=0.9),
}

# ---------- 3. Entrenar y evaluar en validacion ----------
resultados = []

for nom_top, capas in topologias.items():
    for nom_alg, params in algoritmos.items():
        red = MLPRegressor(hidden_layer_sizes=capas,
                           max_iter=5000,
                           random_state=42,
                           **params)
        red.fit(X_train_e, y_train)

        p_tr  = red.predict(X_train_e)
        p_val = red.predict(X_val_e)

        resultados.append({
            "Topologia":  nom_top,
            "Algoritmo":  nom_alg,
            "R2_train":   round(r2_score(y_train, p_tr), 4),
            "R2_val":     round(r2_score(y_val, p_val), 4),
            "RMSE_val":   round(np.sqrt(mean_squared_error(y_val, p_val)), 4),
            "MAE_val":    round(mean_absolute_error(y_val, p_val), 4),
        })
        print(f"OK  {nom_top:20s} {nom_alg}")

tabla = pd.DataFrame(resultados).sort_values("R2_val", ascending=False)

print("\n=== RESULTADOS EN VALIDACION (12 configuraciones) ===")
print(tabla.to_string(index=False))

# ---------- 4. Mejor configuracion ----------
mejor = tabla.iloc[0]
print(f"\nMEJOR CONFIGURACION: {mejor['Topologia']} + {mejor['Algoritmo']}")
print(f"R2 validacion = {mejor['R2_val']}")

tabla.to_csv("resultados_redes.csv", index=False)
