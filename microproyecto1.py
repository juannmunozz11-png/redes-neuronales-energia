"""
=============================================================================
MICRO PROYECTO 1 - REDES NEURONALES ARTIFICIALES EN INGENIERIA DE LA
CONSTRUCCION

Problema  : Prediccion de la carga termica de calefaccion de edificaciones
            a partir de sus caracteristicas geometricas.
Dataset   : Energy Efficiency Data Set (Tsanas & Xifara, 2012)
            UCI Machine Learning Repository
            https://archive.ics.uci.edu/dataset/242/energy+efficiency
Variables de entrada:
    X1 - Compacidad relativa
    X2 - Area de superficie (m2)
    X3 - Area de muro (m2)
    X4 - Area de cubierta (m2)
    X5 - Altura total (m)
    X6 - Orientacion
    X7 - Area de acristalamiento
    X8 - Distribucion del acristalamiento
Variable objetivo:
    Y1 - Carga de calefaccion (kWh/m2)

Ejecutar con:  py microproyecto1.py
=============================================================================
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEMILLA = 42


# =============================================================================
# 1. CARGA Y EXPLORACION DE LOS DATOS
# =============================================================================
print("=" * 70)
print("1. CARGA Y EXPLORACION DE LOS DATOS")
print("=" * 70)

df = pd.read_excel("ENB2012_data.xlsx")

ENTRADAS = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
OBJETIVO = "Y1"

X = df[ENTRADAS]
y = df[OBJETIVO]

print(f"Registros: {df.shape[0]}   Columnas: {df.shape[1]}")
print(f"Valores faltantes: {df.isnull().sum().sum()}")
print("\nEstadisticas de la variable objetivo (Y1 - carga de calefaccion):")
print(y.describe().round(2).to_string())


# =============================================================================
# 2. PARTICION DE LOS DATOS  (60% entrenamiento / 20% validacion / 20% prueba)
# =============================================================================
print("\n" + "=" * 70)
print("2. PARTICION DE LOS DATOS")
print("=" * 70)

X_tmp, X_test, y_tmp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEMILLA
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=SEMILLA
)

print(f"Entrenamiento : {len(X_train)} registros")
print(f"Validacion    : {len(X_val)} registros")
print(f"Prueba        : {len(X_test)} registros")

# Escalado: el escalador se ajusta UNICAMENTE con el conjunto de entrenamiento
# para evitar filtracion de informacion hacia validacion y prueba.
escalador = StandardScaler()
X_train_e = escalador.fit_transform(X_train)
X_val_e = escalador.transform(X_val)
X_test_e = escalador.transform(X_test)


# =============================================================================
# 3. CONFIGURACIONES DE LA RED NEURONAL
# =============================================================================
print("\n" + "=" * 70)
print("3. ENTRENAMIENTO DE LAS CONFIGURACIONES DE RNA")
print("=" * 70)

# Cuatro topologias: se varia el numero de capas ocultas y de neuronas.
TOPOLOGIAS = {
    "1 capa (10)": (10,),
    "1 capa (30)": (30,),
    "2 capas (20,10)": (20, 10),
    "3 capas (30,20,10)": (30, 20, 10),
}

# Tres algoritmos de entrenamiento disponibles en scikit-learn.
ALGORITMOS = {
    "Adam": dict(solver="adam"),
    "L-BFGS": dict(solver="lbfgs"),
    "SGD": dict(solver="sgd", momentum=0.9),
}


def metricas(nombre, y_real, y_pred):
    """Devuelve MAE, RMSE y R2 en un diccionario."""
    return {
        "Modelo": nombre,
        "MAE": round(mean_absolute_error(y_real, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_real, y_pred)), 4),
        "R2": round(r2_score(y_real, y_pred), 4),
    }


resultados = []

for nombre_top, capas in TOPOLOGIAS.items():
    for nombre_alg, parametros in ALGORITMOS.items():
        red = MLPRegressor(
            hidden_layer_sizes=capas,
            max_iter=5000,
            random_state=SEMILLA,
            **parametros,
        )
        red.fit(X_train_e, y_train)

        pred_train = red.predict(X_train_e)
        pred_val = red.predict(X_val_e)

        resultados.append(
            {
                "Topologia": nombre_top,
                "Algoritmo": nombre_alg,
                "MAE_val": round(mean_absolute_error(y_val, pred_val), 4),
                "RMSE_val": round(
                    np.sqrt(mean_squared_error(y_val, pred_val)), 4
                ),
                "R2_train": round(r2_score(y_train, pred_train), 4),
                "R2_val": round(r2_score(y_val, pred_val), 4),
            }
        )
        print(f"  entrenada: {nombre_top:20s} + {nombre_alg}")

tabla = pd.DataFrame(resultados).sort_values("R2_val", ascending=False)

print("\nDesempeno en el conjunto de VALIDACION (ordenado por R2):")
print(tabla.to_string(index=False))
tabla.to_csv("resultados_redes.csv", index=False)


# =============================================================================
# 4. SELECCION DE LA MEJOR CONFIGURACION
# =============================================================================
print("\n" + "=" * 70)
print("4. SELECCION DE LA MEJOR CONFIGURACION")
print("=" * 70)

mejor = tabla.iloc[0]
mejores_capas = TOPOLOGIAS[mejor["Topologia"]]
mejores_params = ALGORITMOS[mejor["Algoritmo"]]

print(f"Mejor RNA: {mejor['Topologia']} + {mejor['Algoritmo']}")
print(f"  R2 validacion   = {mejor['R2_val']}")
print(f"  MAE validacion  = {mejor['MAE_val']}")
print(f"  RMSE validacion = {mejor['RMSE_val']}")


# =============================================================================
# 5. COMPARACION FINAL SOBRE EL CONJUNTO DE PRUEBA
# =============================================================================
print("\n" + "=" * 70)
print("5. COMPARACION CONTRA REGRESION LINEAL Y RANDOM FOREST")
print("=" * 70)

# --- Mejor RNA ---
rna = MLPRegressor(
    hidden_layer_sizes=mejores_capas,
    max_iter=5000,
    random_state=SEMILLA,
    **mejores_params,
)
rna.fit(X_train_e, y_train)
pred_rna = rna.predict(X_test_e)

# --- Regresion lineal ---
lineal = LinearRegression()
lineal.fit(X_train_e, y_train)
pred_lin = lineal.predict(X_test_e)

# --- Random Forest (los arboles no requieren escalado) ---
bosque = RandomForestRegressor(n_estimators=200, random_state=SEMILLA)
bosque.fit(X_train, y_train)
pred_rf = bosque.predict(X_test)

comparacion = pd.DataFrame(
    [
        metricas(f"RNA {mejor['Topologia']} + {mejor['Algoritmo']}",
                 y_test, pred_rna),
        metricas("Regresion lineal", y_test, pred_lin),
        metricas("Random Forest", y_test, pred_rf),
    ]
)

print("\nDesempeno en el conjunto de PRUEBA:")
print(comparacion.to_string(index=False))
comparacion.to_csv("comparacion_final.csv", index=False)

# Importancia de variables segun Random Forest (util para el analisis)
importancias = (
    pd.Series(bosque.feature_importances_, index=ENTRADAS)
    .sort_values(ascending=False)
    .round(4)
)
print("\nImportancia de las variables (Random Forest):")
print(importancias.to_string())


# =============================================================================
# 6. GRAFICAS
# =============================================================================
print("\n" + "=" * 70)
print("6. GENERACION DE GRAFICAS")
print("=" * 70)

# --- Figura 1: valores reales vs. valores predichos ---
fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
modelos = [
    (pred_rna, f"RNA {mejor['Topologia']} + {mejor['Algoritmo']}"),
    (pred_lin, "Regresion lineal"),
    (pred_rf, "Random Forest"),
]

for eje, (pred, titulo) in zip(ejes, modelos):
    eje.scatter(y_test, pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    limites = [y_test.min(), y_test.max()]
    eje.plot(limites, limites, "r--", linewidth=1.5, label="Prediccion ideal")
    eje.set_xlabel("Carga real (kWh/m2)")
    eje.set_ylabel("Carga predicha (kWh/m2)")
    eje.set_title(f"{titulo}\nR2 = {r2_score(y_test, pred):.4f}")
    eje.legend(loc="upper left", fontsize=8)
    eje.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("reales_vs_predichos.png", dpi=150)
plt.close()
print("  guardada: reales_vs_predichos.png")

# --- Figura 2: evolucion del error durante el entrenamiento ---
# Nota: L-BFGS no expone curvas de error por epoca. Cuando el mejor
# algoritmo no las permite, se entrena la misma topologia con Adam
# unicamente para ilustrar el proceso de convergencia.
if mejor["Algoritmo"] == "L-BFGS":
    print("  nota: L-BFGS no genera curvas por epoca; se ilustra con Adam")

red_curva = MLPRegressor(
    hidden_layer_sizes=mejores_capas,
    solver="adam",
    max_iter=2000,
    random_state=SEMILLA,
    early_stopping=True,
    validation_fraction=0.25,
)
red_curva.fit(X_train_e, y_train)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

ax1.plot(red_curva.loss_curve_, color="tab:blue")
ax1.set_xlabel("Epoca")
ax1.set_ylabel("Error (MSE)")
ax1.set_title("Error de entrenamiento")
ax1.set_yscale("log")
ax1.grid(alpha=0.3)

ax2.plot(red_curva.validation_scores_, color="tab:orange")
ax2.set_xlabel("Epoca")
ax2.set_ylabel("R2")
ax2.set_title("Desempeno en validacion")
ax2.set_ylim(0.8, 1.0)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("curva_entrenamiento.png", dpi=150)
plt.close()
print("  guardada: curva_entrenamiento.png")

print("\n" + "=" * 70)
print("PROCESO COMPLETADO")
print("=" * 70)
