import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ---------- Datos (misma particion) ----------
df = pd.read_excel("ENB2012_data.xlsx")
X = df[["X1","X2","X3","X4","X5","X6","X7","X8"]]
y = df["Y1"]

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42)

esc = StandardScaler()
X_train_e = esc.fit_transform(X_train)
X_val_e   = esc.transform(X_val)
X_test_e  = esc.transform(X_test)

def metricas(nombre, y_real, y_pred):
    return {
        "Modelo": nombre,
        "MAE":  round(mean_absolute_error(y_real, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_real, y_pred)), 4),
        "R2":   round(r2_score(y_real, y_pred), 4),
    }

# ---------- Mejor RNA ----------
mejor_red = MLPRegressor(hidden_layer_sizes=(30,), solver="lbfgs",
                         max_iter=5000, random_state=42)
mejor_red.fit(X_train_e, y_train)
pred_rna = mejor_red.predict(X_test_e)

# ---------- Regresion lineal ----------
lineal = LinearRegression()
lineal.fit(X_train_e, y_train)
pred_lin = lineal.predict(X_test_e)

# ---------- Random Forest ----------
bosque = RandomForestRegressor(n_estimators=200, random_state=42)
bosque.fit(X_train, y_train)          # no necesita escalado
pred_rf = bosque.predict(X_test)

comparacion = pd.DataFrame([
    metricas("RNA (30) L-BFGS", y_test, pred_rna),
    metricas("Regresion lineal", y_test, pred_lin),
    metricas("Random Forest",    y_test, pred_rf),
])

print("\n=== DESEMPENO EN CONJUNTO DE PRUEBA ===")
print(comparacion.to_string(index=False))
comparacion.to_csv("comparacion_final.csv", index=False)

# ---------- Grafica 1: reales vs predichos ----------
fig, ejes = plt.subplots(1, 3, figsize=(15, 4.5))
for eje, (pred, titulo) in zip(ejes, [
        (pred_rna, "RNA (30) L-BFGS"),
        (pred_lin, "Regresion lineal"),
        (pred_rf,  "Random Forest")]):
    eje.scatter(y_test, pred, alpha=0.6, edgecolors="k", linewidths=0.3)
    lims = [y_test.min(), y_test.max()]
    eje.plot(lims, lims, "r--", linewidth=1.5)
    eje.set_xlabel("Carga real (kWh/m2)")
    eje.set_ylabel("Carga predicha (kWh/m2)")
    eje.set_title(f"{titulo}\nR2 = {r2_score(y_test, pred):.4f}")
    eje.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("reales_vs_predichos.png", dpi=150)
print("\nGuardada: reales_vs_predichos.png")

# ---------- Grafica 2: curva de error (Adam, que si la permite) ----------
red_curva = MLPRegressor(hidden_layer_sizes=(30,), solver="adam",
                         max_iter=2000, random_state=42,
                         early_stopping=True, validation_fraction=0.25)
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
print("Guardada: curva_entrenamiento.png")