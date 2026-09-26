import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 1. Cargar datos
df = pd.read_excel("ENB2012_data.xlsx")
X = df[["X1","X2","X3","X4","X5","X6","X7","X8"]]
y = df["Y1"]   # carga de calefaccion

# 2. Partir en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Escalar las variables
escalador = StandardScaler()
X_train_esc = escalador.fit_transform(X_train)
X_test_esc = escalador.transform(X_test)

# 4. Entrenar la red
red = MLPRegressor(hidden_layer_sizes=(10,), solver="adam",
                   max_iter=2000, random_state=42)
red.fit(X_train_esc, y_train)

# 5. Predecir y evaluar
y_pred = red.predict(X_test_esc)

print("R2   :", round(r2_score(y_test, y_pred), 4))
print("RMSE :", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("MAE  :", round(mean_absolute_error(y_test, y_pred), 4))
