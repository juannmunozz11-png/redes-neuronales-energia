import pandas as pd

df = pd.read_excel("ENB2012_data.xlsx")

print("Dimensiones:", df.shape)
print("\nColumnas:", list(df.columns))
print("\nPrimeras filas:")
print(df.head())
print("\nValores faltantes por columna:")
print(df.isnull().sum())
print("\nEstadísticas:")
print(df.describe())