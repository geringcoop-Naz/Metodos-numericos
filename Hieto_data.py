#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 20:13:24 2025

@author: nass
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# === Cargar datos ===
# El archivo CSV debe tener columnas: fecha, precipitacion
df = pd.read_csv("/home/nass/Descargas/Hidrologia/Estaciones_climatológicas/PPC.csv")

# Número total de registros
n = len(df)

# Ordenar los datos de mayor a menor
df_ordenado = df.sort_values(by="PPC", ascending=False).reset_index(drop=True)

# Agregar columna de ranking (1 = valor más alto)
df_ordenado["Orden"] = df_ordenado.index + 1

# Calcular período de retorno con fórmula de Weibull
df_ordenado["Tr_Weibull"] = (n + 1) / df_ordenado["Orden"]

# Calcular variable reducida de Gumbel
df_ordenado["Z"] = -np.log(-np.log(1 - 1/df_ordenado["Tr_Weibull"]))

# === Ajuste lineal en espacio Gumbel ===
slope, intercept, _, _, _ = linregress(df_ordenado["Z"], df_ordenado["PPC"])
df_ordenado["P_ajustada"] = slope * df_ordenado["Z"] + intercept

# === Ajuste potencial P = a * Tr^b ===
x = np.log(df_ordenado["Tr_Weibull"])
y = np.log(df_ordenado["PPC"])
b, log_a, _, _, _ = linregress(x, y)   # pendiente = b, intercepto = log(a)
a = np.exp(log_a)
df_ordenado["P_potencial"] = a * df_ordenado["Tr_Weibull"]**b

# === Estimación de precipitaciones de diseño para Tr específicos ===
Tr_objetivos = [10, 25, 50, 100, 200]
P_pot = a * np.array(Tr_objetivos)**b
resumen = pd.DataFrame({"Tr (años)": Tr_objetivos, "Precipitación diseño (mm)": P_pot})

# === Guardar resultados en Excel con 2 hojas ===
with pd.ExcelWriter("precipitacion_analisis.xlsx", engine="openpyxl") as writer:
    df_ordenado.to_excel(writer, sheet_name="Datos procesados", index=False)
    resumen.to_excel(writer, sheet_name="Resumen diseño", index=False)

print("\nArchivo 'precipitacion_analisis.xlsx' generado con dos hojas:")
print(" - Hoja 1: Datos procesados")
print(" - Hoja 2: Resumen de diseño")

# === Gráfica Gumbel ===
plt.figure(figsize=(8,6))
plt.scatter(df_ordenado["Z"], df_ordenado["PPC"], color="blue", label="Datos observados")
plt.plot(df_ordenado["Z"], df_ordenado["P_ajustada"], color="red", label="Recta de ajuste Gumbel")
plt.xlabel("Variable reducida de Gumbel (Z)")
plt.ylabel("Precipitación (mm)")
plt.title("Análisis de Frecuencia - Ajuste Gumbel")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# === Gráfica P vs Tr (ajuste potencial) ===
plt.figure(figsize=(8,6))
plt.scatter(df_ordenado["Tr_Weibull"], df_ordenado["PPC"], color="blue", label="Datos observados")
plt.plot(Tr_objetivos, P_pot, color="purple", marker="o", label=f"Ajuste potencial: P = {a:.2f}·Tr^{b:.2f}")
plt.xscale("log")
plt.xlabel("Periodo de Retorno Tr (años)")
plt.ylabel("Precipitación (mm)")
plt.title("Curva Precipitación vs Periodo de Retorno (Ajuste Potencial)")
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.legend()
plt.show()
