#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 14:59:41 2025

@author: nass
"""

import numpy as np
import matplotlib.pyplot as plt

# Datos
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(-x)

# Crear figura
plt.figure(figsize=(8, 6))

# Dibujar todas las funciones en el mismo gráfico
plt.plot(x, y1, label="sin(x)", linestyle="-", color="b")
plt.plot(x, y2, label="cos(x)", linestyle="--", color="r")
plt.plot(x, y3, label="tan(x)", linestyle=":", color="g")
plt.plot(x, y4, label="exp(-x)", linestyle="-.", color="m")

# Configuración del gráfico
plt.title("Múltiples Funciones en el mismo gráfico")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()  # Agregar leyenda
plt.grid(True)  # Agregar cuadrícula

# Mostrar gráfico
plt.show()

# Crear datos
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)
y4 = np.exp(-x)

# Crear figura con 2 filas y 2 columnas
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Agregar gráficos
axes[0, 0].plot(x, y1, color='b', label='sin(x)')
axes[0, 1].plot(x, y2, color='r', label='cos(x)')
axes[1, 0].plot(x, y3, color='g', label='tan(x)')
axes[1, 1].plot(x, y4, color='m', label='exp(-x)')

# Títulos y leyendas
axes[0, 0].set_title("Seno")
axes[0, 1].set_title("Coseno")
axes[1, 0].set_title("Tangente")
axes[1, 1].set_title("Exponencial")
for ax in axes.flat:
    ax.legend()
    ax.grid(True)

plt.tight_layout()  # Ajustar espacios automáticamente
plt.show()

# Definir la función a graficar
def f(x):
    return x**2 - 2  # Ejemplo de función cuadrática

def grafica_f():
    x = np.linspace(-4, 4, 100)
    y = f(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='f(x)')

    # Mover ejes al centro
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    # Quitar bordes superior y derecho
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')

    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.legend()

    plt.show()

grafica_f()