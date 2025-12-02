#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  7 09:15:17 2025

@author: nass
"""

import numpy as np
import matplotlib.pyplot as plt

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    """
    Método de Newton-Raphson para encontrar raíces de una función.

    Parámetros:
    f: función para la cual se busca la raíz
    df: derivada de la función f
    x0: aproximación inicial
    tol: tolerancia (precisión)
    max_iter: número máximo de iteraciones

    Retorna:
    x: aproximación de la raíz
    iteraciones: número de iteraciones realizadas
    historia: lista con el historial de aproximaciones
    """
    x = x0
    historia = [x0]
    
    for iteraciones in range(1, max_iter + 1):
        fx = f(x)
        if abs(fx) < tol:
            break
            
        dfx = df(x)
        if dfx == 0:
            raise ValueError("Derivada cero. No se puede continuar.")
            
        x_nuevo = x - fx / dfx
        historia.append(x_nuevo)
        
        if abs(x_nuevo - x) < tol:
            break
            
        x = x_nuevo
    else:
        iteraciones = max_iter
        
    return x, iteraciones, historia

# Definir la función y su derivada
def f(x):
    return x**2 - 2*x - 5

def df(x):
    return 2*x - 2

# Parámetros iniciales
x0 = 2.0
tolerancia = 1e-6
max_iteraciones = 100

# Aplicar el método
raiz, iteraciones, historia = newton_raphson(f, df, x0, tolerancia, max_iteraciones)

print(f"Raíz aproximada: {raiz:.6f}")
print(f"Número de iteraciones: {iteraciones}")

# Crear gráfica
x_vals = np.linspace(min(historia)-1, max(historia)+1, 400)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x)')
plt.axhline(y=0, color='k', linestyle='--', alpha=0.7)
plt.scatter(historia, [f(x) for x in historia], c='red', alpha=0.7, label='Iteraciones')
plt.plot(historia, [f(x) for x in historia], 'r--', alpha=0.5)

# Anotar las iteraciones
for i, (x, y) in enumerate(zip(historia, [f(x) for x in historia])):
    plt.annotate(f'{i}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Método de Newton-Raphson')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()