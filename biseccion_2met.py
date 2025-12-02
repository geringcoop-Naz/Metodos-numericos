#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 20:42:28 2025

@author: nass
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

def biseccion(f, a, b, tolerancia=1e-6, max_iteraciones=100):
  
    if f(a) * f(b) >= 0:
        print("El método de bisección no funciona en este intervalo. La función no cambia de signo.")
        return None

    for _ in range(max_iteraciones): 
        m = (a + b) / 2         # Calcula el punto medio
        f_m = f(m)

        if abs(f_m) < tolerancia:  # Verifica condición de raíz
            return m

        if f(a) * f_m < 0:      # Actualiza intervalo
            b = m
        else:        # Actualiza intervalo
            a = m

    print("Se alcanzó el número máximo de iteraciones.")
    return m


def f(x):
    return x**3 - x - 2       # Se define la función f(x) = x^3 - x - 2

# Se establece el intervalo inicial y se llama a la función
a = 1
b = 2
raiz = biseccion(f, a, b) 

if raiz is not None:
    print(f"La raíz aproximada es: {raiz}")
    

#Bisección optimize
    x=np.linspace(a,b,100)
    y=f(x)
    y0=0.0
    x0=optimize.bisect(f, a, b)

    plt.plot(x,y, label='sin(x)', color = 'blue')
    plt.axhline(0, color = 'grey', linewidth=1)
    plt.scatter(a, f(a))
    plt.scatter(b, f(b))
    

    print(f"La raíz con optimize es: {x0}")
    plt.show()
    
    
