#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 14:46:21 2025

@author: nass
"""

def newton_raphson(f_derivada, f_segunda_derivada, x0, tol=1e-6, max_iter=100):
    """
    Encuentra un punto crítico de la función usando el método de Newton-Raphson.

    Parámetros:
    - f_derivada: Función que representa la primera derivada de f.
    - f_segunda_derivada: Función que representa la segunda derivada de f.
    - x0: Punto inicial para la iteración.
    - tol: Tolerancia para la convergencia.
    - max_iter: Número máximo de iteraciones.

    Retorna:
    - x: Aproximación del punto crítico.
    - tipo: "Mínimo" si f''(x) > 0, "Máximo" si f''(x) < 0, "Indeterminado" si f''(x) = 0.
    """
    x = x0
    for i in range(max_iter):
        f_prime = f_derivada(x)
        f_double_prime = f_segunda_derivada(x)
        
        if abs(f_prime) < tol:
            # Clasificar el punto crítico
            if f_double_prime > tol:
                return x, "Mínimo"
            elif f_double_prime < -tol:
                return x, "Máximo"
            else:
                return x, "Indeterminado"
        
        if abs(f_double_prime) < tol:
            raise ValueError("La segunda derivada es casi cero. El método puede no converger.")
        
        x = x - f_prime / f_double_prime
    
    raise RuntimeError("El método no convergió después de {} iteraciones.".format(max_iter))

# Ejemplo de uso:
if __name__ == "__main__":
    # Definir la función, su primera y segunda derivada
    def f(x):
        return x**3 - 2*x**2 + 2
    
    def f_derivada(x):
        return 3*x**2 - 4*x
    
    def f_segunda_derivada(x):
        return 6*x - 4

    # Puntos iniciales para buscar puntos críticos
    puntos_iniciales = [0.0, 1.5]
    
    for x0 in puntos_iniciales:
        try:
            punto_critico, tipo = newton_raphson(f_derivada, f_segunda_derivada, x0)
            print(f"Punto crítico en x = {punto_critico:.5f}, Tipo: {tipo}")
        except Exception as e:
            print(f"Error con x0={x0}: {e}")
            