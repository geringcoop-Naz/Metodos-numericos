#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 08:31:39 2025

@author: nass
"""

"""3.2 - Métodos cerrados
Definición 3.2.1 — Método cerrados. Los métodos cerrados requieren de dos puntos x0 y x1 que encierren la raíz. Se debe cumplir que f(x0) f (x1) < 0. Sólo localizan las raíces reales que cruzan el eje x.
3.2.1 - Bisección, Bolzano o partición binaria
El método de bisección, al ser un método cerrado, requiere de dos puntos x0 y x1 que encierren la raíz, el intervalo [x0 , x1] se parte a la mitad, generando 2 subintervalos [x0, x] y [x, x1], en uno de los dos se encuentra la raíz, se selecciona dicho subintervalo y se vuelve a repetir el procedimiento hasta que se cumpla el criterio de convergencia |f(x)| < tolerancia.

Mitad del intervalo
x= (x_i + x_i+1)/2


Método de bisección
"""

from math import *
import numpy as np
import matplotlib.pyplot as plt

# Algoritmo de Biseccion
def biseccion (f , x0 , x1 , tol ) :
    if f (x0) * f (x1) > 0:
        raise ValueError ('La funcion no cambia de signo en este rango')

    sigue = False
        print ('{:^10s} {:^10s} {:^10s} {:^10s} {:^10s} {:^10s}'
format ('x0', 'x', 'x1', 'f (x0)', 'f(x)', 'f(x1)'))
	while (not sigue) :
	x = (x0 + x1) / 2
	print ('{:10.5 f} {:10.5 f} {:10.5 f} {:10.5 f} {:10.5f} {:10.5f}'.\
	format ( x0, x, x1, f(x0), f (x), f (x1)))
	if f (x0) * f(x) < 0:
	x1 = x
	else :
	x0 = x
	sigue = abs (f(x)) < tol
	return x

#Funcion a evaluar
	def f ( x ) :
	return x **2 -2

	def main () :
	# valores iniciales
	x0 = -1.6
	x1 = -1.4

tol =1 e -4
# Llamada al algoritmo
raiz = biseccion (f , x0 , x1 , tol )
print ( 'f ({: e }) ={: e } '. format ( raiz , f ( raiz ) ) )

x = np . linspace ( x0 , x1 ,100)
y=f(x)

fig = plt.figure ()
ax = plt.gca ()
plt.plot (x, y)
plt.scatter (raiz, f (raiz))
plt.text (raiz, f ( raiz ), 'Raíz' + str (raiz), color = 'red')
plt.grid ()

plt.show ()
# fig.savefig ("biseccion.pdf", bbox_inches=' tight')

if __name__ == " __main__ " : main ()
