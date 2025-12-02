#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 08:34:51 2025

@author: nass
"""

#Programa 3.4. Método de Newton-Raphson
# importar las funciones de la biblioteca math
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath

def newtonRaphson (f, df, x, imax=100, tol=1e-8): 
    cumple = False
    print ('{:^10 s} {:^10 s} {:^10 s}' .\
    format ('x', 'f (x)','df (x)'))
    k =0
    while (not cumple and k<imax):
        if df (x) !=0:
            x =x - f(x)/df(x)
        else:
            x = x + tol
    print ('{:10.5 f} {:10.5 f} {:10.5 f}' .\
    format (x, f (x), df (x)))
    cumple = abs (f(x)) <= tol
    k +=1
    if k < imax :
            return x
    else:
            raise ValueError ('La función no converge')

# Funcion a evaluar
def f (x):
    return x**2
# Derivada
def df (x):
    return 2*x

def main():
    x0 =0.01    # valores iniciales
    raiz = newtonRaphson (f, df, x0)    # Llamada al algoritmo
    print ('f({:e})={:e}'.format (raiz, f (raiz)))

    x = np.linspace (0.0001, 0.05, 100)
    y=f(x)

    fig = plt.figure ()
    plt.plot (x, y)
    plt.title ('Factor de fricción de Colebrook ')

    plt.scatter (raiz, f(raiz))
    plt.text ( raiz, f(raiz), 'Raíz'+ str (raiz), color='red')
    plt.grid ()

    plt.show ()
    fig.savefig ("newtonraphson.pdf", bbox_inches ='tight')

if __name__ == " __main__ " : main ()
