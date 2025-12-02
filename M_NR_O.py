#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 08:36:09 2025

@author: nass
"""

#Programa 3.5. Método de Newton-Raphson de optimize (scipy)

# importar las funciones de la biblioteca math
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
#import math
#import cmath

# Funcion a evaluar
def f (x):
	return  np.sin(x)

def main ():
    x0 =0.01 # valores iniciales
    raiz = optimize.newton (f, x0) # Llamada al algoritmo
    print ('f ({:e}) ={:e}'.format (raiz, f (raiz)))

    x=np.linspace (0.0001 ,0.05 ,100)
    y=f(x)
    
   # fig = plt.figure ()
    plt.plot (x,y)
    plt.title ('Factor de fricción de Colebrook')
    plt.scatter (raiz, f (raiz))
    plt.text (raiz, f (raiz), 'Raíz'+ str (raiz), color = 'red')
    plt.grid ()

    plt.show ()
#fig.savefig ("newtonraphsonopt.pdf", bbox_inches = 'tight')

#if __name__ == " __main__ " : main ()
