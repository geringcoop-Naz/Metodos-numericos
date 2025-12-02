#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 12:53:24 2025

@author: nass
"""

import time
import numpy as np

#Definimos un tamaño grande para la colección de arrays puros
n_elementos =100
lista_py = list(range(n_elementos))
t_inicial=time.time 
lista_cuadrado = [i**2 for i in lista_py]
t_final = time.time()
print(f"El tiempo es: {t_final - t_inicial}")

lista_np=np.array(lista_py)
t_inicial=time.time()
lista_cuadrado_np=np.array(lista_py)**2
t_final=time.time()
tiempo_numpy=t_final-t_inicial
print(f"El tiempo es: {tiempo_numpy}")

lista1=[4,5,2,7,4,2,9]
lista2=[9,7,4,5,8,3,1]
lista_new_np=lista1_np+lista2_np

lista=[1.3,4.5,7.7,9.12,3.8,8.6]
umbral=5.1
lista_new=[]
for i in lista:
    if i·•


lista_np=np.array([4,5,2,7,4,2,9])
2*lista_np