#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 12:33:47 2025

@author: nass
"""
from IPython.display import display
import numpy as np
import pandas as pd


#2. Array de 8 ceros
list2 = np.array([0,0,0,0,0,0,0,0])
print(list2)
print("\n" + "="*50 + "\n")

# 2. arange y matriz 3x4
list2 = np.arange(12)
matriz = list2.reshape(3,4)
print(matriz)
print("\n" + "="*50 + "\n")

#2. Matriz 5x5
num=np.arange(1,26)
matriz=num.reshape(5,5)
print(matriz)
print("\n" + "="*50 + "\n")

#2 Dataframe 
df = pd.DataFrame ({
" Nombre " : [ " Pao " , " Dan " , " Naz ", 'Oscar','Nacho' ] ,
" Edad " : [31, 37, 32, 25, 27]})

display(df)
forma = df.shape
cabeza = df.head(1)
cola = df.tail(1)

print(forma)
print("\n" + "="*50 + "\n")
print(cabeza)
print("\n" + "="*50 + "\n")
print(cola)
print("\n" + "="*50 + "\n")

#2.2 mayor de 30 y ordenar por ciudad y edad
datos = {
    'Nombre': ['Pao', 'Dan', 'Naz', 'Oscar', 'Nacho'],
    'Edad': [31, 37, 32, 25, 27],
    'Ciudad': ['Gto', 'Mx', 'Leon', 'Gto', 'Leon']
}

df = pd.DataFrame(datos)

# Filtrar mayores de 30 y ordenar por Ciudad y Edad
df_filtrado = df[df['Edad'] > 30].sort_values(by=['Ciudad', 'Edad'])
print("DataFrame filtrado (mayores de 30) y ordenado por Ciudad y Edad:")
print(df_filtrado)


