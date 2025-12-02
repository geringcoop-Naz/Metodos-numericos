#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 10:23:40 2025

@author: nass
"""
import numpy as np
#import pandas as pd

#1. Array con números del 1 al 10
list1 = np.array([1,2,3,4,5,6,7,8,9,10])
print(list1)

#2. Array de 8 ceros
list2 = np.array([0,0,0,0,0,0,0,0])
print(list2)

#3. Matriz identidad de 3x3
list3 = np.identity(3)
print(list3)

#4. Array de pares 

#5. Sumar
a=np.array([1,2,3])
b=np.array([4,5,6])
c= a+b
print(c)

#6 Con [5,12,8,15,3,10] suma, promedio y desviación estándar
list6 = np.array ([5,12,8,15,3,10])
suma = list6.sum()
prom = list6.mean()
desvE = list6.std()

print(suma)
print(prom)
print(desvE)

#7 Celsius a Farenheit
