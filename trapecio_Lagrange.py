#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 13:21:47 2025

@author: jagc
"""

import numpy as np

def f(x):
    return x**2

a, b = 0, 1
#un trapecio (x_{k+1}-x_k)*(y_{k+1}+y_k)/2
def area_trapecio(xi,xf,f):
    return (xf-xi)*(f(xf)+f(xi))/2

x = np.linspace(a,b,1001)
area_k = []
for i in range(len(x)-1):
    area_k.append(area_trapecio(x[i], x[i+1], f))
    print(i,x[i],x[i+1],area_k[i])

resultado = np.sum(area_k[:-1])

print(f"El área es {resultado}")
