#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 11:44:53 2025

@author: nass
"""
import numpy as np


#1. Normalizar
a=[10,20,30,40,50]
a_norm = ((a - np.min() )/(np.max() - np.min()))
print(a_norm)

#2. Matriz 5x5
num=np.arange(1,26)
matriz=num.reshape(5,5)
print(matriz)

