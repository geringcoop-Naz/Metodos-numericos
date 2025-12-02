#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 12:24:33 2025

@author: jagc
"""

import numpy as np
import matplotlib.pyplot as plt

t0 = 0
tf = 10
k0 = 0.5
y0 = 1
h  = 0.1

n_steps = int((tf-t0)/h) + 1
print(n_steps)
y_aprox = np.zeros(n_steps)
y_aprox[0] = y0

for i in range(n_steps-1):
    y_aprox[i+1] = y_aprox[i] + y_aprox[i]*k0*h
    
t = np.linspace(t0,tf,200)
y_exac = y0*np.exp(k0*(t - t0))

plt.plot(t,y_exac)
plt.plot([i*h for i in range(n_steps)],y_aprox)