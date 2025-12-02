#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 13:28:38 2025

@author: nass
"""

import numpy as np
import matplotlib.pyplot as plt

t0 = 0
tf = 100
h = 0.01
n_steps = int((tf -t0)/h) + 1
t = np.array([i*h for i in range(n_steps)])

x0 = 3

def Lotka_Volterra(ti,xi):
    dxdt = 2*ti*xi + ti
    return dxdt

x = np.zeros(n_steps)
y = np.zeros(n_steps)
#k1 = np.array([0,0])
x[0] = x0
y[0] = y0
for i in range(n_steps-1):
    k1 = Lotka_Volterra(t[i], x[i])
    k2 = Lotka_Volterra(t[i]+h/2, x[i] + k1/2)
    k3 = Lotka_Volterra(x[i]+k2[0]/2, x[i]+ k2/2)
    k4 = Lotka_Volterra(x[i]+k3[0]/2, x[i]+ k3)
    x[i+1] = x[i] + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6
   

plt.plot(t,x)
plt.plot(t,y)
plt.ylim(-1,20)