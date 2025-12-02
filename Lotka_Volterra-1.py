#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 13:59:11 2025

@author: jagc
"""

import numpy as np
import matplotlib.pyplot as plt

alfa = 0
beta = 0.5
delta = 1
gama = 2

t0 = 0
tf = 100
h = 0.01
n_steps = int((tf -t0)/h) + 1
t = np.array([i*h for i in range(n_steps)])

x0 = 3
y0 = 1

Y0 = np.array([x0,y0])

def Lotka_Volterra(xi,yi):
    dxdt = h*(alfa*xi - beta*xi*yi)
    dydt = h*(-delta*yi+gama*xi*yi)
    return np.array([dxdt,dydt])

x = np.zeros(n_steps)
y = np.zeros(n_steps)
#k1 = np.array([0,0])
x[0] = x0
y[0] = y0
for i in range(n_steps-1):
    k1 = Lotka_Volterra(x[i], y[i])
    k2 = Lotka_Volterra(x[i]+k1[0]/2, y[i]+k1[1]/2)
    k3 = Lotka_Volterra(x[i]+k2[0]/2, y[i]+k2[1]/2)
    k4 = Lotka_Volterra(x[i]+k3[0]/2, y[i]+k3[1]/2)
    x[i+1] = x[i] + (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])/6
    y[i+1] = y[i] + (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])/6

plt.plot(t,x)
plt.plot(t,y)
plt.ylim(-1,20)