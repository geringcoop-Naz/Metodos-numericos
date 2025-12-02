#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:29:26 2025

@author: nass
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


def f(x):
    return np.sin(x)

#Bisección
x=np.linspace(0, 2*np.pi,100)
y=f(x)
y0=0.0
x0=optimize.bisect(f, 1, 5)

plt.plot(x,y, label='sin(x)', color = 'blue')
plt.axhline(0, color = 'grey', linewidth=1)
#plt.scatter(x0, y0)

print(x0)
plt.show()