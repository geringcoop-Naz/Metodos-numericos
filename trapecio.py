import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def trapecio(a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    integral = h * (np.sum(y) - (y[0] + y[-1]) / 2)
    plt.plot(x,y,label='sin(x)', color = 'blue')
    plt.axhline(0, color = 'grey', linewidth=1)
    #plt.scatter(x0, y0)
    plt.show()
    print(x)
    return integral

#Grafica
   
   

# Comparación con la solución exacta
resultado_trapecio = trapecio(0, np.pi, 100)
resultado_real, error_quad = quad(f, 0, np.pi)
print(f"Integral numérica: {resultado_trapecio:.20f}, Exacta: {resultado_real:.10f}")
print(f"Error de quad: {error_quad:.20f}")










































