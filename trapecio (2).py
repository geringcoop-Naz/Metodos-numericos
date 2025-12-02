import numpy as np
from scipy.integrate import quad

def f(x):
    return np.sin(x)

def trapecio(a, b, n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    integral = h * (np.sum(y) - (y[0] + y[-1]) / 2)
    return integral

# Comparación con la solución exacta
resultado_trapecio = trapecio(0, np.pi, 1000)
resultado_real, error_quad = quad(f, 0, np.pi)
print(f"Integral numérica: {resultado_trapecio:.20f}, Exacta: {resultado_real:.10f}")
print(f"Error de quad: {error_quad:.20f}")









































