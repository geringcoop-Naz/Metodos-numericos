#Integrales
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

def f(x):
    return np.sin(x)

def trapecio(a,b,n):
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    integral_trap = h * (np.sum(y) - (y[0] + y[-1]) / 2)
    return integral_trap
# Comparación con la solución exacta
resultado_trapecio = trapecio(0, np.pi, 1000)
resultado_real, error_quad = quad(f, 0, np.pi)
print(f"Integral numérica: {resultado_trapecio:.20f}, Exacta: {resultado_real:.10f}")
print(f"Error de quad: {error_quad:.20f}")

#Integral con integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
def (f(x)):
    return np.sin(x)
    
def integral_rect_izq(f,a,b,n):
    dx=(b-a)/n
    x = np.arange(a,b,dx)
    y=f(x)
    integral=dx*np.sum(y)
    plt.bar(x,y,width=dx)
    x_fine=np.linspace(a,b,100)
    y_fine = f(x_fine)
    plt.plot(x_fine,y_fine)
    plt.show()
    resulquad,erroquad=integrate.quad(f,a,b)
    print(resulquad)
    return integral
 
 
#Bisección con optimize 
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


