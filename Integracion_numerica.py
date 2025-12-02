
import matplotlib.pyplot as plt

def integración_numerica(f,a,b,n): 	#Definición de función de integración numérica
    h=(b-a)/n 	#altura trapecio
    integral = (f(a)+f(b))/2.0 	#Se suman las funciones evaluadas en los límites y se dividen entre dos 

    for i in range(1,n):
        integral += func(a+i*h)
    integral *= h
    return integral

def f(x):
    return x**2 - 6
a=1
b=3

plt.plot(x,y)
plt.show()


