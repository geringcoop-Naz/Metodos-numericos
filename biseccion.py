"""
**   Autor :       Juan Carlos Quiroz Aguilar
**   Fecha :       02/21
**   Descripcion:  Metodo de biseccion para encontrar raices de funciones
**                 no necesariamente lineales

"""

import math as mt
import numpy as np

def biseccion(a,b):

    diferencia = 1
    tol = 1e-6

    while (diferencia>tol):      
        c = (a+b)/2       #valor medio
        d = fx(a)*fx(c)   #multiplicación funciones evaluadas en a y c
        if(d<0):      #si se encuentra en el primer intervalo se actualiza el limite final a el valor medio
            b = c
        else:
            d = fx(c)*fx(b)
            if (d < 0): #si la raiz se encuentra en el segundo intervalo se actualiza el limite inicial al valor medio
                a = c 
        diferencia = abs(a - b)
    return (a)

def fx(x):
    return mt.sin(x)

def main():
    raices = []     #Lista para guardar las raices

    a = 0
    b = np.pi

    if (fx(a) == 0):    #Si existe una raiz en los limites del intervalo las guardamos
        raices.append(a)
    if (fx(b) == 0):
        raices.append(b)

    i = a + 0.1

    #El cliclo se detiene cuando llegamos al limite superior del intervalo
    while (i <= b):
            r = biseccion(a,i)
            print("**********************************************************************\n*\t\t\t La raiz es: ", r
                  , "\n**********************************************************************")
            raices.append(r)

    a = round(i,3)
    i = round(a + 0.1,3)

    print("Raices -> ",raices)


# Ejecuta la funcion principal
if __name__ == '__main__':
    main()

