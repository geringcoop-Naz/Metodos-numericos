#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 11:52:04 2025

@author: nass
"""
from IPython.display import display
import pandas as pd


#2 Dataframe 
df = pd.DataFrame ({
" Nombre " : [ " Pao " , " Dan " , " Naz " ] ,
" Edad " : [31 , 37 , 32]})

display(df)
forma = df.shape
cabeza = df.head(1)
cola = df.tail(1)

print(forma)
print(cabeza)
print(cola)

#2.2 mayor de 30 y ordenar por ciudad y edad
df2 = pd . DataFrame ({
" Nombre " : [ " Pao " , " Dan " , " Naz " , " Oscar " ] ,
" Edad " : [31 , 37 , 32 , 27] ,
"Ciudad " : [ "Gto" , "MX" , " Leon " , " Gto. " ] ,
})

Es_Mayor_de_30 = df2.query("Edad >= 30 and Ciudad")
print(Es_Mayor_de_30)





