#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 22:08:21 2025

@author: nass
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt

# Cargar datos de calibración existentes
data = pd.read_csv('optimizacion_nash_TETIS.csv', index_col=0)

# Eliminar filas vacías y la fila de descripción
data = data.dropna()
data = data[data.index != '']

print("Datos de calibración existentes:")
print(data)

# Definir límites para los parámetros FC1-FC9 basados en los datos existentes
bounds = [
    (0.8, 50),      # FC1 - Almacenamiento estático
    (0, 1),         # FC2 - Evapotranspiración
    (0, 0.2),       # FC3 - Infiltración
    (0.1, 10),      # FC4 - Escorrentía directa
    (0, 0.5),       # FC5 - Percolación
    (0, 12),        # FC6 - Interflujo
    (0, 7),         # FC7 - Pérdidas subterráneas
    (0, 600),       # FC8 - Flujo base
    (0, 1)          # FC9 - Velocidad de onda
]

def funcion_objetivo(params, estacion_referencia=None):
    """
    Función objetivo que calcula el negativo del NSE
    (para minimización en la optimización)
    """
    # Si no se proporciona estación de referencia, usar valores promedio
    if estacion_referencia is None:
        # Calcular NSE basado en similitud con parámetros existentes
        nse_values = []
        for idx, row in data.iterrows():
            # Calcular distancia euclidiana entre parámetros
            dist = np.sqrt(np.sum([(params[i] - row[f'FC{i+1}'])**2 for i in range(9)]))
            # Convertir distancia a similitud (inversamente proporcional)
            similitud = 1 / (1 + dist)
            nse_values.append(similitud * row['NSE'])
        
        # Promedio ponderado por similitud
        if nse_values:
            return -np.mean(nse_values)
        else:
            return -0.5  # Valor por defecto si no hay datos
    
    else:
        # Optimizar para parecerse a una estación específica
        if estacion_referencia in data.index:
            ref_row = data.loc[estacion_referencia]
            dist = np.sqrt(np.sum([(params[i] - ref_row[f'FC{i+1}'])**2 for i in range(9)]))
            return dist  # Minimizar distancia
        else:
            return funcion_objetivo(params)

def optimizar_parametros(estacion_objetivo=None, num_ejecuciones=3):
    """
    Optimiza los parámetros FC1-FC9 usando evolución diferencial
    """
    mejores_resultados = []
    
    for i in range(num_ejecuciones):
        print(f"\nEjecución de optimización {i+1}/{num_ejecuciones}")
        
        resultado = differential_evolution(
            funcion_objetivo,
            bounds,
            args=(estacion_objetivo,),
            strategy='best1bin',
            maxiter=100,
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=i
        )
        
        nse_estimado = -resultado.fun if estacion_objetivo is None else 1/(1+resultado.fun)
        mejores_resultados.append({
            'params': resultado.x,
            'nse': nse_estimado,
            'success': resultado.success
        })
        
        print(f"Parámetros optimizados: {resultado.x}")
        print(f"NSE estimado: {nse_estimado:.4f}")
        print(f"Optimización exitosa: {resultado.success}")
    
    # Seleccionar el mejor resultado
    mejor = max(mejores_resultados, key=lambda x: x['nse'])
    return mejor

def generar_recomendaciones(parametros_optimizados):
    """
    Genera recomendaciones basadas en los parámetros optimizados
    """
    recomendaciones = []
    
    nombres_fc = [
        "FC1 - Almacenamiento estático",
        "FC2 - Evapotranspiración", 
        "FC3 - Infiltración",
        "FC4 - Escorrentía directa",
        "FC5 - Percolación",
        "FC6 - Interflujo",
        "FC7 - Pérdidas subterráneas",
        "FC8 - Flujo base",
        "FC9 - Velocidad de onda"
    ]
    
    # Analizar cada parámetro
    for i, (param, nombre) in enumerate(zip(parametros_optimizados, nombres_fc)):
        valor = param
        
        if i == 0:  # FC1
            if valor > 25:
                recomendaciones.append("Alto almacenamiento estático - cuenca con buena capacidad de retención")
            elif valor < 5:
                recomendaciones.append("Bajo almacenamiento estático - respuesta rápida a la precipitación")
        
        elif i == 2:  # FC3 - Infiltración
            if valor < 0.05:
                recomendaciones.append("Baja infiltración - suelos poco permeables o saturados")
            elif valor > 0.15:
                recomendaciones.append("Alta infiltración - suelos permeables")
        
        elif i == 3:  # FC4 - Escorrentía directa
            if valor > 5:
                recomendaciones.append("Alta escorrentía directa - respuesta rápida del sistema")
        
        elif i == 8:  # FC9 - Velocidad de onda
            if valor < 0.3:
                recomendaciones.append("Baja velocidad de traslación - cauces con alta resistencia")
            elif valor > 0.7:
                recomendaciones.append("Alta velocidad de traslación - cauces eficientes")
    
    return recomendaciones

# Ejecutar optimización para diferentes escenarios
print("\n" + "="*60)
print("OPTIMIZACIÓN DE PARÁMETROS DEL MODELO TETIS")
print("="*60)

# Escenario 1: Optimización general
print("\n1. OPTIMIZACIÓN GENERAL:")
mejor_general = optimizar_parametros()
print(f"\nMEJOR RESULTADO GENERAL:")
print(f"NSE estimado: {mejor_general['nse']:.4f}")
print("Parámetros optimizados:")
for i, param in enumerate(mejor_general['params'], 1):
    print(f"  FC{i}: {param:.4f}")

# Escenario 2: Optimización similar a estación Q004 (mejor NSE)
print("\n2. OPTIMIZACIÓN SIMILAR A ESTACIÓN Q004 (NSE=0.9947):")
mejor_q004 = optimizar_parametros(estacion_objetivo='Q004')

# Escenario 3: Optimización similar a estación Q007 (parámetros completos)
print("\n3. OPTIMIZACIÓN SIMILAR A ESTACIÓN Q007:")
mejor_q007 = optimizar_parametros(estacion_objetivo='Q007')

# Generar recomendaciones
print("\n" + "="*60)
print("RECOMENDACIONES HIDROLÓGICAS")
print("="*60)
recomendaciones = generar_recomendaciones(mejor_general['params'])
for i, rec in enumerate(recomendaciones, 1):
    print(f"{i}. {rec}")

# Crear resumen comparativo
print("\n" + "="*60)
print("RESUMEN COMPARATIVO")
print("="*60)

resumen = pd.DataFrame({
    'Optimización General': mejor_general['params'],
    'Similar Q004': mejor_q004['params'],
    'Similar Q007': mejor_q007['params']
}, index=[f'FC{i}' for i in range(1, 10)])

resumen['NSE_Estimado'] = [
    mejor_general['nse'],
    mejor_q004['nse'], 
    mejor_q007['nse']
]

print(resumen.round(4))

# Gráfico comparativo de parámetros
plt.figure(figsize=(12, 8))
x = np.arange(9)
width = 0.25

plt.bar(x - width, mejor_general['params'], width, label='Optimización General', alpha=0.8)
plt.bar(x, mejor_q004['params'], width, label='Similar Q004', alpha=0.8)
plt.bar(x + width, mejor_q007['params'], width, label='Similar Q007', alpha=0.8)

plt.xlabel('Parámetros FC')
plt.ylabel('Valor del Parámetro')
plt.title('Comparación de Parámetros Optimizados')
plt.xticks(x, [f'FC{i+1}' for i in range(9)])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('comparacion_parametros_optimizados.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n¡Optimización completada! Los resultados han sido guardados.")