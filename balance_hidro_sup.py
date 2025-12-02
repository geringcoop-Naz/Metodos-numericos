#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 18:35:01 2025

@author: nass
"""

import pandas as pd
import numpy as np
import glob
from datetime import datetime

def leer_datos_estaciones(ruta_estaciones):
    """
    Lee datos de todas las estaciones meteorológicas en la cuenca
    """
    archivos = glob.glob(ruta_estaciones)
    datos_estaciones = []
    
    for archivo in archivos:
        try:
            if archivo.endswith('.csv'):
                df = pd.read_csv(archivo)
            elif archivo.endswith('.txt'):
                df = pd.read_csv(archivo, delimiter='\t')  # Ajustar delimitador según necesidad
            else:
                continue
                
            # Convertir columna de fecha a datetime
            if 'Fecha' in df.columns:
                df['Fecha'] = pd.to_datetime(df['Fecha'])
                datos_estaciones.append(df)
        except Exception as e:
            print(f"Error leyendo {archivo}: {e}")
    
    return datos_estaciones

def leer_datos_caudal(ruta_caudal):
    """
    Lee datos de la estación hidrométrica
    """
    try:
        if ruta_caudal.endswith('.csv'):
            df = pd.read_csv(ruta_caudal)
        elif ruta_caudal.endswith('.txt'):
            df = pd.read_csv(ruta_caudal, delimiter='\t')
        
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'])
        return df
    except Exception as e:
        print(f"Error leyendo datos de caudal: {e}")
        return None

def calcular_precipitacion_media(datos_estaciones):
    """
    Calcula precipitación media acumulada usando método aritmético
    """
    precipitaciones = []
    
    for estacion in datos_estaciones:
        if 'Precipitacion' in estacion.columns:
            # Agrupar por fecha y sumar precipitación diaria
            precip_diaria = estacion.groupby('Fecha')['Precipitacion'].sum()
            precipitaciones.append(precip_diaria)
    
    if not precipitaciones:
        return None
    
    # Combinar todas las estaciones y calcular promedio
    precip_combinada = pd.concat(precipitaciones, axis=1)
    precip_combinada.columns = [f'Estacion_{i}' for i in range(len(precip_combinada.columns))]
    precip_media = precip_combinada.mean(axis=1)
    
    return precip_media

def calcular_evaporacion_penman_monteith(datos_estaciones, precip_media):
    """
    Calcula evapotranspiración de referencia usando Penman-Monteith
    """
    # Esta es una implementación simplificada - ajustar según datos disponibles
    evaporaciones = []
    
    for estacion in datos_estaciones:
        if all(col in estacion.columns for col in ['Tmax', 'Tmin', 'Humedad', 'Viento', 'Radiacion']):
            # Calcular ETo usando Penman-Monteith simplificado
            estacion['Tmedia'] = (estacion['Tmax'] + estacion['Tmin']) / 2
            
            # Fórmula simplificada de Penman-Monteith (ajustar según necesidades específicas)
            # Esta es una versión básica - se debe implementar la fórmula completa
            delta = 4098 * (0.6108 * np.exp(17.27 * estacion['Tmedia'] / (estacion['Tmedia'] + 237.3))) / ((estacion['Tmedia'] + 237.3) ** 2)
            gamma = 0.665 * 0.001 * 101.3  # Constante psicrométrica simplificada
            
            # Cálculo simplificado - REEMPLAZAR con fórmula completa de Penman-Monteith
            eto = (0.408 * delta * estacion['Radiacion'] + gamma * (900 / (estacion['Tmedia'] + 273)) * 
                   estacion['Viento'] * (0.6108 * np.exp(17.27 * estacion['Tmedia'] / (estacion['Tmedia'] + 237.3)) - 
                   (estacion['Humedad']/100) * 0.6108 * np.exp(17.27 * estacion['Tmedia'] / (estacion['Tmedia'] + 237.3)))) / \
                  (delta + gamma * (1 + 0.34 * estacion['Viento']))
            
            evaporacion_diaria = estacion.groupby('Fecha')[eto.name].sum()
            evaporaciones.append(evaporacion_diaria)
    
    if not evaporaciones:
        return calcular_evaporacion_simple(datos_estaciones, precip_media)
    
    # Combinar y promediar evaporación de todas las estaciones
    evap_combinada = pd.concat(evaporaciones, axis=1)
    evap_combinada.columns = [f'Estacion_{i}' for i in range(len(evap_combinada.columns))]
    evap_media = evap_combinada.mean(axis=1)
    
    return evap_media

def calcular_evaporacion_simple(datos_estaciones, precip_media):
    """
    Calcula evaporación usando datos medidos directamente
    """
    evaporaciones = []
    
    for estacion in datos_estaciones:
        if 'Evaporacion' in estacion.columns:
            evap_diaria = estacion.groupby('Fecha')['Evaporacion'].sum()
            evaporaciones.append(evap_diaria)
    
    if not evaporaciones:
        return None
    
    evap_combinada = pd.concat(evaporaciones, axis=1)
    evap_combinada.columns = [f'Estacion_{i}' for i in range(len(evap_combinada.columns))]
    evap_media = evap_combinada.mean(axis=1)
    
    return evap_media

def aplicar_condiciones_evaporacion(evaporacion, precipitacion, factor_reduccion=0.5):
    """
    Aplica condiciones: solo días con precipitación y factor de reducción
    """
    # Crear máscara para días con precipitación
    mascara_precipitacion = precipitacion > 0
    
    # Aplicar condiciones
    evaporacion_ajustada = evaporacion.copy()
    evaporacion_ajustada[~mascara_precipitacion] = 0  # Anular evaporación sin precipitación
    evaporacion_ajustada[mascara_precipitacion] *= factor_reduccion  # Aplicar factor de reducción
    
    return evaporacion_ajustada

def calcular_caudal_equivalente(datos_caudal, area_cuenca):
    """
    Convierte caudal a lámina equivalente en mm
    """
    if 'Caudal' not in datos_caudal.columns:
        return None
    
    # Calcular caudal promedio diario
    caudal_diario = datos_caudal.groupby('Fecha')['Caudal'].mean()
    
    # Convertir m³/s a mm/día
    # 1 m³/s = 86400 m³/día
    # Lámina (mm) = Volumen (m³) / Área (m²) * 1000
    caudal_mm = (caudal_diario * 86400) / (area_cuenca * 1e6) * 1000
    
    return caudal_mm

def balance_hidrologico(precipitacion, evaporacion, caudal, factor_infiltracion=0.35):
    """
    Realiza el balance hidrológico completo
    """
    # Asegurar que todas las series tengan las mismas fechas
    fechas_comunes = precipitacion.index.intersection(evaporacion.index).intersection(caudal.index)
    
    P = precipitacion[fechas_comunes].sum()
    E = evaporacion[fechas_comunes].sum()
    R = caudal[fechas_comunes].sum()
    
    # Balance: P = E + I + R + ΔS
    # Asumiendo ΔS = 0 para año normal
    infiltracion = P - E - R
    
    # Calcular recarga del acuífero
    recarga = infiltracion * factor_infiltracion
    
    return {
        'Precipitacion_anual_mm': P,
        'Evaporacion_anual_mm': E,
        'Escorrentia_anual_mm': R,
        'Infiltracion_mm': infiltracion,
        'Recarga_acuifero_mm': recarga
    }

def main():
    """
    Función principal que ejecuta todo el balance hidrológico
    """
    # Configuración - AJUSTAR ESTOS PARÁMETROS
    ruta_estaciones = "estaciones/*.csv"  # Patrón para archivos de estaciones
    ruta_caudal = "caudal.csv"  # Archivo de datos de caudal
    area_cuenca = 1000  # Área de la cuenca en km² - AJUSTAR
    
    # Factores ajustables
    factor_evaporacion = 0.5  # 50% de reducción
    factor_infiltracion = 0.35  # 35% de infiltración recarga acuífero
    metodo_evaporacion = 'penman'  # 'simple' o 'penman'
    
    # Leer datos
    print("Leyendo datos de estaciones meteorológicas...")
    datos_estaciones = leer_datos_estaciones(ruta_estaciones)
    
    print("Leyendo datos de caudal...")
    datos_caudal = leer_datos_caudal(ruta_caudal)
    
    if not datos_estaciones or datos_caudal is None:
        print("Error: No se pudieron leer los datos necesarios")
        return
    
    # Calcular precipitación media
    print("Calculando precipitación media...")
    precipitacion_media = calcular_precipitacion_media(datos_estaciones)
    
    if precipitacion_media is None:
        print("Error: No se encontraron datos de precipitación")
        return
    
    # Calcular evaporación según método seleccionado
    print("Calculando evaporación...")
    if metodo_evaporacion == 'penman':
        evaporacion_media = calcular_evaporacion_penman_monteith(datos_estaciones, precipitacion_media)
    else:
        evaporacion_media = calcular_evaporacion_simple(datos_estaciones, precipitacion_media)
    
    if evaporacion_media is None:
        print("Error: No se pudieron calcular datos de evaporación")
        return
    
    # Aplicar condiciones a evaporación
    evaporacion_ajustada = aplicar_condiciones_evaporacion(
        evaporacion_media, precipitacion_media, factor_evaporacion
    )
    
    # Calcular caudal equivalente
    print("Calculando escorrentía...")
    caudal_mm = calcular_caudal_equivalente(datos_caudal, area_cuenca)
    
    if caudal_mm is None:
        print("Error: No se pudieron calcular datos de caudal")
        return
    
    # Realizar balance hidrológico
    print("Realizando balance hidrológico...")
    resultados = balance_hidrologico(
        precipitacion_media, 
        evaporacion_ajustada, 
        caudal_mm, 
        factor_infiltracion
    )
    
    # Mostrar resultados
    print("\n=== RESULTADOS DEL BALANCE HIDROLÓGICO ===")
    for key, value in resultados.items():
        print(f"{key}: {value:.2f}")
    
    return resultados

if __name__ == "__main__":
    resultados = main()