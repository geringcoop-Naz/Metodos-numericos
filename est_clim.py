#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 28 21:20:23 2025

@author: nass
"""

import os
import re
import pandas as pd
from openpyxl import Workbook
from datetime import datetime

def procesar_archivo_txt(ruta_archivo):
    """Procesa un archivo TXT y extrae los datos de la estación meteorológica"""
    
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        lineas = f.readlines()
    
    # Extraer clave de la estación (primera línea)
    primera_linea = lineas[0].strip()
    clave_match = re.search(r'(\d+)', primera_linea)
    if clave_match:
        clave = clave_match.group(1)
    else:
        clave = "Desconocida"
    
    # Extraer coordenadas
    latitud = None
    longitud = None
    
    for linea in lineas:
        if 'LATITUD :' in linea:
            lat_match = re.search(r'LATITUD :\s*([-\d.]+)', linea)
            if lat_match:
                latitud = float(lat_match.group(1))
        elif 'LONGITUD :' in linea:
            lon_match = re.search(r'LONGITUD :\s*([-\d.]+)', linea)
            if lon_match:
                longitud = float(lon_match.group(1))
    
    # Procesar datos de precipitación
    datos_precipitacion = []
    datos_anuales = {}
    
    for linea in lineas[2:]:  # Saltar las primeras dos líneas (encabezados)
        partes = re.split(r'\s+', linea.strip())
        
        if len(partes) >= 2 and re.match(r'\d{4}-\d{2}-\d{2}', partes[0]):
            # Es una línea de datos diarios
            fecha_str = partes[0] + ' ' + partes[1]
            try:
                fecha = datetime.strptime(fecha_str, '%Y-%m-%d %H:%M:%S')
                precip = float(partes[2]) if partes[2] != '' else 0.0
                
                datos_precipitacion.append({
                    'Fecha': fecha,
                    'Precipitacion_mm': precip
                })
            except (ValueError, IndexError):
                continue
                
        elif len(partes) >= 2 and re.match(r'\d{4}', partes[0]):
            # Es una línea de datos anuales
            try:
                año = int(partes[0])
                p_max = float(partes[1]) if partes[1] != '' else 0.0
                datos_anuales[año] = p_max
            except (ValueError, IndexError):
                continue
    
    return {
        'clave': clave,
        'latitud': latitud,
        'longitud': longitud,
        'datos_diarios': datos_precipitacion,
        'datos_anuales': datos_anuales
    }

def crear_dataframe_estacion(datos_estacion):
    """Crea un DataFrame con los datos de la estación"""
    
    df_diarios = pd.DataFrame(datos_estacion['datos_diarios'])
    
    # Agregar columna de año
    if not df_diarios.empty:
        df_diarios['Año'] = df_diarios['Fecha'].dt.year
        
        # Combinar con datos anuales
        datos_anuales_lista = []
        for año, p_max in datos_estacion['datos_anuales'].items():
            datos_anuales_lista.append({'Año': año, 'P_Max': p_max})
        
        df_anuales = pd.DataFrame(datos_anuales_lista)
        
        # Combinar datos diarios y anuales
        df_final = pd.merge(df_diarios, df_anuales, on='Año', how='left')
        
        return df_final[['Fecha', 'Precipitacion_mm', 'Año', 'P_Max']]
    
    return pd.DataFrame()

def main():
    # Configuración
    carpeta_entrada = '/home/nass/Documentos/QGIS/Nasser/Estaciones_climatológicas/datos_estaciones'  # Cambia por tu carpeta
    archivo_salida = '/home/nass/Documentos/QGIS/Nasser/Estaciones_climatológicas/estaciones_meteorologicas.xlsx'
    
    # Buscar archivos TXT en la carpeta
    archivos_txt = [f for f in os.listdir(carpeta_entrada) if f.endswith('.txt')]
    
    if not archivos_txt:
        print("No se encontraron archivos TXT en la carpeta especificada.")
        return
    
    # Procesar cada archivo
    datos_estaciones = []
    resumen_estaciones = []
    
    for archivo in archivos_txt:
        ruta_completa = os.path.join(carpeta_entrada, archivo)
        print(f"Procesando: {archivo}")
        
        try:
            datos = procesar_archivo_txt(ruta_completa)
            datos_estaciones.append(datos)
            
            # Calcular precipitación máxima global
            if datos['datos_anuales']:
                año_max = max(datos['datos_anuales'], key=datos['datos_anuales'].get)
                precip_max = datos['datos_anuales'][año_max]
                
                resumen_estaciones.append({
                    'Clave': datos['clave'],
                    'Latitud': datos['latitud'],
                    'Longitud': datos['longitud'],
                    'Año_Precip_Max': año_max,
                    'Precip_Max_Global': precip_max
                })
            
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")
    
    # Crear archivo Excel
    with pd.ExcelWriter(archivo_salida, engine='openpyxl') as writer:
        # Hoja de resumen
        df_resumen = pd.DataFrame(resumen_estaciones)
        df_resumen.to_excel(writer, sheet_name='Resumen_Estaciones', index=False)
        
        # Hojas individuales para cada estación
        for datos in datos_estaciones:
            df_estacion = crear_dataframe_estacion(datos)
            if not df_estacion.empty:
                # Formatear la hoja como en el ejemplo
                nombre_hoja = datos['clave']
                df_estacion.to_excel(writer, sheet_name=nombre_hoja, index=False)
                
                # Obtener la hoja para formatear
                worksheet = writer.sheets[nombre_hoja]
                
                # Agregar encabezado con coordenadas (similar al ejemplo)
                worksheet['G1'] = 'LATITUD :'
                worksheet['H1'] = f"{datos['latitud']} °" if datos['latitud'] else 'N/A'
                worksheet['G2'] = 'LONGITUD :'
                worksheet['H2'] = f"{datos['longitud']} °" if datos['longitud'] else 'N/A'
    
    print(f"Proceso completado. Archivo guardado como: {archivo_salida}")
    print(f"Se procesaron {len(datos_estaciones)} estaciones")

if __name__ == "__main__":
    main()