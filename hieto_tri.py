#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 02:03:10 2025

@author: nass
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class HietogramaTriangularCSV:
    """
    Clase para generar hietogramas de diseño triangulares a partir de un archivo CSV
    con datos de precipitación.
    """
    
    def __init__(self):
        self.datos = None
        self.parametros = {}
        
    def cargar_y_ordenar_datos(self, archivo_csv, columna_precipitacion, columna_anio=None):
        """
        Cargar y ordenar datos desde un archivo CSV
        
        Args:
            archivo_csv (str): Ruta al archivo CSV
            columna_precipitacion (str): Nombre de la columna con datos de precipitación
            columna_anio (str): Nombre de la columna con años (opcional)
        """
        try:
            # Cargar datos desde CSV
            self.datos = pd.read_csv(archivo_csv)
            
            # Verificar que la columna de precipitación existe
            if columna_precipitacion not in self.datos.columns:
                raise ValueError(f"La columna '{columna_precipitacion}' no existe en el archivo CSV")
            
            # Ordenar datos por año si se proporciona la columna
            if columna_anio and columna_anio in self.datos.columns:
                self.datos = self.datos.sort_values(by=columna_anio)
                print(f"Datos ordenados por la columna '{columna_anio}'")
            
            # Eliminar valores nulos en la columna de precipitación
            n_antes = len(self.datos)
            self.datos = self.datos.dropna(subset=[columna_precipitacion])
            n_despues = len(self.datos)
            
            if n_antes != n_despues:
                print(f"Se eliminaron {n_antes - n_despues} registros con valores nulos")
            
            print(f"Datos cargados correctamente: {len(self.datos)} registros")
            print(f"Columna de precipitación: '{columna_precipitacion}'")
            
            # Guardar nombres de columnas
            self.columna_precipitacion = columna_precipitacion
            self.columna_anio = columna_anio
            
            return True
            
        except Exception as e:
            print(f"Error al cargar el archivo: {e}")
            return False
    
    def analisis_frecuencia(self, periodo_retorno=100):
        """
        Realizar análisis de frecuencia para obtener la precipitación de diseño
        
        Args:
            periodo_retorno (int): Periodo de retorno en años
            
        Returns:
            float: Precipitación de diseño para el periodo de retorno especificado
        """
        try:
            # Obtener datos de precipitación
            precipitaciones = self.datos[self.columna_precipitacion].values
            
            # Ajustar distribución Gumbel (Extremos Tipo I)
            params_gumbel = stats.gumbel_r.fit(precipitaciones)
            
            # Calcular precipitación para el periodo de retorno
            prob = 1 - (1/periodo_retorno)
            precip_diseno = stats.gumbel_r.ppf(prob, *params_gumbel)
            
            # Guardar parámetros
            self.parametros['analisis_frecuencia'] = {
                'distribucion': 'Gumbel',
                'params_gumbel': params_gumbel,
                'periodo_retorno': periodo_retorno,
                'precipitacion_diseno': precip_diseno,
                'media': np.mean(precipitaciones),
                'desviacion_estandar': np.std(precipitaciones),
                'coeficiente_variacion': np.std(precipitaciones) / np.mean(precipitaciones),
                'maximo_observado': np.max(precipitaciones),
                'minimo_observado': np.min(precipitaciones)
            }
            
            print(f"Precipitación de diseño para TR {periodo_retorno} años: {precip_diseno:.2f} mm")
            return precip_diseno
            
        except Exception as e:
            print(f"Error en el análisis de frecuencia: {e}")
            # Fallback: usar percentil 99 si el ajuste de Gumbel falla
            precip_diseno = np.percentile(self.datos[self.columna_precipitacion].values, 99)
            print(f"Usando percentil 99 como fallback: {precip_diseno:.2f} mm")
            return precip_diseno
    
    def calcular_parametros_triangulares(self, precipitacion_diseno, duracion_total=24, 
                                       fraccion_pico=0.375):
        """
        Calcular parámetros del hietograma triangular
        
        Args:
            precipitacion_diseno (float): Precipitación de diseño en mm
            duracion_total (int): Duración total de la tormenta en horas
            fraccion_pico (float): Fracción de la duración hasta el pico (0-1)
            
        Returns:
            dict: Parámetros del hietograma triangular
        """
        # Calcular tiempo al pico
        tiempo_pico = fraccion_pico * duracion_total
        
        # Calcular intensidad pico
        intensidad_pico = (2 * precipitacion_diseno) / tiempo_pico
        
        # Calcular pendientes
        pendiente_ascendente = intensidad_pico / tiempo_pico
        pendiente_descendente = intensidad_pico / (duracion_total - tiempo_pico)
        
        # Guardar parámetros
        parametros = {
            'precipitacion_diseno': precipitacion_diseno,
            'duracion_total': duracion_total,
            'tiempo_pico': tiempo_pico,
            'intensidad_pico': intensidad_pico,
            'pendiente_ascendente': pendiente_ascendente,
            'pendiente_descendente': pendiente_descendente,
            'fraccion_pico': fraccion_pico
        }
        
        self.parametros['hietograma_triangular'] = parametros
        return parametros
    
    def generar_hietograma(self, resolucion=0.1):
        """
        Generar datos del hietograma triangular
        
        Args:
            resolucion (float): Resolución temporal en horas
            
        Returns:
            tuple: (tiempos, intensidades, precipitacion_acumulada)
        """
        if 'hietograma_triangular' not in self.parametros:
            raise ValueError("Primero debe calcular los parámetros triangulares")
            
        params = self.parametros['hietograma_triangular']
        
        # Generar array de tiempos
        tiempos = np.arange(0, params['duracion_total'] + resolucion, resolucion)
        intensidades = np.zeros_like(tiempos)
        
        # Calcular intensidad para cada tiempo
        for i, t in enumerate(tiempos):
            if t <= params['tiempo_pico']:
                # Rama ascendente
                intensidades[i] = params['pendiente_ascendente'] * t
            else:
                # Rama descendente
                intensidades[i] = params['intensidad_pico'] - params['pendiente_descendente'] * (t - params['tiempo_pico'])
        
        # Calcular precipitación acumulada (integral de las intensidades)
        precipitacion_acumulada = np.cumsum(intensidades) * resolucion
        
        # Ajustar para que la precipitación total sea exactamente la de diseño
        factor_ajuste = params['precipitacion_diseno'] / precipitacion_acumulada[-1]
        intensidades *= factor_ajuste
        precipitacion_acumulada *= factor_ajuste
        
        return tiempos, intensidades, precipitacion_acumulada
    
    def graficar_hietograma(self, tiempos, intensidades, precipitacion_acumulada, 
                          guardar_archivo=None, mostrar_grafico=True):
        """
        Graficar el hietograma triangular y la curva de precipitación acumulada
        
        Args:
            tiempos (array): Array de tiempos
            intensidades (array): Array de intensidades
            precipitacion_acumulada (array): Array de precipitación acumulada
            guardar_archivo (str): Ruta para guardar el gráfico
            mostrar_grafico (bool): Si se muestra el gráfico en pantalla
        """
        params = self.parametros['hietograma_triangular']
        freq_params = self.parametros.get('analisis_frecuencia', {})
        
        # Crear figura con dos subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Gráfico 1: Hietograma triangular (barras)
        ax1.bar(tiempos, intensidades, width=0.8, alpha=0.7, 
               color='skyblue', edgecolor='navy', label='Intensidad')
        ax1.set_xlabel('Tiempo (horas)')
        ax1.set_ylabel('Intensidad (mm/h)')
        ax1.set_title(
            f'Hietograma Triangular de Diseño\n'
            f'TR = {freq_params.get("periodo_retorno", "N/A")} años | '
            f'Duración = {params["duracion_total"]} h | '
            f'Precipitación total = {params["precipitacion_diseno"]:.1f} mm'
        )
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, params['duracion_total'])
        ax1.legend()
        
        # Gráfico 2: Precipitación acumulada
        ax2.plot(tiempos, precipitacion_acumulada, 'r-', linewidth=2, 
                label='Precipitación acumulada')
        ax2.set_xlabel('Tiempo (horas)')
        ax2.set_ylabel('Precipitación acumulada (mm)')
        ax2.set_title('Precipitación Acumulada')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_xlim(0, params['duracion_total'])
        
        plt.tight_layout()
        
        if guardar_archivo:
            plt.savefig(guardar_archivo, dpi=300, bbox_inches='tight')
        
        if mostrar_grafico:
            plt.show()
        
        return fig, (ax1, ax2)
    
    def generar_reporte(self, archivo_salida=None):
        """
        Generar un reporte con todos los parámetros calculados
        
        Args:
            archivo_salida (str): Ruta para guardar el reporte (opcional)
        """
        reporte = []
        reporte.append("=" * 60)
        reporte.append("REPORTE - HIETOGRAMA TRIANGULAR DE DISEÑO")
        reporte.append("=" * 60)
        
        # Información general
        reporte.append("\nINFORMACIÓN GENERAL:")
        reporte.append(f"- Número de registros: {len(self.datos)}")
        if self.columna_anio and self.columna_anio in self.datos.columns:
            anos = self.datos[self.columna_anio]
            reporte.append(f"- Período de datos: {min(anos)} - {max(anos)}")
        
        # Parámetros de análisis de frecuencia
        if 'analisis_frecuencia' in self.parametros:
            af = self.parametros['analisis_frecuencia']
            reporte.append("\nANÁLISIS DE FRECUENCIA:")
            reporte.append(f"- Distribución utilizada: {af['distribucion']}")
            reporte.append(f"- Periodo de retorno: {af['periodo_retorno']} años")
            reporte.append(f"- Precipitación de diseño: {af['precipitacion_diseno']:.2f} mm")
            reporte.append(f"- Media: {af['media']:.2f} mm")
            reporte.append(f"- Desviación estándar: {af['desviacion_estandar']:.2f} mm")
            reporte.append(f"- Coeficiente de variación: {af['coeficiente_variacion']:.3f}")
            reporte.append(f"- Máximo observado: {af['maximo_observado']:.2f} mm")
            reporte.append(f"- Mínimo observado: {af['minimo_observado']:.2f} mm")
        
        # Parámetros del hietograma triangular
        if 'hietograma_triangular' in self.parametros:
            ht = self.parametros['hietograma_triangular']
            reporte.append("\nPARÁMETROS DEL HIETOGRAMA TRIANGULAR:")
            reporte.append(f"- Duración total: {ht['duracion_total']} h")
            reporte.append(f"- Tiempo al pico: {ht['tiempo_pico']:.2f} h")
            reporte.append(f"- Fracción hasta el pico: {ht['fraccion_pico']:.3f}")
            reporte.append(f"- Intensidad pico: {ht['intensidad_pico']:.2f} mm/h")
            reporte.append(f"- Pendiente ascendente: {ht['pendiente_ascendente']:.4f} mm/h²")
            reporte.append(f"- Pendiente descendente: {ht['pendiente_descendente']:.4f} mm/h²")
        
        reporte.append("=" * 60)
        
        # Unir reporte
        reporte_completo = "\n".join(reporte)
        
        # Imprimir en pantalla
        print(reporte_completo)
        
        # Guardar en archivo si se especifica
        if archivo_salida:
            with open(archivo_salida, 'w') as f:
                f.write(reporte_completo)
            print(f"Reporte guardado en: {archivo_salida}")
        
        return reporte_completo

# Función principal para ejecutar el proceso completo
def generar_hietograma_desde_csv(archivo_csv, columna_precipitacion, columna_anio=None,
                                periodo_retorno=100, duracion_total=24, fraccion_pico=0.375,
                                resolucion=0.1, guardar_grafico=None, guardar_reporte=None):
    """
    Función principal para generar un hietograma triangular desde un archivo CSV
    
    Args:
        archivo_csv (str): Ruta al archivo CSV
        columna_precipitacion (str): Nombre de la columna con precipitación
        columna_anio (str): Nombre de la columna con año (opcional)
        periodo_retorno (int): Periodo de retorno en años
        duracion_total (int): Duración total de la tormenta en horas
        fraccion_pico (float): Fracción de la duración hasta el pico (0-1)
        resolucion (float): Resolución temporal en horas
        guardar_grafico (str): Ruta para guardar el gráfico (opcional)
        guardar_reporte (str): Ruta para guardar el reporte (opcional)
    """
    # Crear instancia de la clase
    hietograma = HietogramaTriangularCSV()
    
    # Cargar y ordenar datos
    if not hietograma.cargar_y_ordenar_datos(archivo_csv, columna_precipitacion, columna_anio):
        return None
    
    # Realizar análisis de frecuencia
    precip_diseno = hietograma.analisis_frecuencia(periodo_retorno)
    
    # Calcular parámetros del hietograma triangular
    hietograma.calcular_parametros_triangulares(precip_diseno, duracion_total, fraccion_pico)
    
    # Generar datos del hietograma
    tiempos, intensidades, precip_acumulada = hietograma.generar_hietograma(resolucion)
    
    # Graficar resultados
    hietograma.graficar_hietograma(tiempos, intensidades, precip_acumulada, guardar_grafico)
    
    # Generar reporte
    hietograma.generar_reporte(guardar_reporte)
    
    return hietograma

# Ejemplo de uso
if __name__ == "__main__":
    # Configuración de parámetros
    ARCHIVO_CSV = "/home/nass/Descargas/M.Num/PPC.csv"  # Reemplazar con la ruta a tu archivo CSV
    COLUMNA_PRECIPITACION = "PPC"  # Reemplazar con el nombre de tu columna de precipitación
    COLUMNA_ANIO = "año"  # Reemplazar con el nombre de tu columna de año (opcional)
    PERIODO_RETORNO = 100  # Periodo de retorno en años
    DURACION_TOTAL = 24  # Duración total en horas
    FRACCION_PICO = 0.375  # Fracción de la duración hasta el pico (37.5% es típico)
    
    # Generar hietograma
    hietograma = generar_hietograma_desde_csv(
        archivo_csv=ARCHIVO_CSV,
        columna_precipitacion=COLUMNA_PRECIPITACION,
        columna_anio=COLUMNA_ANIO,
        periodo_retorno=PERIODO_RETORNO,
        duracion_total=DURACION_TOTAL,
        fraccion_pico=FRACCION_PICO,
        guardar_grafico="hietograma_triangular.png",
        guardar_reporte="reporte_hietograma.txt"
    )