#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 20:47:09 2025

@author: nass
"""

import pandas as pd

# Crear un cuadro comparativo con categorías de análisis
data = {
    "Problemática": [
        "Escasez y sobreexplotación de acuíferos",
        "Distribución desigual del recurso",
        "Contaminación del agua",
        "Infraestructura deficiente",
        "Impactos del cambio climático",
        "Gestión institucional",
        "Desigualdad social en el acceso"
    ],
    "Ambiental": [
        "Descenso de niveles freáticos, hundimientos, pérdida de reservas subterráneas",
        "Regiones con estrés hídrico extremo y ecosistemas degradados",
        "Ríos, lagos y acuíferos contaminados, pérdida de biodiversidad",
        "Desperdicio de agua por fugas y baja eficiencia en plantas",
        "Sequías, inundaciones, alteración de ciclos hidrológicos",
        "Falta de manejo integral de cuencas y acuíferos",
        "Degradación de fuentes locales y contaminación de pozos"
    ],
    "Económico": [
        "Afecta a la agricultura, industria y abasto urbano por costos de extracción",
        "Altos costos de trasvases y bombeo en regiones deficitarias",
        "Impactos en salud, pesca, turismo y productividad",
        "Altos gastos en mantenimiento y rehabilitación de redes",
        "Daños a infraestructura, pérdidas agrícolas y productivas",
        "Menor inversión en proyectos estratégicos hídricos",
        "Costos de atención sanitaria y rezago económico"
    ],
    "Social": [
        "Comunidades con acceso limitado y conflictos por agua",
        "Migración y desigualdad entre regiones con abundancia y escasez",
        "Problemas de salud pública y malestar social",
        "Desigual cobertura de servicios urbanos y rurales",
        "Afectación a comunidades vulnerables y marginadas",
        "Débil participación ciudadana en decisiones del agua",
        "Carencia de servicios básicos perpetúa pobreza"
    ],
    "Institucional": [
        "Monitoreo insuficiente de acuíferos sobreexplotados",
        "Falta de planeación coordinada entre regiones",
        "Insuficiente control de descargas y sanciones laxas",
        "Gestión fragmentada de infraestructura local y federal",
        "Escasa integración de la gestión del riesgo climático",
        "Fragmentación de competencias y limitaciones presupuestales",
        "Atención prioritaria limitada en zonas rurales"
    ],
    "Legal/Normativo": [
        "Títulos de concesión sin control efectivo del volumen extraído",
        "Falta de políticas de redistribución justa del agua",
        "Cumplimiento débil de NOMs sobre calidad del agua",
        "Rezago en normas de eficiencia y estándares de servicio",
        "Necesidad de actualización de marcos para adaptación climática",
        "Aplicación débil de la Ley de Aguas Nacionales",
        "Derecho humano al agua aún sin plena garantía práctica"
    ]
}

df = pd.DataFrame(data)

#import caas_jupyter_tools
#caas_jupyter_tools.display_dataframe_to_user("Cuadro comparativo de problemáticas del agua en México", df)
