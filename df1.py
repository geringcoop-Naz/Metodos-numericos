#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:29:35 2025

@author: nass
"""
import pandas as pd

usuarios = {
    "Nombre": ["Juan", "María", "Carlos", "Ana", "Luis"],
    "Apellido": ["Gómez", "López", "Rodríguez", "Pérez", "Martínez"],
    "Email": ["juan@example.com", "maria@example.com", "carlos@example.com", "ana@example.com", "luis@example.com"],
    "Telefono": ["123-123-4567", "456-987-6543", "789-567-8901", "654-234-5678", "963-678-9012"],
    "Edad": [12, 27, 22, 30, 16]
}

usuarios_df = pd.DataFrame(usuarios)

df_ordenado = usuarios_df.sort_values(by='Edad', ascending=False)
print(df_ordenado)

