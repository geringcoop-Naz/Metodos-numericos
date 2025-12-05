# ============================================================================
# ANÁLISIS ESTOCÁSTICO HIDROLÓGICO COMPLETO
# Autor: Sistema de Análisis Hidrológico
# Universidad de Guanajuato
# ============================================================================

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import warnings
import json
import matplotlib.backends.backend_pdf as pdf_backend
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN INICIAL
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14

# ============================================================================
# 1. DATOS DE EJEMPLO (CAUDALES MÁXIMOS ANUALES - Página 34 PDF)
# ============================================================================

DATOS_CAUDALES = {
    'Año': [1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 
            1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970],
    'Q_m3_s': [117.8, 139.1, 37.0, 192.8, 126.1, 71.5, 15.4, 34.5,
               68.5, 240.5, 48.2, 65.7, 526.4, 135.6, 134.4, 128.6]
}

# ============================================================================
# 2. FUNCIONES DE ANÁLISIS ESTADÍSTICO BÁSICO
# ============================================================================

def calcular_estadisticos_muestrales(datos):
    """Calcula estadísticos descriptivos básicos de la muestra"""
    n = len(datos)
    
    estadisticos = {
        'n': n,
        'media': np.mean(datos),
        'mediana': np.median(datos),
        'minimo': np.min(datos),
        'maximo': np.max(datos),
        'rango': np.max(datos) - np.min(datos),
        'varianza': np.var(datos, ddof=1),
        'desviacion_std': np.std(datos, ddof=1),
        'coef_variacion': np.std(datos, ddof=1) / np.mean(datos),
        'coef_asimetria': stats.skew(datos),
        'coef_curtosis': stats.kurtosis(datos),
        'percentil_25': np.percentile(datos, 25),
        'percentil_50': np.percentile(datos, 50),
        'percentil_75': np.percentile(datos, 75),
        'percentil_90': np.percentile(datos, 90),
        'percentil_95': np.percentile(datos, 95)
    }
    
    return estadisticos

# ============================================================================
# 3. PRUEBAS DE HOMOGENEIDAD
# ============================================================================

def prueba_helmert(datos, alpha=0.05):
    """Prueba de Helmert para homogeneidad de varianzas"""
    n = len(datos)
    n1 = n // 2
    n2 = n - n1
    
    serie1 = datos[:n1]
    serie2 = datos[n1:]
    
    var1 = np.var(serie1, ddof=1)
    var2 = np.var(serie2, ddof=1)
    
    # Estadístico F (razón de varianzas)
    if var2 > var1:
        F = var2 / var1
        df1, df2 = n2 - 1, n1 - 1
    else:
        F = var1 / var2
        df1, df2 = n1 - 1, n2 - 1
    
    # Valor crítico
    F_critico = stats.f.ppf(1 - alpha/2, df1, df2)
    
    return {
        'estadistico_F': F,
        'F_critico': F_critico,
        'pasa': F < F_critico,
        'varianza_serie1': var1,
        'varianza_serie2': var2,
        'n1': n1,
        'n2': n2
    }

def prueba_t_student(datos, alpha=0.05):
    """Prueba t-Student para homogeneidad de medias"""
    n = len(datos)
    n1 = n // 2
    n2 = n - n1
    
    serie1 = datos[:n1]
    serie2 = datos[n1:]
    
    media1 = np.mean(serie1)
    media2 = np.mean(serie2)
    var1 = np.var(serie1, ddof=1)
    var2 = np.var(serie2, ddof=1)
    
    # Prueba F para homogeneidad de varianzas
    F_prueba = max(var1, var2) / min(var1, var2)
    F_critico_var = stats.f.ppf(1 - alpha/2, n1-1, n2-1)
    
    if F_prueba < F_critico_var:
        # Varianzas iguales
        pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
        std_error = np.sqrt(pooled_var * (1/n1 + 1/n2))
        gl = n1 + n2 - 2
    else:
        # Varianzas diferentes (Welch)
        std_error = np.sqrt(var1/n1 + var2/n2)
        gl = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    t_stat = (media1 - media2) / std_error
    t_critico = stats.t.ppf(1 - alpha/2, gl)
    
    return {
        'estadistico_t': t_stat,
        't_critico': t_critico,
        'pasa': abs(t_stat) < t_critico,
        'media_serie1': media1,
        'media_serie2': media2,
        'gl': gl
    }

def prueba_cramer(datos, alpha=0.05):
    """Prueba de Cramer para detección de valores atípicos"""
    media = np.mean(datos)
    std = np.std(datos, ddof=1)
    
    # Estadístico de Cramer: τ = |Xi - X̄| / S
    estadisticos_cramer = np.abs(datos - media) / std
    
    # Valores críticos aproximados para Cramer
    n = len(datos)
    if n <= 20:
        critico = 2.64
    elif n <= 30:
        critico = 2.75
    else:
        critico = 3.0
    
    # Identificar valores atípicos
    valores_atipicos = estadisticos_cramer > critico
    indices_atipicos = np.where(valores_atipicos)[0]
    
    return {
        'estadisticos_cramer': estadisticos_cramer,
        'criterio': critico,
        'valores_atipicos': valores_atipicos,
        'indices_atipicos': indices_atipicos.tolist(),
        'pasa': len(indices_atipicos) == 0
    }

def pruebas_homogeneidad_completas(datos, alpha=0.05):
    """Realiza todas las pruebas de homogeneidad"""
    resultados = {
        'helmert': prueba_helmert(datos, alpha),
        't_student': prueba_t_student(datos, alpha),
        'cramer': prueba_cramer(datos, alpha)
    }
    
    # Conclusión general
    pruebas_pasan = sum([r['pasa'] for r in resultados.values()])
    resultados['conclusion'] = {
        'pruebas_pasan': pruebas_pasan,
        'total_pruebas': 3,
        'homogenea': pruebas_pasan >= 2,
        'recomendacion': 'Apta para análisis' if pruebas_pasan >= 2 else 'Requiere tratamiento'
    }
    
    return resultados

# ============================================================================
# 4. PRUEBA DE INDEPENDENCIA DE ANDERSON
# ============================================================================

def prueba_anderson_autocorrelacion(datos, alpha=0.05, max_lag=5):
    """Prueba de independencia de Anderson para autocorrelación"""
    n = len(datos)
    
    autocorrs = []
    estadisticos_z = []
    p_values = []
    
    for k in range(1, max_lag + 1):
        if k < n:
            corr_matrix = np.corrcoef(datos[:-k], datos[k:])
            rk = corr_matrix[0, 1]
        else:
            rk = 0
        
        # Estadístico Z de Anderson
        if k == 1:
            zk = rk * np.sqrt(n)
        else:
            zk = rk * np.sqrt(n - k)
        
        p_val = 2 * (1 - stats.norm.cdf(abs(zk)))
        
        autocorrs.append(rk)
        estadisticos_z.append(zk)
        p_values.append(p_val)
    
    # Intervalo de confianza
    conf_int = stats.norm.ppf(1 - alpha/2) / np.sqrt(n)
    independiente_por_lag = [abs(r) < conf_int for r in autocorrs]
    
    return {
        'autocorrelaciones': autocorrs,
        'estadisticos_z': estadisticos_z,
        'p_values': p_values,
        'intervalo_confianza': conf_int,
        'independiente_por_lag': independiente_por_lag,
        'independencia_general': all(independiente_por_lag),
        'alpha': alpha,
        'max_lag': max_lag
    }

# ============================================================================
# 5. PROBABILIDADES EMPÍRICAS
# ============================================================================

def calcular_probabilidades_empiricas(serie, metodos=['weibull', 'hazen', 'cunnane']):
    """Calcula probabilidades de no excedencia empíricas"""
    n = len(serie)
    i = np.arange(1, n + 1)
    
    formulas = {
        'weibull': lambda i, n: i / (n + 1),
        'hazen': lambda i, n: (i - 0.5) / n,
        'cunnane': lambda i, n: (i - 0.4) / (n + 0.2),
        'gringorten': lambda i, n: (i - 0.44) / (n + 0.12),
        'california': lambda i, n: i / n,
        'chegodaev': lambda i, n: (i - 0.3) / (n + 0.4)
    }
    
    resultados = {}
    for metodo in metodos:
        if metodo in formulas:
            p = formulas[metodo](i, n)
            resultados[metodo] = p
            resultados[metodo + '_T'] = 1 / (1 - p)
    
    return resultados

# ============================================================================
# 6. DISTRIBUCIONES DE PROBABILIDAD
# ============================================================================

class DistribucionesHidrologicas:
    """Clase para ajustar distribuciones de probabilidad hidrológicas"""
    
    def __init__(self, datos):
        self.datos = np.array(datos)
        self.n = len(datos)
        self.datos_ordenados = np.sort(datos)
    
    def _calcular_error_ajuste(self, cdf_func):
        """Calcula métricas de error de ajuste"""
        try:
            p_emp = (np.arange(1, self.n + 1)) / (self.n + 1)
            p_teo = cdf_func(self.datos_ordenados)
            
            mse = np.mean((p_emp - p_teo) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(p_emp - p_teo))
            
            ss_res = np.sum((p_emp - p_teo) ** 2)
            ss_tot = np.sum((p_emp - np.mean(p_emp)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
            
            see = np.sqrt(ss_res / (self.n - 2)) if self.n > 2 else np.nan
            
            return {'rmse': rmse, 'mae': mae, 'r2': r2, 'see': see}
        except:
            return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'see': np.nan}
    
    # 6.1 Distribución Normal
    def normal(self):
        media = np.mean(self.datos)
        sigma = np.std(self.datos, ddof=1)
        
        def cdf(x): return stats.norm.cdf(x, media, sigma)
        def pdf(x): return stats.norm.pdf(x, media, sigma)
        def ppf(p): return stats.norm.ppf(p, media, sigma)
        
        return {
            'nombre': 'Normal',
            'parametros': {'mu': media, 'sigma': sigma},
            'cdf': cdf, 'pdf': pdf, 'ppf': ppf,
            'error_ajuste': self._calcular_error_ajuste(cdf)
        }
    
    # 6.2 Log-Normal 2 parámetros
    def lognormal_2p(self):
        datos_log = np.log(self.datos)
        mu_log = np.mean(datos_log)
        sigma_log = np.std(datos_log, ddof=1)
        
        def cdf(x): return stats.lognorm.cdf(x, s=sigma_log, scale=np.exp(mu_log))
        def pdf(x): return stats.lognorm.pdf(x, s=sigma_log, scale=np.exp(mu_log))
        def ppf(p): return stats.lognorm.ppf(p, s=sigma_log, scale=np.exp(mu_log))
        
        return {
            'nombre': 'Log-Normal (2P)',
            'parametros': {'mu_log': mu_log, 'sigma_log': sigma_log},
            'cdf': cdf, 'pdf': pdf, 'ppf': ppf,
            'error_ajuste': self._calcular_error_ajuste(cdf)
        }
    
    # 6.3 Log-Normal 3 parámetros
    def lognormal_3p(self):
        # Estimación simplificada del umbral
        gamma = np.min(self.datos) * 0.9
        datos_transformados = self.datos - gamma
        datos_log = np.log(datos_transformados[datos_transformados > 0])
        
        mu_log = np.mean(datos_log)
        sigma_log = np.std(datos_log, ddof=1)
        
        def cdf(x):
            if x <= gamma: return 0
            return stats.lognorm.cdf(x - gamma, s=sigma_log, scale=np.exp(mu_log))
        
        def pdf(x):
            if x <= gamma: return 0
            return stats.lognorm.pdf(x - gamma, s=sigma_log, scale=np.exp(mu_log))
        
        def ppf(p):
            return gamma + stats.lognorm.ppf(p, s=sigma_log, scale=np.exp(mu_log))
        
        return {
            'nombre': 'Log-Normal (3P)',
            'parametros': {'gamma': gamma, 'mu_log': mu_log, 'sigma_log': sigma_log},
            'cdf': cdf, 'pdf': pdf, 'ppf': ppf,
            'error_ajuste': self._calcular_error_ajuste(cdf)
        }
    
    # 6.4 Gamma 2 parámetros
    def gamma_2p(self):
        # Método de momentos
        media = np.mean(self.datos)
        varianza = np.var(self.datos, ddof=1)
        
        alpha = (media ** 2) / varianza
        beta = varianza / media
        
        def cdf(x): return stats.gamma.cdf(x, alpha, scale=beta)
        def pdf(x): return stats.gamma.pdf(x, alpha, scale=beta)
        def ppf(p): return stats.gamma.ppf(p, alpha, scale=beta)
        
        return {
            'nombre': 'Gamma (2P)',
            'parametros': {'alpha': alpha, 'beta': beta},
            'cdf': cdf, 'pdf': pdf, 'ppf': ppf,
            'error_ajuste': self._calcular_error_ajuste(cdf)
        }
    
    # 6.5 Gamma 3 parámetros (Pearson III)
    def gamma_3p(self):
        params = stats.pearson3.fit(self.datos)
        skew, loc, scale = params[0], params[1], params[2]
        
        def cdf(x): return stats.pearson3.cdf(x, skew, loc=loc, scale=scale)
        def pdf(x): return stats.pearson3.pdf(x, skew, loc=loc, scale=scale)
        def ppf(p): return stats.pearson3.ppf(p, skew, loc=loc, scale=scale)
        
        return {
            'nombre': 'Pearson III',
            'parametros': {'skew': skew, 'loc': loc, 'scale': scale},
            'cdf': cdf, 'pdf': pdf, 'ppf': ppf,
            'error_ajuste': self._calcular_error_ajuste(cdf)
        }
    
    # 6.6 Log-Pearson III
    def log_pearson3(self):
        datos_log = np.log(self.datos)
        params = stats.pearson3.fit(datos_log)
        skew, loc, scale = params[0], params[1], params[2]
        
        def cdf(x):
            if x <= 0: return 0
            return stats.pearson3.cdf(np.log(x), skew, loc=loc, scale=scale)
        
        def pdf(x):
            if x <= 0: return 0
            return stats.pearson3.pdf(np.log(x), skew, loc=loc, scale=scale) / x
        
        def ppf(p):
            return np.exp(stats.pearson3.ppf(p, skew, loc=loc, scale=scale))
        
        return {
            'nombre': 'Log-Pearson III',
            'parametros': {'skew': skew, 'loc': loc, 'scale': scale},
            'cdf': cdf, 'pdf': pdf, 'ppf': ppf,
            'error_ajuste': self._calcular_error_ajuste(cdf)
        }
    
    # 6.7 Gumbel (EV1)
    def gumbel(self):
        media = np.mean(self.datos)
        std = np.std(self.datos, ddof=1)
        
        beta = std * np.sqrt(6) / np.pi
        mu = media - 0.5772 * beta
        
        def cdf(x): return np.exp(-np.exp(-(x - mu) / beta))
        def pdf(x):
            z = (x - mu) / beta
            return (1/beta) * np.exp(-z - np.exp(-z))
        def ppf(p): return mu - beta * np.log(-np.log(p))
        
        return {
            'nombre': 'Gumbel (EV1)',
            'parametros': {'mu': mu, 'beta': beta},
            'cdf': cdf, 'pdf': pdf, 'ppf': ppf,
            'error_ajuste': self._calcular_error_ajuste(cdf)
        }
    
    # 6.8 General de Valores Extremos (GEV)
    def gev(self):
        params = stats.genextreme.fit(self.datos)
        c, loc, scale = params[0], params[1], params[2]
        
        def cdf(x): return stats.genextreme.cdf(x, c, loc=loc, scale=scale)
        def pdf(x): return stats.genextreme.pdf(x, c, loc=loc, scale=scale)
        def ppf(p): return stats.genextreme.ppf(p, c, loc=loc, scale=scale)
        
        return {
            'nombre': 'GEV',
            'parametros': {'c': c, 'loc': loc, 'scale': scale},
            'cdf': cdf, 'pdf': pdf, 'ppf': ppf,
            'error_ajuste': self._calcular_error_ajuste(cdf)
        }
    
    # 6.9 Ajustar todas las distribuciones
    def ajustar_todas(self):
        distribuciones = [
            self.normal(),
            self.lognormal_2p(),
            self.lognormal_3p(),
            self.gamma_2p(),
            self.gamma_3p(),
            self.log_pearson3(),
            self.gumbel(),
            self.gev()
        ]
        
        # Filtrar solo las que tienen error calculado
        distribuciones_validas = []
        for dist in distribuciones:
            if dist['error_ajuste'] and not np.isnan(dist['error_ajuste']['rmse']):
                distribuciones_validas.append(dist)
        
        return distribuciones_validas

# ============================================================================
# 7. VISUALIZACIONES
# ============================================================================

def graficar_serie_temporal(df):
    """Grafica serie temporal de caudales"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Serie temporal
    ax1.plot(df['Año'], df['Q_m3_s'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Año')
    ax1.set_ylabel('Caudal (m³/s)')
    ax1.set_title('SERIE TEMPORAL DE CAUDALES MÁXIMOS ANUALES')
    ax1.grid(True, alpha=0.3)
    
    # Histograma
    ax2.hist(df['Q_m3_s'], bins=6, density=True, alpha=0.6, edgecolor='black')
    ax2.set_xlabel('Caudal (m³/s)')
    ax2.set_ylabel('Densidad')
    ax2.set_title('DISTRIBUCIÓN DE FRECUENCIAS')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def graficar_distribuciones_ajustadas(datos, distribuciones):
    """Grafica comparación de distribuciones ajustadas"""
    n = len(datos)
    datos_ordenados = np.sort(datos)
    p_emp = (np.arange(1, n + 1)) / (n + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Gráfico P-P
    ax1 = axes[0, 0]
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    for dist in distribuciones[:5]:  # Solo las primeras 5 para claridad
        p_teo = dist['cdf'](datos_ordenados)
        ax1.scatter(p_emp, p_teo, alpha=0.6, s=30, label=dist['nombre'])
    ax1.set_xlabel('Probabilidad empírica')
    ax1.set_ylabel('Probabilidad teórica')
    ax1.set_title('GRÁFICO P-P')
    ax1.legend(fontsize=8)
    
    # Gráfico Q-Q
    ax2 = axes[0, 1]
    for dist in distribuciones[:5]:
        cuantiles_teoricos = dist['ppf'](p_emp)
        ax2.scatter(datos_ordenados, cuantiles_teoricos, alpha=0.6, s=30)
    min_val, max_val = np.min(datos_ordenados), np.max(datos_ordenados)
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    ax2.set_xlabel('Cuantiles observados')
    ax2.set_ylabel('Cuantiles teóricos')
    ax2.set_title('GRÁFICO Q-Q')
    
    # PDFs
    ax3 = axes[1, 0]
    x_range = np.linspace(np.min(datos)*0.5, np.max(datos)*1.5, 1000)
    for dist in distribuciones[:5]:
        pdf_vals = dist['pdf'](x_range)
        ax3.plot(x_range, pdf_vals, linewidth=2, label=dist['nombre'])
    ax3.hist(datos, bins=6, density=True, alpha=0.3, edgecolor='black')
    ax3.set_xlabel('Caudal (m³/s)')
    ax3.set_ylabel('Densidad')
    ax3.set_title('FUNCIONES DE DENSIDAD')
    ax3.legend(fontsize=8)
    
    # CDFs
    ax4 = axes[1, 1]
    ax4.step(datos_ordenados, p_emp, where='post', label='Empírica', linewidth=2)
    for dist in distribuciones[:5]:
        cdf_vals = dist['cdf'](x_range)
        ax4.plot(x_range, cdf_vals, label=dist['nombre'], linewidth=2)
    ax4.set_xlabel('Caudal (m³/s)')
    ax4.set_ylabel('Probabilidad acumulada')
    ax4.set_title('FUNCIONES DE DISTRIBUCIÓN')
    ax4.legend(fontsize=8)
    
    plt.suptitle('COMPARACIÓN DE DISTRIBUCIONES DE PROBABILIDAD', fontsize=16)
    plt.tight_layout()
    return fig

def graficar_curvas_frecuencia(distribuciones, periodos_retorno):
    """Grafica curvas de frecuencia para diferentes períodos de retorno"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    periodos_log = np.log10(periodos_retorno)
    
    for dist in distribuciones[:5]:  # Solo 5 para claridad
        try:
            caudales = []
            for T in periodos_retorno:
                p = 1 - 1/T
                Q = dist['ppf'](p)
                caudales.append(Q)
            
            ax.plot(periodos_log, caudales, 'o-', linewidth=2, label=dist['nombre'])
        except:
            continue
    
    ax.set_xlabel('Período de retorno, T (años) - escala log')
    ax.set_ylabel('Caudal (m³/s)')
    ax.set_title('CURVAS DE FRECUENCIA')
    ax.set_xticks(periodos_log)
    ax.set_xticklabels([f'T={T}' for T in periodos_retorno])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def graficar_qq(datos, distribucion, nombre_distribucion):
    """
    Genera un gráfico Q-Q para una distribución ajustada.
    
    Parameters:
    -----------
    datos : array, datos observados
    distribucion : dict, con función 'ppf' para calcular cuantiles teóricos
    nombre_distribucion : str, nombre de la distribución para el título
    """
    # Ordenar datos
    datos_ordenados = np.sort(datos)
    n = len(datos)
    
    # Probabilidades empíricas (Weibull)
    p_emp = (np.arange(1, n + 1) - 0.5) / n  # Usamos Hazen para mejor ajuste
    
    # Calcular cuantiles teóricos
    cuantiles_teoricos = distribucion['ppf'](p_emp)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Scatter plot de Q-Q
    ax.scatter(cuantiles_teoricos, datos_ordenados, alpha=0.6, s=50)
    
    # Línea y=x (ajustada para que pase por la mediana)
    min_val = min(cuantiles_teoricos.min(), datos_ordenados.min())
    max_val = max(cuantiles_teoricos.max(), datos_ordenados.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    
    # Etiquetas y título
    ax.set_xlabel('Cuantiles teóricos')
    ax.set_ylabel('Cuantiles observados')
    ax.set_title(f'Gráfico Q-Q - {nombre_distribucion}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Calcular R² para la regresión lineal entre cuantiles teóricos y observados
    from scipy import stats as sp_stats
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(cuantiles_teoricos, datos_ordenados)
    ax.text(0.05, 0.95, f'R² = {r_value**2:.4f}', transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    return fig

# ============================================================================
# 8. ANÁLISIS COMPLETO
# ============================================================================

def analisis_estocastico_completo(df, periodo_diseno=100):
    """
    Realiza análisis estocástico hidrológico completo
    
    Parameters:
    -----------
    df : DataFrame con columnas 'Año' y 'Q_m3_s'
    periodo_diseno : Período de retorno para diseño
    
    Returns:
    --------
    dict : Resultados completos del análisis
    """
    datos = df['Q_m3_s'].values
    años = df['Año'].values
    
    print("=" * 80)
    print("ANÁLISIS ESTOCÁSTICO HIDROLÓGICO COMPLETO")
    print("=" * 80)
    
    # 1. Estadísticos básicos
    print("\n1. ESTADÍSTICOS MUESTRALES:")
    estadisticos = calcular_estadisticos_muestrales(datos)
    for key, value in estadisticos.items():
        if isinstance(value, float):
            print(f"   {key:20}: {value:10.4f}")
        else:
            print(f"   {key:20}: {value:10}")
    
    # 2. Pruebas de homogeneidad
    print("\n2. PRUEBAS DE HOMOGENEIDAD:")
    homogeneidad = pruebas_homogeneidad_completas(datos)
    
    print(f"   Prueba de Helmert: {'PASA' if homogeneidad['helmert']['pasa'] else 'FALLA'}")
    print(f"   Prueba t-Student:  {'PASA' if homogeneidad['t_student']['pasa'] else 'FALLA'}")
    print(f"   Prueba de Cramer:  {'PASA' if homogeneidad['cramer']['pasa'] else 'FALLA'}")
    
    if homogeneidad['cramer']['indices_atipicos']:
        print(f"   Valores atípicos encontrados en índices: {homogeneidad['cramer']['indices_atipicos']}")
    
    # 3. Prueba de independencia
    print("\n3. PRUEBA DE INDEPENDENCIA DE ANDERSON:")
    independencia = prueba_anderson_autocorrelacion(datos)
    print(f"   Independencia general: {'SÍ' if independencia['independencia_general'] else 'NO'}")
    print(f"   Autocorrelación lag-1: {independencia['autocorrelaciones'][0]:.4f}")
    
    # 4. Probabilidades empíricas
    print("\n4. PROBABILIDADES EMPÍRICAS:")
    probs = calcular_probabilidades_empiricas(np.sort(datos))
    print(f"   Métodos calculados: {len([k for k in probs.keys() if not k.endswith('_T')])}")
    
    # 5. Ajuste de distribuciones
    print("\n5. AJUSTE DE DISTRIBUCIONES DE PROBABILIDAD:")
    dist_hidro = DistribucionesHidrologicas(datos)
    distribuciones = dist_hidro.ajustar_todas()
    
    # Ordenar por RMSE (mejor ajuste primero)
    distribuciones.sort(key=lambda x: x['error_ajuste']['rmse'])
    
    for i, dist in enumerate(distribuciones[:5], 1):
        err = dist['error_ajuste']
        print(f"   {i:2}. {dist['nombre']:20} RMSE: {err['rmse']:.6f}, R²: {err['r2']:.6f}")
    
    # 6. Caudal de diseño
    print(f"\n6. CAUDAL DE DISEÑO (T = {periodo_diseno} años):")
    mejor_dist = distribuciones[0]  # Mejor ajuste
    p_diseno = 1 - 1/periodo_diseno
    
    try:
        caudal_diseno = mejor_dist['ppf'](p_diseno)
        print(f"   Distribución seleccionada: {mejor_dist['nombre']}")
        print(f"   Caudal de diseño (Q{periodo_diseno}): {caudal_diseno:.2f} m³/s")
        
        # Intervalo de confianza
        if not np.isnan(mejor_dist['error_ajuste']['see']):
            see = mejor_dist['error_ajuste']['see']
            conf_inf = caudal_diseno - 1.96 * see
            conf_sup = caudal_diseno + 1.96 * see
            print(f"   Intervalo 95%: [{conf_inf:.2f}, {conf_sup:.2f}] m³/s")
    except Exception as e:
        print(f"   Error calculando caudal: {e}")
        caudal_diseno = None
    
    # 7. Generar gráficos
    print("\n7. GENERANDO GRÁFICOS...")
    fig1 = graficar_serie_temporal(df)
    fig2 = graficar_distribuciones_ajustadas(datos, distribuciones)
    fig3 = graficar_curvas_frecuencia(distribuciones, [2, 5, 10, 20, 50, 100, 500, 1000])
    fig4 = graficar_qq(datos, mejor_dist, mejor_dist['nombre'])
    fig5 = graficar_qq(datos, distribuciones[4], distribuciones[4]['nombre'])
    
    # Guardar gráficos
    fig1.savefig('serie_temporal.png', dpi=300, bbox_inches='tight')
    fig2.savefig('distribuciones.png', dpi=300, bbox_inches='tight')
    fig3.savefig('curvas_frecuencia.png', dpi=300, bbox_inches='tight')
    fig4.savefig('qq_mejor_distribucion.png', dpi=300, bbox_inches='tight')
    fig5.savefig('qq_segunda_distribucion.png', dpi=300, bbox_inches='tight')
    
    # 8. Resultados completos
    resultados = {
        'estadisticos': estadisticos,
        'homogeneidad': homogeneidad,
        'independencia': independencia,
        'mejor_distribucion': {
            'nombre': mejor_dist['nombre'],
            'parametros': mejor_dist['parametros'],
            'error_ajuste': mejor_dist['error_ajuste']
        },
        'caudal_diseno': caudal_diseno,
        'periodo_diseno': periodo_diseno,
        'distribuciones_comparadas': [
            {
                'nombre': d['nombre'],
                'rmse': d['error_ajuste']['rmse'],
                'r2': d['error_ajuste']['r2']
            }
            for d in distribuciones[:5]
        ]
    }
    
    # Guardar resultados en JSON
    with open('resultados_analisis.json', 'w') as f:
        json.dump(resultados, f, indent=5, default=str)
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 80)
    print("\nARCHIVOS GENERADOS:")
    print("  1. serie_temporal.png     - Serie temporal e histograma")
    print("  2. distribuciones.png     - Comparación de distribuciones")
    print("  3. curvas_frecuencia.png  - Curvas de frecuencia")
    print("  4. resultados_analisis.json - Resultados completos en JSON")
    print("  5. qq_mejor_distribucion.png - Gráfico Q-Q mejor distribución")
    print("  6. qq_segunda_distribucion.png - Gráfico Q-Q segunda distribución")    
    
    return resultados

# ============================================================================
# 9. EJECUCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal del análisis estocástico"""
    # Cargar datos
    df = pd.DataFrame(DATOS_CAUDALES)
    
    print("DATOS DE CAUDALES MÁXIMOS ANUALES")
    print("-" * 40)
    print(df.to_string(index=False))
    print(f"\nPeríodo: {df['Año'].min()} - {df['Año'].max()}")
    print(f"Número de años: {len(df)}")
    
    # Realizar análisis completo
    resultados = analisis_estocastico_completo(df, periodo_diseno=100)
    
    # Mostrar resumen ejecutivo
    print("\n" + "=" * 80)
    print("RESUMEN EJECUTIVO")
    print("=" * 80)
    
    print(f"\n1. HOMOGENEIDAD DE LA SERIE:")
    hom = resultados['homogeneidad']['conclusion']
    print(f"   {hom['pruebas_pasan']}/{hom['total_pruebas']} pruebas pasan")
    print(f"   Conclusión: {'HOMOGÉNEA' if hom['homogenea'] else 'NO HOMOGÉNEA'}")
    
    print(f"\n2. INDEPENDENCIA:")
    ind = resultados['independencia']
    print(f"   Autocorrelación lag-1: {ind['autocorrelaciones'][0]:.4f}")
    print(f"   Independiente: {'SÍ' if ind['independencia_general'] else 'NO'}")
    
    print(f"\n3. MEJOR DISTRIBUCIÓN:")
    mejor = resultados['mejor_distribucion']
    print(f"   {mejor['nombre']}")
    print(f"   RMSE: {mejor['error_ajuste']['rmse']:.6f}")
    print(f"   R²: {mejor['error_ajuste']['r2']:.6f}")
    
    print(f"\n4. CAUDAL DE DISEÑO (T=100 años):")
    if resultados['caudal_diseno']:
        print(f"   {resultados['caudal_diseno']:.2f} m³/s")
    else:
        print("   No se pudo calcular")
    
    # Mostrar ranking de distribuciones
    print(f"\n5. RANKING DE DISTRIBUCIONES (por RMSE):")
    for i, dist in enumerate(resultados['distribuciones_comparadas'], 1):
        print(f"   {i:2}. {dist['nombre']:20} RMSE: {dist['rmse']:.6f}")
    
    # Recomendaciones
    print(f"\n6. RECOMENDACIONES:")
    print("   - Validar resultados con información adicional del sitio")
    print("   - Considerar incertidumbre en estimaciones extremas")
    print("   - Revisar valores atípicos identificados")
    
    return resultados

# ============================================================================
# 10. EJECUCIÓN DEL PROGRAMA
# ============================================================================

if __name__ == "__main__":
    try:
        resultados_finales = main()
        
        # Preguntar si se desea exportar a Excel
        respuesta = input("\n¿Desea exportar los resultados a Excel? (s/n): ")
        if respuesta.lower() == 's':
            # Crear Excel con resultados
            with pd.ExcelWriter('resultados_hidrologia.xlsx') as writer:
                # Datos originales
                df_datos = pd.DataFrame(DATOS_CAUDALES)
                df_datos.to_excel(writer, sheet_name='Datos', index=False)
                
                # Estadísticos
                df_estad = pd.DataFrame([resultados_finales['estadisticos']])
                df_estad.to_excel(writer, sheet_name='Estadísticos', index=False)
                
                # Distribuciones comparadas
                df_dist = pd.DataFrame(resultados_finales['distribuciones_comparadas'])
                df_dist.to_excel(writer, sheet_name='Distribuciones', index=False)
                
                # Homogeneidad
                datos_hom = {
                    'Prueba': ['Helmert', 't-Student', 'Cramer'],
                    'Resultado': [
                        'PASA' if resultados_finales['homogeneidad']['helmert']['pasa'] else 'FALLA',
                        'PASA' if resultados_finales['homogeneidad']['t_student']['pasa'] else 'FALLA',
                        'PASA' if resultados_finales['homogeneidad']['cramer']['pasa'] else 'FALLA'
                    ]
                }
                df_hom = pd.DataFrame(datos_hom)
                df_hom.to_excel(writer, sheet_name='Homogeneidad', index=False)
            
            print("✓ Resultados exportados a 'resultados_hidrologia.xlsx'")
        
        print("\n" + "=" * 80)
        print("PROGRAMA FINALIZADO")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Revise los datos de entrada y las dependencias instaladas.")
