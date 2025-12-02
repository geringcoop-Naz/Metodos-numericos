import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. MOTOR DE INTEGRACIÓN (Dormand-Prince 5(4))
# ==========================================

# Coeficientes
C = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1.0, 1.0], dtype=float)
A = [
    [], 
    [1/5],
    [3/40, 9/40],
    [44/45, -56/15, 32/9],
    [19372/6561, -25360/2187, 64448/6561, -212/729],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656],
    [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84], 
]
B5 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0], dtype=float)
B4 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40], dtype=float)

def rk45_step(f, t, y, h):
    """Un paso Dormand-Prince."""
    k = []
    k1 = f(t, y); k.append(k1)
    k2 = f(t + C[1]*h, y + h*(A[1][0]*k[0])); k.append(k2)
    k3 = f(t + C[2]*h, y + h*(A[2][0]*k[0] + A[2][1]*k[1])); k.append(k3)
    k4 = f(t + C[3]*h, y + h*(A[3][0]*k[0] + A[3][1]*k[1] + A[3][2]*k[2])); k.append(k4)
    k5 = f(t + C[4]*h, y + h*(A[4][0]*k[0] + A[4][1]*k[1] + A[4][2]*k[2] + A[4][3]*k[3])); k.append(k5)
    k6 = f(t + C[5]*h, y + h*(A[5][0]*k[0] + A[5][1]*k[1] + A[5][2]*k[2] + A[5][3]*k[3] + A[5][4]*k[4])); k.append(k6)
    k7 = f(t + C[6]*h, y + h*(A[6][0]*k[0] + A[6][1]*k[1] + A[6][2]*k[2] + A[6][3]*k[3] + A[6][4]*k[4] + A[6][5]*k[5])); k.append(k7)
    
    y5 = y + h * sum(B5[i]*k[i] for i in range(7))
    y4 = y + h * sum(B4[i]*k[i] for i in range(7))
    return y5, y4, k

def integrate_rk45(f, t0, y0, tf, h0=1e-2, rtol=1e-6, atol=1e-9,
                   hmin=1e-8, hmax=0.5, safety=0.9, minfac=0.2, maxfac=5.0):
    """Integrador adaptativo principal."""
    t = float(t0)
    y = np.array(y0, dtype=float)
    T = [t]; Y = [y.copy()]; H_used = []
    accepted = 0; rejected = 0
    h = float(h0); p = 5.0
    
    while t < tf:
        if h < hmin: h = hmin
        if h > hmax: h = hmax
        if t + h > tf: h = tf - t
        
        y5, y4, k = rk45_step(f, t, y, h)
        
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y5))
        err = np.sqrt(np.mean(((y5 - y4)/scale)**2))
        
        if err <= 1.0 or h <= hmin*1.000000000001:
            t += h; y = y5
            T.append(t); Y.append(y.copy()); H_used.append(h)
            accepted += 1
            fac = safety * (1.0 / max(err, 1e-16))**(1.0/p)
            h = np.clip(h*fac, h*minfac, h*maxfac)
        else:
            rejected += 1
            fac = safety * (1.0 / err)**(1.0/p)
            h = np.clip(h*fac, h*minfac, h*maxfac)
            continue
                
    return np.array(T), np.vstack(Y), np.array(H_used), accepted, rejected

# ==========================================
# 2. DEFINICIÓN DEL PROBLEMA (Lotka-Volterra)
# ==========================================

def lotka_volterra(t, y, alpha=1.1, beta=0.4, delta=0.1, gamma=0.4):
    """
    Ecuaciones Predador-Presa:
    dx/dt = alpha*x - beta*x*z
    dz/dt = delta*x*z - gamma*z
    """
    x, z = y 
    dxdt = alpha*x - beta*x*z
    dzdt = delta*x*z - gamma*z
    return np.array([dxdt, dzdt])

# ==========================================
# 3. EJECUCIÓN Y GRÁFICOS
# ==========================================

# Condiciones iniciales
t0, tf = 0.0, 80.0
y0 = np.array([20.0, 5.0]) # 20 Presas, 5 Depredadores

# Llamada al integrador
print("Iniciando integración...")
T, Y, H, acc, rej = integrate_rk45(lotka_volterra, t0, y0, tf,
                                   h0=0.05, rtol=1e-6, atol=1e-9,
                                   hmin=1e-6, hmax=0.5)
plt.plot(T[:-1],H)
print(f"Tiempo total sumando los pasos: {np.sum(H)}")

print(f"Integración completada.")
print(f"Pasos aceptados: {acc}, Rechazados: {rej}")
print(f"Paso temporal promedio (h): {H.mean():.4e}")

# Extracción de variables
x_vals = Y[:, 0] # Presas
y_vals = Y[:, 1] # Depredadores

# Gráficos
plt.figure(figsize=(12, 5))

# Subplot 1: Series de tiempo
plt.subplot(1, 2, 1)
plt.plot(T, x_vals, label='Presas (x)', color='blue')
plt.plot(T, y_vals, label='Depredadores (z)', color='red')
plt.xlabel('Tiempo (t)')
plt.ylabel('Población')
plt.title('Dinámica de Poblaciones')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Espacio de fases (Ciclo límite)
plt.subplot(1, 2, 2)
plt.plot(x_vals, y_vals, color='purple')
plt.xlabel('Presas (x)')
plt.ylabel('Depredadores (z)')
plt.title('Espacio de Fases (x vs z)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()