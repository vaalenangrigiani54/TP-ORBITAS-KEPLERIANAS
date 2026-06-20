import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
import os

# Agregamos la carpeta padre al path para poder importar lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.integradores import rk4_step

# Constante de gravitación terrestre (mu = G * mTierra) en km^3/s^2
MU_TIERRA = 398600.4418

# ==========================================
# LÓGICA FÍSICA Y DE PROPAGACIÓN
# ==========================================

def dinamica_orbital(t, state):
    """Evalúa la ecuación diferencial del satélite LEO."""
    r_vec = state[0:3]
    v_vec = state[3:6]
    r_norm = np.linalg.norm(r_vec)
    
    if math.isclose(r_norm, 0.0, abs_tol=1e-9):
        raise ValueError("Error: División por radio casi nulo (posible colisión).")
    
    a_vec = - (MU_TIERRA / (r_norm**3)) * r_vec
    
    d_state = np.zeros(6)
    d_state[0:3] = v_vec
    d_state[3:6] = a_vec
    
    return d_state

def calcular_periodo_orbital(r0_vec, v0_vec):
    """Calcula el período orbital en segundos usando la energía específica."""
    r = np.linalg.norm(r0_vec)
    v = np.linalg.norm(v0_vec)
    energia_esp = (v**2) / 2.0 - (MU_TIERRA / r)
    a = -MU_TIERRA / (2.0 * energia_esp)
    T = 2 * np.pi * np.sqrt((a**3) / MU_TIERRA)
    return T

def propagar_orbita(P0, V0, paso_dt=1.0):
    """Propaga una órbita completa usando RK4."""
    periodo = calcular_periodo_orbital(P0, V0)
    pasos_totales = int(np.floor(periodo / paso_dt))
    
    historial = np.zeros((pasos_totales, 6))
    historial[0, 0:3] = P0
    historial[0, 3:6] = V0
    
    t_actual = 0.0
    estado_actual = historial[0]
    
    for i in range(1, pasos_totales):
        estado_actual = rk4_step(dinamica_orbital, t_actual, estado_actual, paso_dt)
        historial[i] = estado_actual
        t_actual += paso_dt
        
    return historial

# ==========================================
# LÓGICA DE PRESENTACIÓN (GRÁFICOS)
# ==========================================

def graficar_orbitas(resultados_orbitas):
    """
    Recibe un diccionario con los historiales de las órbitas y las grafica en 3D.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Iteramos sobre los resultados para graficar dinámicamente
    for nombre, datos in resultados_orbitas.items():
        historial = datos['historial']
        ax.plot(historial[:, 0], historial[:, 1], historial[:, 2], 
                label=nombre, color=datos['color'])
    
    # Tierra en el origen
    ax.scatter(0, 0, 0, color='black', s=100, label='Tierra')
    
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_title('Superposición de Órbitas (1 período)')
    ax.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# FUNCIÓN PRINCIPAL
# ==========================================

def main():
    # Agrupamos la configuración inicial en un diccionario (Estructura de datos limpia)
    config_orbitas = {
        'Circular': {
            'P0': np.array([6125.24, -3547.04, -3.31277]),
            'V0': np.array([-0.519174, -0.903482, 7.43159]),
            'color': 'blue'
        },
        'Geoestacionaria': {
            'P0': np.array([-41974.9, 4012.93, 23.5515]),
            'V0': np.array([-0.292606, -3.06063, 0.000293508]),
            'color': 'orange'
        },
        'Molniya': {
            'P0': np.array([305.926, 3162.83, 6324.79]),
            'V0': np.array([-9.86976, 0.326984, 0.313879]),
            'color': 'green'
        }
    }

    # Diccionario para guardar los resultados
    resultados = {}

    # Bucle de propagación: procesa cualquier cantidad de órbitas automáticamente
    for nombre, config in config_orbitas.items():
        print(f"Propagando órbita {nombre}...")
        historial = propagar_orbita(config['P0'], config['V0'], paso_dt=1.0)
        
        # Guardamos el historial y mantenemos el color para el gráfico
        resultados[nombre] = {
            'historial': historial,
            'color': config['color']
        }

    # Llamamos a la función dedicada exclusivamente a graficar
    graficar_orbitas(resultados)

if __name__ == "__main__":
    main()