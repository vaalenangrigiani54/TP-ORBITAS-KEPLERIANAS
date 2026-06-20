import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os
from defs import MU_TIERRA
from lib.integradores import rk4_step

# Agregamos la carpeta padre al path para poder importar lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def dinamica_orbital(t, state):
    """
    Evalúa la ecuación diferencial del satélite LEO.
    state = [x, y, z, vx, vy, vz]
    """
    r_vec = state[0:3]
    v_vec = state[3:6]
    
    r_norm = np.linalg.norm(r_vec)
    
    # Verificamos que el radio no sea casi nulo antes de dividir
    if math.isclose(r_norm, 0.0, abs_tol=1e-9):
        raise ValueError("Error: División por radio casi nulo (posible colisión).")
    
    # Aceleración = - (mu / r^3) * r
    a_vec = - (MU_TIERRA / (r_norm**3)) * r_vec
    
    # Derivada del estado: [velocidades, aceleraciones]
    d_state = np.zeros(6)
    d_state[0:3] = v_vec
    d_state[3:6] = a_vec
    
    return d_state

def calcular_periodo_orbital(r0_vec, v0_vec):
    """Calcula el período orbital en segundos usando la energía específica."""
    r = np.linalg.norm(r0_vec)
    v = np.linalg.norm(v0_vec)
    energia_esp = (v**2) / 2.0 - (MU_TIERRA / r)
    a = -MU_TIERRA / (2.0 * energia_esp) # Semieje mayor
    T = 2 * np.pi * np.sqrt((a**3) / MU_TIERRA)
    return T

def propagar_orbita(P0, V0, paso_dt=1.0):
    """Propaga una órbita completa usando RK4."""
    periodo = calcular_periodo_orbital(P0, V0)
    pasos_totales = int(np.floor(periodo / paso_dt)) # Usando floor permitido
    
    # Inicializamos historial
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

def main():
    # Condiciones iniciales extraídas del enunciado 
    # Órbita circular
    P0_circ = np.array([6125.24, -3547.04, -3.31277])
    V0_circ = np.array([-0.519174, -0.903482, 7.43159])
    
    # Órbita geoestacionaria
    P0_geo = np.array([-41974.9, 4012.93, 23.5515])
    V0_geo = np.array([-0.292606, -3.06063, 0.000293508])
    
    # Órbita Molniya
    P0_mol = np.array([305.926, 3162.83, 6324.79])
    V0_mol = np.array([-9.86976, 0.326984, 0.313879])

    print("Propagando órbita Circular...")
    orb_circ = propagar_orbita(P0_circ, V0_circ, paso_dt=1.0) # Paso de 1 seg
    
    print("Propagando órbita Geoestacionaria...")
    orb_geo = propagar_orbita(P0_geo, V0_geo, paso_dt=1.0)
    
    print("Propagando órbita Molniya...")
    orb_mol = propagar_orbita(P0_mol, V0_mol, paso_dt=1.0)

    # Gráfico 3D superpuesto
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d') #
    
    ax.plot(orb_circ[:, 0], orb_circ[:, 1], orb_circ[:, 2], label='Circular', color='blue')
    ax.plot(orb_geo[:, 0], orb_geo[:, 1], orb_geo[:, 2], label='Geoestacionaria', color='orange')
    ax.plot(orb_mol[:, 0], orb_mol[:, 1], orb_mol[:, 2], label='Molniya', color='green')
    
    # Tierra en el origen
    ax.scatter(0, 0, 0, color='black', s=100, label='Tierra')
    
    ax.set_xlabel('X [km]')
    ax.set_ylabel('Y [km]')
    ax.set_zlabel('Z [km]')
    ax.set_title('Superposición de Órbitas (1 período)')
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()