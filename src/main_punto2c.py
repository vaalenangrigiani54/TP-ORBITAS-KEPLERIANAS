import numpy as np
import matplotlib.pyplot as plt
import math
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lib.integradores import rk4_step

MU_TIERRA = 398600.4418

def dinamica_orbital(t, state):
    r_vec = state[0:3]
    v_vec = state[3:6]
    r_norm = np.linalg.norm(r_vec)
    if math.isclose(r_norm, 0.0, abs_tol=1e-9):
        raise ValueError("Error: División por radio casi nulo.")
    a_vec = - (MU_TIERRA / (r_norm**3)) * r_vec
    d_state = np.zeros(6)
    d_state[0:3] = v_vec
    d_state[3:6] = a_vec
    return d_state

def producto_cruz(a, b):
    """Implementación manual del producto vectorial (^) para no usar librerías extra."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])

def propagar_24hs(P0, V0, paso_dt=1.0):
    """Propaga la órbita estrictamente por 24 horas."""
    pasos_totales = int(24 * 60 * 60 / paso_dt) # 86400 pasos para 1 segundo
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

def calcular_magnitudes(historial):
    pasos = len(historial)
    norm_h = np.zeros(pasos)
    norm_h_dot = np.zeros(pasos)
    energia = np.zeros(pasos)
    
    for i in range(pasos):
        r = historial[i, 0:3]
        v = historial[i, 3:6]
        r_norm = np.linalg.norm(r)
        
        # Aceleración (r punto punto)
        a = - (MU_TIERRA / (r_norm**3)) * r
        
        # Momento angular: h = r ^ v
        h_vec = producto_cruz(r, v)
        norm_h[i] = np.linalg.norm(h_vec)
        
        # Derivada momento angular: h_dot = (v ^ v) + (r ^ a)
        h_dot_vec = producto_cruz(v, v) + producto_cruz(r, a)
        norm_h_dot[i] = np.linalg.norm(h_dot_vec)
        
        # Energía: eps = 0.5 * (v . v) - mu / r
        energia[i] = 0.5 * np.dot(v, v) - (MU_TIERRA / r_norm)
        
    return norm_h, norm_h_dot, energia

def main_2c():
    P0_circ = np.array([6125.24, -3547.04, -3.31277])
    V0_circ = np.array([-0.519174, -0.903482, 7.43159])
    
    P0_geo = np.array([-41974.9, 4012.93, 23.5515])
    V0_geo = np.array([-0.292606, -3.06063, 0.000293508])
    
    P0_mol = np.array([305.926, 3162.83, 6324.79])
    V0_mol = np.array([-9.86976, 0.326984, 0.313879])

    print("Integrando 24hs...")
    orb_circ = propagar_24hs(P0_circ, V0_circ)
    orb_geo  = propagar_24hs(P0_geo, V0_geo)
    orb_mol  = propagar_24hs(P0_mol, V0_mol)

    h_circ, hd_circ, e_circ = calcular_magnitudes(orb_circ)
    h_geo, hd_geo, e_geo = calcular_magnitudes(orb_geo)
    h_mol, hd_mol, e_mol = calcular_magnitudes(orb_mol)
    
    tiempo = np.linspace(0, 24, len(h_circ)) # Eje X en horas

    # Gráficos
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Gráfico 1: Momento Angular
    axs[0].plot(tiempo, h_circ, label='Circular', color='blue')
    axs[0].plot(tiempo, h_geo, label='Geoestacionaria', color='orange')
    axs[0].plot(tiempo, h_mol, label='Molniya', color='green')
    axs[0].set_title('Norma del Momento Angular (||h||)')
    axs[0].set_ylabel('km^2/s')
    axs[0].legend()
    
    # Gráfico 2: Derivada del Momento Angular
    axs[1].plot(tiempo, hd_circ, label='Circular', color='blue')
    axs[1].plot(tiempo, hd_geo, label='Geoestacionaria', color='orange')
    axs[1].plot(tiempo, hd_mol, label='Molniya', color='green')
    axs[1].set_title('Norma de la Derivada del Momento Angular (||h_dot||)')
    axs[1].set_ylabel('km^2/s^2')
    
    # Gráfico 3: Energía Específica
    axs[2].plot(tiempo, e_circ, label='Circular', color='blue')
    axs[2].plot(tiempo, e_geo, label='Geoestacionaria', color='orange')
    axs[2].plot(tiempo, e_mol, label='Molniya', color='green')
    axs[2].set_title('Energía Específica Total (ε)')
    axs[2].set_xlabel('Tiempo [horas]')
    axs[2].set_ylabel('km^2/s^2')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main_2c()