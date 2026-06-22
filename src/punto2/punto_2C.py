import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Agregamos la carpeta padre al path para poder importar utils y lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from punto2.utils import propagar_24hs, calcular_magnitudes

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