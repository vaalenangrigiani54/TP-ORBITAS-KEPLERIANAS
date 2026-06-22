import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Agregamos la carpeta padre al path para poder importar utils y lib
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from punto2.utils import propagar_orbita


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

def main_2b():
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
    main_2b()