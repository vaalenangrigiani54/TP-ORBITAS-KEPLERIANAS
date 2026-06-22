import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.sacd_tpv import *
from lib.spline_cubica import interpolar
from utils import *


if __name__ == "__main__":
    muestras_10min = parsear_muestras("src/muestras/SACD_TPV_step_600s.txt")
    muestras_1seg = parsear_muestras("src/muestras/SACD_TPV_step_1s.txt")
    t_10min = instantes_relativos(muestras_10min)
    t_1seg = instantes_relativos(muestras_1seg)
    x_10min, y_10min, z_10min = separar_posiciones(muestras_10min)
    x_1seg, y_1seg, z_1seg = separar_posiciones(muestras_1seg)

    x_interpolado = interpolar(t_10min, x_10min)
    y_interpolado = interpolar(t_10min, y_10min)
    z_interpolado = interpolar(t_10min, z_10min)

    # Período orbital: Aproximadamente 110 minutos (6600 segundos).
    periodo_orbital = 110 * 60

    # Graficar posiciones con splines superpuestos y errores después
    graficar_interpolacion_posiciones(
        t_1seg,
        x_1seg,
        y_1seg,
        z_1seg,
        x_interpolado,
        y_interpolado,
        z_interpolado,
        t_max=periodo_orbital,
        opacidad_puntos = 0.25
    )

    # Graficar posiciones con splines superpuestos y errores después
    graficar_diferencias(
        t_1seg,
        x_1seg,
        y_1seg,
        z_1seg,
        x_interpolado,
        y_interpolado,
        z_interpolado,
        t_max=periodo_orbital,
    )