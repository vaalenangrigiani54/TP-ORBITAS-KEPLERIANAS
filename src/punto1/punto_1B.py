import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.sacd_tpv import *
from lib.spline_cubica import interpolar
from utils import *


if __name__ == "__main__":
    muestras_10min = parsear_muestras("src/muestras/SACD_TPV_step_600s.txt")
    t = instantes_relativos(muestras_10min)
    x, y, z = separar_posiciones(muestras_10min)

    x_interpolado = interpolar(t, x)
    y_interpolado = interpolar(t, y)
    z_interpolado = interpolar(t, z)

    # Período orbital: Aproximadamente 110 minutos (6600 segundos).
    periodo_orbital = 110 * 60

    graficar_interpolacion_posiciones(
        t,
        x,
        y,
        z,
        x_interpolado,
        y_interpolado,
        z_interpolado,
        t_max=periodo_orbital,
    )

    graficar_trayectoria_3d(
        t,
        x,
        y,
        z,
        x_interpolado,
        y_interpolado,
        z_interpolado,
        t_max=periodo_orbital,
    )