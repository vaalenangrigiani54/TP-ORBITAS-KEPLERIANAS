import sys
import os
import numpy as np
import math

# Aseguramos que el módulo defs y lib estén accesibles desde la carpeta padre.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from defs import MU_TIERRA
from lib.integradores import rk4_step


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


def producto_cruz(a, b):
    """Producto vectorial manual para evitar dependencias adicionales."""
    return np.array([
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ])


def propagar_24hs(P0, V0, paso_dt=1.0):
    """Propaga la órbita estrictamente por 24 horas."""
    pasos_totales = int(24 * 60 * 60 / paso_dt)
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

        a = - (MU_TIERRA / (r_norm**3)) * r
        h_vec = producto_cruz(r, v)
        norm_h[i] = np.linalg.norm(h_vec)

        h_dot_vec = producto_cruz(v, v) + producto_cruz(r, a)
        norm_h_dot[i] = np.linalg.norm(h_dot_vec)

        energia[i] = 0.5 * np.dot(v, v) - (MU_TIERRA / r_norm)

    return norm_h, norm_h_dot, energia
