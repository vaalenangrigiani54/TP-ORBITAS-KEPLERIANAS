import numpy as np

def rk4_step(f, t, y, dt):
    """
    Calcula un paso de integración utilizando el método de Runge-Kutta de 4to orden.

    Parámetros:
    f  : Función que calcula la derivada del estado. Debe tener la firma f(t, y).
    t  : Tiempo actual.
    y  : Vector de estado actual (numpy array).
    dt : Paso de integración (delta t en segundos).

    Retorna:
    y_next : El vector de estado en el tiempo t + dt (numpy array).
    """
    # Calculamos los 4 coeficientes k
    k1 = f(t, y)
    k2 = f(t + dt / 2.0, y + (dt / 2.0) * k1)
    k3 = f(t + dt / 2.0, y + (dt / 2.0) * k2)
    k4 = f(t + dt, y + dt * k3)
    
    # Combinamos para obtener el siguiente estado
    y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    
    return y_next