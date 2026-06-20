from polinomio import PolinomioCubico
import numpy as np


def interpolar(puntos_X: list[float], puntos_Y: list[float]) -> list[PolinomioCubico]:
    """Interpola mediante Spline Cúbica un conjunto de puntos"""
    
    N = min(len(puntos_X), len(puntos_Y))
    if N < 2:
        return []
    
    matriz_A = np.zeros((3*(N-1), 3*(N-1)), dtype=float)
    matriz_B = np.zeros(3*(N-1), dtype=float)
    fila = 0

    # Interpolación al extremo derecho: S_i(x_i+1) = y_i+1
    # A_i + B_i(x_i+1 - x_i) + C_i(x_i+1 - x_i)² + D_i(x_i+1 - x_i)³ = y_i+1
    # B_i(x_i+1 - x_i) + C_i(x_i+1 - x_i)² + D_i(x_i+1 - x_i)³ = y_i+1 - A_i = y_i+1 - y_i
    for i in range(N-1):
        i_spline = 3*i
        x_i, x_imas1 = puntos_X[i], puntos_X[i+1]
        y_i, y_imas1 = puntos_Y[i], puntos_Y[i+1]
        hx_i = x_imas1 - x_i
        
        matriz_A[fila, i_spline]   = hx_i
        matriz_A[fila, i_spline+1] = hx_i**2
        matriz_A[fila, i_spline+2] = hx_i**3

        matriz_B[fila] = y_imas1 - y_i

        fila += 1

    # Continuidad de la primera derivada: S_i'(x_i+1) = S_i+1'(x_i+1)
    # B_i + 2C_i(x_i+1 - x_i) + 3D_i(x_i+1 - x_i)² = B_i+1 + 2C_i+1(x_i+1 - x_i+1) + 3D_i+1(x_i+1 - x_i+1)²
    # B_i + 2C_i(x_i+1 - x_i) + 3D_i(x_i+1 - x_i)² - B_i+1 = 0
    for i in range(N-2):
        i_spline = 3*i
        x_i, x_imas1 = puntos_X[i], puntos_X[i+1]
        hx_i = x_imas1 - x_i
        
        matriz_A[fila, i_spline]   = 1
        matriz_A[fila, i_spline+1] = 2*hx_i
        matriz_A[fila, i_spline+2] = 3*hx_i**2
        matriz_A[fila, i_spline+3] = -1

        fila += 1

    # Continuidad de la segunda derivada: S_i"(x_i+1) = S_i+1"(x_i+1)
    # 2C_i + 6D_i(x_i+1 - x_i) = 2C_i+1 + 6D_i+1(x_i+1 - x_i+1)
    # 2C_i + 6D_i(x_i+1 - x_i) - 2C_i+1 = 0
    for i in range(N-2):
        i_spline = 3*i
        x_i, x_imas1 = puntos_X[i], puntos_X[i+1]
        hx_i = x_imas1 - x_i

        matriz_A[fila, i_spline+1] = 2
        matriz_A[fila, i_spline+2] = 6*hx_i
        matriz_A[fila, i_spline+4] = -2

        fila += 1

    # Extremo izquierdo: S_1"(x_1) = 0
    # 2C_1 + 6D_1(x_1 - x_1) = 0
    # 2C_1 = 0 <=> C_1 = 0
    matriz_A[fila, 1] = 1
    fila += 1

    # Extremo derecho: S_N-1"(x_N) = 0
    # 2C_N-1 + 6D_N-1(x_N - x_N-1) = 0
    # C_N-1 + 3D_N-1(x_N - x_N-1) = 0
    matriz_A[fila, -2] = 1
    matriz_A[fila, -1] = 3*(puntos_X[N-1] - puntos_X[N-2])
    fila += 1

    # Compruebo que se generó la cantidad correcta de ecuaciones
    assert fila == 3*(N-1)

    # Resolución del sistema de ecuaciones
    solucion = np.linalg.solve(matriz_A, matriz_B)

    # Creación de los polinomios a partir de la solución
    splines = []
    for i in range(N-1):
        i_spline = 3*i
        x_i, x_imas1 = puntos_X[i], puntos_X[i+1]

        splines.append(
            PolinomioCubico(
                A=puntos_Y[i],
                B=solucion[i_spline],
                C=solucion[i_spline+1],
                D=solucion[i_spline+2],
                k=x_i,
                rango_X=(x_i, x_imas1)
            )
        )

    return splines


def interpolar2(puntos: list[tuple[float, float]]) -> list[PolinomioCubico]:
    """Interpola mediante Spline Cúbica un conjunto de puntos"""
    puntos_X, puntos_Y = [], []

    for x, y in puntos:
        puntos_X.append(x)
        puntos_Y.append(y)

    return interpolar(puntos_X, puntos_Y)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    puntos_X: list[float] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    puntos_Y: list[float] = [2, 1, 4, 7, 4, 8, 3, 6, 4, 7]
    splines = interpolar(puntos_X, puntos_Y)
    
    for s in splines:
        s.plot(color="Black")

    plt.scatter(puntos_X, puntos_Y, label="Puntos", zorder=10)
    plt.grid()
    plt.legend()
    plt.title("Spline Cúbica")
    plt.show()