import matplotlib.pyplot as plt
import numpy as np


class PolinomioCubico:
    def __init__(self, A: float, B: float, C: float, D: float, k: float = 0.0, rango_X: tuple[float, float] = (-10.0, 10.0)) -> None:
        """
        Crea un polinomio cúbico de la forma A + B(x-k) + C(x-k)² + D(x-k)³
        - `A`: Término independiente.
        - `B`: Coeficiente de x.
        - `C`: Coeficiente de x².
        - `D`: Coeficiente de x³.
        - `k`: Punto central del polinomio.
        """
        self.k = k
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.rango_X = rango_X

    def plot(self, puntos: int = 256, **kwargs):
        """
        Grafica el polinomio sobre el gráfico actual de `matplotlib`.

        Args:
            puntos: Cantidad de muestras para dibujar la curva
            **kwargs: Argumentos extra para `matplotlib.pyplot.plot()`
        """

        xmin, xmax = self.rango_X
        x = np.linspace(xmin, xmax, puntos)
        y = self(x)

        return plt.plot(x, y, **kwargs)

    def __call__(self, x) -> float:
        dx = x - self.k
        return (self.A + self.B * dx + self.C * dx**2 + self.D * dx**3)

    def __str__(self) -> str:
        rango = " {" + f"{self.rango_X[0]} <= x <= {self.rango_X[1]}" + "}"
        return self.__repr__() + rango

    def __repr__(self) -> str:
        x_menos_k = "x" if self.k == 0.0 else f"(x-{self.k})"
        coef_B = "" if self.B == 0.0 else f" + {self.B}{x_menos_k}"
        coef_C = "" if self.C == 0.0 else f" + {self.C}{x_menos_k}²"
        coef_D = "" if self.D == 0.0 else f" + {self.D}{x_menos_k}³"

        return f"{self.A}{coef_B}{coef_C}{coef_D}"