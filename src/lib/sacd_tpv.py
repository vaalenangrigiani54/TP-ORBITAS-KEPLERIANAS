from dataclasses import dataclass
from datetime import datetime

@dataclass
class MuestraSACD:
    fecha: str
    hora: str

    pos_X: float
    pos_Y: float
    pos_Z: float

    vel_X: float
    vel_Y: float
    vel_Z: float


def parsear_muestras(ruta_archivo: str) -> list[MuestraSACD]:
    """Convierte el archivo de las muestras en una lista con los datos parseados correctamente"""
    muestras = []

    with open(ruta_archivo) as archivo:
        for linea in archivo:
            datos = linea.split()
            
            if len(datos) == 8:
                muestras.append(
                    MuestraSACD(
                        fecha = datos[0],
                        hora = datos[1],
                        pos_X = float(datos[2]),
                        pos_Y = float(datos[3]),
                        pos_Z = float(datos[4]),
                        vel_X = float(datos[5]),
                        vel_Y = float(datos[6]),
                        vel_Z = float(datos[7])
                    ))
                
    return muestras


def _parse_datetime(fecha: str, hora: str) -> datetime:
    """Parsea `fecha` y `hora` en strings."""

    # Normalizo la parte de hora para tener como máximo 6 dígitos en los microsegundos
    if "." in hora:
        base, frac = hora.split(".", 1)
        frac6 = (frac[:6]).ljust(6, "0")
        hora_fixed = f"{base}.{frac6}"
        s = f"{fecha} {hora_fixed}"
        fmt = "%Y/%m/%d %H:%M:%S.%f"
    else:
        s = f"{fecha} {hora}"
        fmt = "%Y/%m/%d %H:%M:%S"

    return datetime.strptime(s, fmt)


def instantes_relativos(muestras: list[MuestraSACD]) -> list[float]:
    """Devuelve los instantes en segundos relativos al primero (t=0).

    El primer instante siempre es 0.0.
    """
    if not muestras:
        return []

    t0 = _parse_datetime(muestras[0].fecha, muestras[0].hora)
    return [(_parse_datetime(m.fecha, m.hora) - t0).total_seconds() for m in muestras]


def separar_posiciones(muestras: list[MuestraSACD]):
    x, y, z = [], [], []

    for m in muestras:
        x.append(m.pos_X)
        y.append(m.pos_Y)
        z.append(m.pos_Z)

    return x, y, z