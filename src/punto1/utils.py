import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
from defs import RADIO_TIERRA_KM


def graficar_interpolacion_posiciones(t, x, y, z, x_interpolado, y_interpolado, z_interpolado, t_max=None, opacidad_puntos=1.0, tamanio_puntos=15, mostrar=True):
    fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axs[0].set_title("Posición X")
    axs[0].scatter(t, x, color="lightcoral", s=tamanio_puntos, alpha=opacidad_puntos, marker='o', zorder=5, label="Puntos X (1s)")
    for idx, s in enumerate(x_interpolado):
        label = "Spline (10min)" if idx == 0 else None
        s.plot(color="black", linewidth=1.0, zorder=10, label=label, ax=axs[0])
    axs[0].set_ylabel("X (km)")
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title("Posición Y")
    axs[1].scatter(t, y, color="lightgreen", s=tamanio_puntos, alpha=opacidad_puntos, marker='o', zorder=5, label="Puntos Y (1s)")
    for idx, s in enumerate(y_interpolado):
        label = "Spline (10min)" if idx == 0 else None
        s.plot(color="black", linewidth=1.0, zorder=10, label=label, ax=axs[1])
    axs[1].set_ylabel("Y (km)")
    axs[1].legend()
    axs[1].grid()

    axs[2].set_title("Posición Z")
    axs[2].scatter(t, z, color="lightblue", s=tamanio_puntos, alpha=opacidad_puntos, marker='o', zorder=5, label="Puntos Z (1s)")
    for idx, s in enumerate(z_interpolado):
        label = "Spline (10min)" if idx == 0 else None
        s.plot(color="black", linewidth=1.0, zorder=10, label=label, ax=axs[2])
    axs[2].set_ylabel("Z (km)")
    axs[2].set_xlabel("Tiempo (s)")
    axs[2].legend()
    axs[2].grid()
    axs[2].xaxis.set_major_locator(MaxNLocator(10))
    axs[2].xaxis.set_minor_locator(MaxNLocator(20))

    if t_max is not None:
        axs[0].set_xlim(0, t_max)

    plt.tight_layout()
    
    if mostrar:
        plt.show()

    return fig, axs


def _funcion_partida(interps, t_array):
    t_array = np.asarray(t_array)
    out = np.full_like(t_array, np.nan, dtype=float)
    for seg in interps:
        seg_start, seg_end = seg.rango_X
        mask = (t_array >= seg_start) & (t_array <= seg_end)
        if mask.any():
            out[mask] = seg(t_array[mask])
    return out


def graficar_diferencias(t_preciso, x_preciso, y_preciso, z_preciso, x_interpolado, y_interpolado, z_interpolado, t_max=None):
    # Primero graficar posiciones e interpolaciones
    fig, axs = graficar_interpolacion_posiciones(
        t_preciso, x_preciso, y_preciso, z_preciso,
        x_interpolado, y_interpolado, z_interpolado,
        t_max=t_max, opacidad_puntos=0.15, tamanio_puntos=6, mostrar=False
    )

    # Calcular errores
    t = np.asarray(t_preciso)
    x_spline = _funcion_partida(x_interpolado, t)
    y_spline = _funcion_partida(y_interpolado, t)
    z_spline = _funcion_partida(z_interpolado, t)

    dx = np.asarray(x_preciso) - x_spline
    dy = np.asarray(y_preciso) - y_spline
    dz = np.asarray(z_preciso) - z_spline

    # Agregar ejes secundarios con errores
    # Eje X
    ax0_twin = axs[0].twinx()
    max_error_x = np.nanmax(np.abs(dx)) if not np.all(np.isnan(dx)) else 1.0
    ax0_twin.plot(t, dx, color='orange', linewidth=1.0, label='Error X', zorder=3)
    ax0_twin.set_ylim(-max_error_x * 1.2, max_error_x * 1.2)
    ax0_twin.set_ylabel('Error X (km)', color='orange')
    ax0_twin.tick_params(axis='y', labelcolor='orange')
    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax0_twin.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Eje Y
    ax1_twin = axs[1].twinx()
    max_error_y = np.nanmax(np.abs(dy)) if not np.all(np.isnan(dy)) else 1.0
    ax1_twin.plot(t, dy, color='orange', linewidth=1.0, label='Error Y', zorder=3)
    ax1_twin.set_ylim(-max_error_y * 1.2, max_error_y * 1.2)
    ax1_twin.set_ylabel('Error Y (km)', color='orange')
    ax1_twin.tick_params(axis='y', labelcolor='orange')
    lines1, labels1 = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    axs[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Eje Z
    ax2_twin = axs[2].twinx()
    max_error_z = np.nanmax(np.abs(dz)) if not np.all(np.isnan(dz)) else 1.0
    ax2_twin.plot(t, dz, color='orange', linewidth=1.0, label='Error Z', zorder=3)
    ax2_twin.set_ylim(-max_error_z * 1.2, max_error_z * 1.2)
    ax2_twin.set_ylabel('Error Z (km)', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    lines1, labels1 = axs[2].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axs[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.show()


def graficar_trayectoria_3d(t, x, y, z, x_interpolado, y_interpolado, z_interpolado, t_max=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if t is not None and t_max is not None:
        t_array = np.asarray(t)
        cutoff = np.searchsorted(t_array, t_max, side='right')
        if cutoff > 0:
            x = x[:cutoff]
            y = y[:cutoff]
            z = z[:cutoff]

    u = np.linspace(0, 2 * np.pi, 32)
    v = np.linspace(0, np.pi, 16)
    X = RADIO_TIERRA_KM * np.outer(np.cos(u), np.sin(v))
    Y = RADIO_TIERRA_KM * np.outer(np.sin(u), np.sin(v))
    Z = RADIO_TIERRA_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(
        X,
        Y,
        Z,
        color='deepskyblue',
        alpha=0.4,
        linewidth=0,
        edgecolor='none',
        antialiased=True,
    )

    orbit_x = []
    orbit_y = []
    orbit_z = []
    for seg_x, seg_y, seg_z in zip(x_interpolado, y_interpolado, z_interpolado):
        seg_start, seg_end = seg_x.rango_X
        if t_max is not None:
            seg_end = min(seg_end, t_max)
        if seg_start >= seg_end:
            continue
        t_segment = np.linspace(seg_start, seg_end, 120)
        orbit_x.append(seg_x(t_segment))
        orbit_y.append(seg_y(t_segment))
        orbit_z.append(seg_z(t_segment))

    if orbit_x:
        orbit_x = np.concatenate(orbit_x)
        orbit_y = np.concatenate(orbit_y)
        orbit_z = np.concatenate(orbit_z)
        ax.plot(orbit_x, orbit_y, orbit_z, color='black', linewidth=1.5, label='Trayectoria interpolada')

    ax.scatter(x, y, z, color='red', s=25, label='Muestras', depthshade=True)
    ax.scatter(x[0], y[0], z[0], color='green', s=50, label='Inicio', depthshade=True)

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Trayectoria orbital completa en 3D')
    ax.legend()
    ax.grid(True)

    trayectoria_x = np.asarray(x)
    trayectoria_y = np.asarray(y)
    trayectoria_z = np.asarray(z)
    max_range = max(
        trayectoria_x.max() - trayectoria_x.min(),
        trayectoria_y.max() - trayectoria_y.min(),
        trayectoria_z.max() - trayectoria_z.min(),
        RADIO_TIERRA_KM * 2,
    )

    mid_x = 0.5 * (trayectoria_x.max() + trayectoria_x.min())
    mid_y = 0.5 * (trayectoria_y.max() + trayectoria_y.min())
    mid_z = 0.5 * (trayectoria_z.max() + trayectoria_z.min())

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=25, azim=130)

    plt.tight_layout()
    plt.show()