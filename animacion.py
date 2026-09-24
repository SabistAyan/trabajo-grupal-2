import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (necesario para proyección 3d)
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection

from simbolico import r_func, T_func, N_func, B_func, kappa_func, tau_func


# Paleta de colores

COLOR_T = '#e63946'       # rojo coral -> tangente
COLOR_N = '#2a9d8f'       # verde azulado -> normal
COLOR_B = '#3a86ff'       # azul vivo -> binormal
COLOR_PARTICULA = '#1d1d1d'
COLOR_PLANO = '#6c757d'   # gris -> plano osculador
COLOR_RECTA_TANGENTE = '#ffb703'  # ámbar -> recta tangente extendida
CMAP_TRAYECTORIA = 'plasma'   # degradado para la curva de fondo
COLOR_FONDO_PANEL = (0.97, 0.97, 0.99)

# Interruptores para los elementos extra 
MOSTRAR_PLANO_OSCULADOR = True
MOSTRAR_RECTA_TANGENTE = True


# 1. Precalcular todo numéricamente (rápido, nada de sympy aquí)

N_FRAMES = 400
t_vals = np.linspace(0, 2 * np.pi, N_FRAMES)

# r_func(t) devuelve una matriz columna 3x1 -> aplanamos a (3,)
puntos = np.array([np.array(r_func(tv)).flatten() for tv in t_vals])
tangentes = np.array([np.array(T_func(tv)).flatten() for tv in t_vals])
normales = np.array([np.array(N_func(tv)).flatten() for tv in t_vals])
binormales = np.array([np.array(B_func(tv)).flatten() for tv in t_vals])
kappas = np.array([float(kappa_func(tv)) for tv in t_vals])
taus = np.array([float(tau_func(tv)) for tv in t_vals])

xs, ys, zs = puntos[:, 0], puntos[:, 1], puntos[:, 2]


# 2. Preparar la figura 3D

fig = plt.figure(figsize=(9, 8))
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('white')

# Curva completa de fondo, coloreada con un degradado según el avance de t
segmentos = np.stack([puntos[:-1], puntos[1:]], axis=1)  # (N-1, 2, 3)
lc = Line3DCollection(segmentos, cmap=CMAP_TRAYECTORIA, linewidth=2.2)
lc.set_array(t_vals[:-1])
ax.add_collection3d(lc)

# Paneles de fondo más claros y discretos
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.set_facecolor(COLOR_FONDO_PANEL)
    pane.set_edgecolor('lightgray')
ax.grid(True, alpha=0.25)

# Límites fijos para que la cámara no "salte" al animar
margen = 3
ax.set_xlim(xs.min() - margen, xs.max() + margen)
ax.set_ylim(ys.min() - margen, ys.max() + margen)
ax.set_zlim(zs.min() - margen, zs.max() + margen)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Triedro de Frenet sobre r(t)', fontsize=13, fontweight='bold')

# Punto móvil (la partícula)
particula, = ax.plot([], [], [], 'o', color=COLOR_PARTICULA, markersize=8,
                      markeredgecolor='white', markeredgewidth=1)

# Vectores del triedro (se recrean cada frame con quiver, ver update())
quiver_T = None
quiver_N = None
quiver_B = None

# Plano osculador (parche) y recta tangente (línea), se recrean cada frame
plano_osculador = None
recta_tangente = None

# Texto con los valores numéricos de kappa y tau
texto_info = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, fontsize=11,
                        family='monospace')

# Factores puramente visuales (no afectan los valores numéricos mostrados)
ESCALA_VECTORES = 3.0
MEDIO_LADO_PLANO = 2.5   # "radio" del cuadrado que representa el plano osculador
LARGO_RECTA_TANGENTE = 4.0  # cuánto se extiende la recta tangente a cada lado


# 3. Función de actualización por cuadro

def update(frame):
    global quiver_T, quiver_N, quiver_B, plano_osculador, recta_tangente

    px, py, pz = xs[frame], ys[frame], zs[frame]
    p = np.array([px, py, pz])
    Tv, Nv, Bv = tangentes[frame], normales[frame], binormales[frame]

    # Mover la partícula
    particula.set_data([px], [py])
    particula.set_3d_properties([pz])

    # Quitar los elementos del frame anterior (no se pueden "mover", se recrean)
    if quiver_T is not None:
        quiver_T.remove()
        quiver_N.remove()
        quiver_B.remove()
    if plano_osculador is not None:
        plano_osculador.remove()
        plano_osculador = None
    if recta_tangente is not None:
        recta_tangente.remove()
        recta_tangente = None

    quiver_T = ax.quiver(px, py, pz, *Tv, length=ESCALA_VECTORES, color=COLOR_T,
                          linewidth=2.2, arrow_length_ratio=0.25)
    quiver_N = ax.quiver(px, py, pz, *Nv, length=ESCALA_VECTORES, color=COLOR_N,
                          linewidth=2.2, arrow_length_ratio=0.25)
    quiver_B = ax.quiver(px, py, pz, *Bv, length=ESCALA_VECTORES, color=COLOR_B,
                          linewidth=2.2, arrow_length_ratio=0.25)

    # Plano osculador: generado por T y N, con normal B, centrado en la partícula
    if MOSTRAR_PLANO_OSCULADOR:
        s = MEDIO_LADO_PLANO
        vertices = [
            p + s * Tv + s * Nv,
            p + s * Tv - s * Nv,
            p - s * Tv - s * Nv,
            p - s * Tv + s * Nv,
        ]
        plano_osculador = Poly3DCollection([vertices], color=COLOR_PLANO, alpha=0.25)
        ax.add_collection3d(plano_osculador)

    # Recta tangente: pasa por la partícula en dirección T, extendida a ambos lados
    if MOSTRAR_RECTA_TANGENTE:
        L = LARGO_RECTA_TANGENTE
        extremo_1 = p - L * Tv
        extremo_2 = p + L * Tv
        recta_tangente, = ax.plot(
            [extremo_1[0], extremo_2[0]],
            [extremo_1[1], extremo_2[1]],
            [extremo_1[2], extremo_2[2]],
            color=COLOR_RECTA_TANGENTE, linewidth=1.8, linestyle='--'
        )

    texto_info.set_text(
        f"t = {t_vals[frame]:.2f}\n"
        f"kappa (curvatura) = {kappas[frame]:.4f}\n"
        f"tau (torsión)     = {taus[frame]:.4f}"
    )
    texto_info.set_bbox(dict(facecolor='white', alpha=0.75, edgecolor='lightgray', boxstyle='round,pad=0.4'))

    return particula, quiver_T, quiver_N, quiver_B, texto_info


# Leyenda fija (una sola vez, no en cada frame)
ax.plot([], [], color=COLOR_T, linewidth=2.5, label='T (tangente)')
ax.plot([], [], color=COLOR_N, linewidth=2.5, label='N (normal)')
ax.plot([], [], color=COLOR_B, linewidth=2.5, label='B (binormal)')
if MOSTRAR_PLANO_OSCULADOR:
    ax.plot([], [], color=COLOR_PLANO, linewidth=6, alpha=0.4, label='Plano osculador')
if MOSTRAR_RECTA_TANGENTE:
    ax.plot([], [], color=COLOR_RECTA_TANGENTE, linewidth=1.8, linestyle='--', label='Recta tangente')
ax.legend(loc='upper right', framealpha=0.9)


# 4. Crear y mostrar la animación

anim = FuncAnimation(fig, update, frames=N_FRAMES, interval=30, blit=False)

# Para exportar como video/gif (descomentar la que necesites).
# Requiere ffmpeg instalado para mp4, o pillow (ya lo tienes) para gif.
# anim.save('triedro_frenet.mp4', writer='ffmpeg', fps=30, dpi=150)
# anim.save('triedro_frenet.gif', writer='pillow', fps=30)

plt.show()

