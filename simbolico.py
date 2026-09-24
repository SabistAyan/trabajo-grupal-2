import sympy as sp

# Definir r(t) (solución obtenida analíticamente a mano)
t = sp.symbols('t', real=True)

x = sp.Rational(23, 2) * sp.cos(t) + sp.Rational(3, 2) * sp.cos(3 * t)
y = sp.Rational(17, 2) * sp.sin(t) + sp.Rational(3, 2) * sp.sin(3 * t)
z = 5 * sp.sin(2 * t) + 4 * sp.sin(4 * t) + 10

r = sp.Matrix([x, y, z])

# Derivadas
r1 = r.diff(t)        # r'(t)
r2 = r.diff(t, 2)      # r''(t)
r3 = r.diff(t, 3)      # r'''(t)


def verificar():
    """Comprueba que r(t) satisface la EDO original y las condiciones iniciales."""
    r2_dado = sp.Matrix([
        -54 * sp.cos(t)**3 + 29 * sp.cos(t),
        54 * sp.sin(t)**3 - 49 * sp.sin(t),
        -20 * sp.sin(2 * t) - 64 * sp.sin(4 * t)
    ])
    diferencia = sp.simplify(sp.expand_trig(r2 - r2_dado))

    print("Verificación r''(t) - r''_dado(t) (debe ser [0,0,0]):", diferencia.T)
    print("r(0)  =", tuple(r.subs(t, 0)), " (debe ser (13, 0, 10))")
    print("r'(0) =", tuple(r1.subs(t, 0)), " (debe ser (0, 13, 26))")


# Triedro de Frenet (simbólico)
norm_r1 = sp.sqrt(r1.dot(r1))              # ||r'(t)||
T_sym = r1 / norm_r1                       # Tangente unitario

cross_12 = r1.cross(r2)                    # r' x r''
norm_cross = sp.sqrt(cross_12.dot(cross_12))
B_sym = cross_12 / norm_cross              # Binormal unitario
N_sym = B_sym.cross(T_sym)                 # Normal unitario = B x T

kappa_sym = norm_cross / norm_r1**3                       # Curvatura
tau_sym = cross_12.dot(r3) / cross_12.dot(cross_12)        # Torsión

# Conversión a funciones numéricas (lambdify) para la animación

r_func = sp.lambdify(t, r, modules='numpy')
T_func = sp.lambdify(t, T_sym, modules='numpy')
N_func = sp.lambdify(t, N_sym, modules='numpy')
B_func = sp.lambdify(t, B_sym, modules='numpy')
kappa_func = sp.lambdify(t, kappa_sym, modules='numpy')
tau_func = sp.lambdify(t, tau_sym, modules='numpy')


# Al ejecutar este archivo directamente: verificar y mostrar valores

if __name__ == "__main__":
    verificar()

    print("\nValores numéricos de ejemplo:")
    for tv in [0, 0.5, 1.0, 2.0, 3.0]:
        print(f"t={tv:>4}  kappa={float(kappa_func(tv)):.5f}   tau={float(tau_func(tv)):.5f}")