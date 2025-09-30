import numpy as np
from scipy.integrate import quad, solve_ivp, cumulative_trapezoid
import matplotlib.pyplot as plt
from time import perf_counter

# Parámetros y funciones de ejemplo
T = 200        # tiempo final
S = 0          # valor fijo de S
phi = lambda S: 2.0  # condición final V(T,S)

# Funciones c(t,S) y f(t,S)
c = lambda t, S: 0.02   # ejemplo (constante)
f = lambda t, S: 0      # ejemplo (nula)

points = 1000

# Parámetros del solver para mayor precisión
METHOD = 'DOP853'   # opciones: 'DOP853' (no rígido, muy preciso) | 'Radau'/'BDF' (rígidos)
RTOL = 1e-10
ATOL = 1e-12
MAX_STEP = (T - 0) / (10 * points)  # paso máx. más pequeño que el de salida

# Jacobiano d(dVdt)/dV = -c(t,S) (recomendado para métodos implícitos)
def jac(t, y):
    return np.array([[-c(t, S)]])

# Crear una función que calcule V(t,S) usando la solución exacta con integrales
# V(t,S) = phi(S)*exp(∫_t^T c(u,S) du) + ∫_t^T f(u,S) * exp(∫_u^T c(v,S) dv) du
def V_integral(t, S):
    def integral_c(a, b):
        result, _ = quad(lambda u: c(u, S), a, b)
        return result

    term1 = phi(S) * np.exp(integral_c(t, T))

    def integrand(u):
        return f(u, S) * np.exp(integral_c(u, T))

    term2, _ = quad(integrand, t, T)
    return term1 + term2

def V_integral2(t, S):
    # t: puede ser escalar o array
    t = np.atleast_1d(t)
    # Calcula c(u,S) en la malla t
    c_vals = np.array([c(u, S) for u in t])
    # Integral acumulativa de c(u,S) desde 0 hasta cada t
    C = cumulative_trapezoid(c_vals, t, initial=0.0)
    # J(t) = ∫_t^T c(u,S) du = C(T) - C(t)
    # Para esto, extendemos la malla hasta T si no está incluido
    if t[-1] < T:
        t_ext = np.append(t, T)
        c_ext = np.append(c_vals, c(T, S))
        C_ext = cumulative_trapezoid(c_ext, t_ext, initial=0.0)
        C_T = C_ext[-1]
        C = np.interp(t, t_ext, C_ext)
    else:
        C_T = C[-1]
    J = C_T - C
    expJ = np.exp(J)
    # f(u,S) en la malla t
    f_vals = np.array([f(u, S) for u in t])
    h_vals = f_vals * expJ
    # Integral acumulativa de h(u) desde 0 hasta cada t
    F = cumulative_trapezoid(h_vals, t, initial=0.0)
    # H(t) = F(T) - F(t)
    if t[-1] < T:
        F_ext = cumulative_trapezoid(np.append(h_vals, 0), np.append(t, T), initial=0.0)
        F_T = F_ext[-1]
        F = np.interp(t, np.append(t, T), F_ext)
    else:
        F_T = F[-1]
    H = F_T - F
    V = phi(S) * expJ + H
    return V if V.size > 1 else V[0]

# Solución exacta en forma cerrada para los c y f actuales (c=0.02, f=0):
# V(t) = phi(S) * exp(c*(T - t))
def V_exacta_cerrada(t, S):
    return phi(S) * np.exp(0.02 * (T - t))

# Definimos la ODE backward: dV/dt = -c(t,S) V - f(t,S), V(T)=phi(S)
def dVdt(t, V):
    return -c(t, S) * V - f(t, S)

# Malla temporal común
t_grid = np.linspace(0, T, points)

# Tiempo: solución exacta cerrada
t_exact_0 = perf_counter()
V_vals_exacta = np.array([V_exacta_cerrada(t, S) for t in t_grid])
t_exact_1 = perf_counter()
tiempo_exacta = t_exact_1 - t_exact_0

# Tiempo: solución con integrales (evaluación punto a punto)
t0 = perf_counter()
V_vals_integral = np.array([V_integral(t, S) for t in t_grid])
t1 = perf_counter()
tiempo_integrales = t1 - t0

# Añadido: Tiempo: solución con integrales vectorizadas (V_integral2)
t_int2_0 = perf_counter()
V_vals_integral2 = V_integral2(t_grid, S)
t_int2_1 = perf_counter()
tiempo_integrales2 = t_int2_1 - t_int2_0

# Tiempo: solución EDO (solve_ivp) integrando desde T hasta 0
t2 = perf_counter()
solver_kwargs = dict(method=METHOD, rtol=RTOL, atol=ATOL, max_step=MAX_STEP, t_eval=np.linspace(T, 0, points))
if METHOD in ('Radau', 'BDF', 'LSODA'):
    solver_kwargs['jac'] = jac
sol = solve_ivp(dVdt, [T, 0], [phi(S)], **solver_kwargs)
t3 = perf_counter()
tiempo_edo = t3 - t2

# Reordenar solución de la EDO para 0 -> T y alinear con t_grid
t_vals_edo = sol.t[::-1]
V_vals_edo = sol.y[0][::-1]

# Errores respecto a la solución exacta cerrada
err_int = V_vals_integral - V_vals_exacta
err_edo = V_vals_edo - V_vals_exacta

# Añadido: errores para V_integral2
err_int2 = V_vals_integral2 - V_vals_exacta

max_abs_diff_int = np.max(np.abs(err_int))
rmse_int = float(np.sqrt(np.mean(err_int**2)))

# Añadido: métricas V_integral2
max_abs_diff_int2 = np.max(np.abs(err_int2))
rmse_int2 = float(np.sqrt(np.mean(err_int2**2)))

max_abs_diff_edo = np.max(np.abs(err_edo))
rmse_edo = float(np.sqrt(np.mean(err_edo**2)))

print(f"Tiempo solución exacta (cerrada): {tiempo_exacta:.6f} s")
print(f"Tiempo solución con integrales:   {tiempo_integrales:.6f} s")
print(f"Tiempo solución con integrales2:  {tiempo_integrales2:.6f} s")
print(f"Tiempo solución con EDO:          {tiempo_edo:.6f} s")

print(f"Error Integrales vs Exacta -> max: {max_abs_diff_int:.3e}, RMSE: {rmse_int:.3e}")
print(f"Error Integrales2 vs Exacta -> max: {max_abs_diff_int2:.3e}, RMSE: {rmse_int2:.3e}")
print(f"Error EDO vs Exacta        -> max: {max_abs_diff_edo:.3e}, RMSE: {rmse_edo:.3e}")

# Gráfica comparativa
plt.figure(num='Comparación de soluciones y tiempos', figsize=(9, 5))
plt.plot(t_grid, V_vals_exacta, '-',  label='Exacta (cerrada)', linewidth=2, color='k')
plt.plot(t_grid, V_vals_integral, '--', label='Exacta (integrales)', linewidth=2)
# Añadido: curva V_integral2
plt.plot(t_grid, V_vals_integral2, '-.', label='Exacta (integrales trapz)', linewidth=2)
plt.plot(t_vals_edo, V_vals_edo, ':', label='EDO (solve_ivp)', linewidth=2)
plt.xlabel('t')
plt.ylabel('V(t, S)')
plt.title('Comparación de soluciones y tiempos')
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()  # eliminado: mostraremos todas las figuras juntas al final

# === Gráfico adicional: errores respecto a la solución exacta ===
eps = 1e-18  # evita log(0) en la escala log
fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True, num='Errores respecto a la solución exacta')

# Error con signo
axs[0].plot(t_grid, err_int, label='Integrales - Exacta')
# Añadido: error con signo de V_integral2
axs[0].plot(t_grid, err_int2, label='Integrales trapz - Exacta')
axs[0].plot(t_grid, err_edo, label='EDO - Exacta')
axs[0].axhline(0, color='k', lw=0.8)
axs[0].set_ylabel('Error (con signo)')
axs[0].legend()
axs[0].grid(True)

# Error absoluto en escala log
axs[1].semilogy(t_grid, np.abs(err_int) + eps, label='|Error| Integrales')
# Añadido: error abs de V_integral2
axs[1].semilogy(t_grid, np.abs(err_int2) + eps, label='|Error| Integrales trapz')
axs[1].semilogy(t_grid, np.abs(err_edo) + eps, label='|Error| EDO')
axs[1].set_xlabel('t')
axs[1].set_ylabel('|Error| (log)')
axs[1].legend()
axs[1].grid(True, which='both')

plt.tight_layout()
plt.show()  # única llamada: abre las dos ventanas a la vez



