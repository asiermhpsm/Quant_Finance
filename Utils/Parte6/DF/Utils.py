import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def _thomas_solver(lower, diag, upper, d):
    """
    Thomas algorithm for tridiagonal system.
    lower: length n-1 (subdiagonal)
    diag:  length n   (diagonal)
    upper: length n-1 (superdiagonal)
    d:     length n   (rhs)
    returns x length n
    """
    n = diag.size
    if n == 0:
        return np.array([])
    # copy to avoid modifying inputs
    a = diag.astype(float).copy()
    b = upper.astype(float).copy()
    c = lower.astype(float).copy()
    d = d.astype(float).copy()

    # forward elimination
    for i in range(1, n):
        if a[i-1] == 0:
            raise np.linalg.LinAlgError("Zero pivot in Thomas solver")
        m = c[i-1] / a[i-1]
        a[i] = a[i] - m * b[i-1]
        d[i] = d[i] - m * d[i-1]

    # back substitution
    x = np.zeros(n, dtype=float)
    if a[-1] == 0:
        raise np.linalg.LinAlgError("Zero pivot in Thomas solver")
    x[-1] = d[-1] / a[-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - b[i] * x[i+1]) / a[i]
    return x

def DF_EDP_parabolica(a, b, c, f, payoff, func_S0=None, func_S_inf=None,
                      func_cond: callable = None,
                      T: float = 1.0, N: int = 100, S_inf: float = 10.0, M: int = 100,
                      theta: float = 0.5):
    """
    Resolución por diferencias finitas de una EDP parabólica unidimensional
    en la forma:
        V_t + a(t,S) V_SS + b(t,S) V_S + c(t,S) V + f(t,S) = 0

    Notas importantes:
    - La malla: t en N+2 puntos (0..T) y S en M+2 puntos (0..S_inf).
    - Se resuelve hacia atrás en el tiempo: dado V_{j+1} se obtiene V_j.
    - A usa coeficientes evaluados en el nivel temporal j; B usa coeficientes en j+1.
    - El término fuente se discretiza como F = theta*f_j + (1-theta)*f_{j+1}.
    - Las funciones a,b,c,f deben aceptar vectores (t_mesh, S_mesh) o devolver escalares.
    - func_S0(t) y func_S_inf(t) deben aceptar array t y devolver array (condiciones de borde
      en S=0 y S=S_inf para todo t). payoff(S) acepta array S.
    """
    # pasos
    Dt = T / (N + 1)
    DS = S_inf / (M + 1)

    alpha = 1.0 / (DS ** 2)
    beta = 1.0 / DS
    gamma = 1.0 / Dt

    # mallas
    t = np.linspace(0.0, T, N + 2)       # tamaño N+2
    S = np.linspace(0.0, S_inf, M + 2)   # tamaño M+2

    # mesh para evaluar coeficientes (shape (M+2, N+2) ) indexing 'ij' hace S en eje 0, t en eje 1
    S_mesh, t_mesh = np.meshgrid(S, t, indexing='xy')  # returns shapes (N+2, M+2) if 'xy'
    t_mesh = t_mesh.T
    S_mesh = S_mesh.T

    # función robusta para evaluar coeficientes (vectorizados o escalares)
    def _eval_coef(fun, name="coef"):
        try:
            vals = fun(t_mesh, S_mesh)  # preferimos (t, S) como en tu código original
            vals = np.asarray(vals, dtype=float)
            if vals.shape != (M + 2, N + 2):
                # intentar broadcasting si devolvió un escalar o 1D
                if vals.shape == ():
                    vals = np.full((M + 2, N + 2), float(vals))
                elif vals.shape == (M + 2,):
                    vals = np.tile(vals.reshape(M + 2, 1), (1, N + 2))
                elif vals.shape == (N + 2,):
                    vals = np.tile(vals.reshape(1, N + 2), (M + 2, 1))
                else:
                    raise ValueError(f"{name} returned array con shape inesperado {vals.shape}")
        except Exception:
            # intentar tomarlo como escalar fun(0,0)
            try:
                s = float(fun(0.0, 0.0))
                vals = np.full((M + 2, N + 2), s, dtype=float)
            except Exception as e:
                raise ValueError(f"No se pudo evaluar el coeficiente {name}: {e}")
        return vals

    a_values = _eval_coef(a, "a")
    b_values = _eval_coef(b, "b")
    c_values = _eval_coef(c, "c")
    f_values = _eval_coef(f, "f")

    # matriz solución: filas -> S (0..M+1), columnas -> t (0..N+1)
    V = np.zeros((M + 2, N + 2), dtype=float)

    #V = func_cond(V)

    # condiciones de contorno
    # func_S0(t) and func_S_inf(t) deben aceptar vector t y devolver vector
    V[0, :] = np.asarray(func_S0(t), dtype=float)
    V[-1, :] = np.asarray(func_S_inf(t), dtype=float)

    # condición terminal (payoff) en t = T -> columna index -1
    V[:, -1] = np.asarray(payoff(S), dtype=float)

    # Prealocar arrays para la parte interior (índices i = 1..M)
    # bucle temporal hacia atrás: j = N, N-1, ..., 0
    for j in range(N, -1, -1):
        # coeficientes en nivel j  (para A)
        eta_j = alpha * a_values[:, j]
        phi_j = 2.0 * alpha * a_values[:, j] + beta * b_values[:, j] - c_values[:, j]
        psi_j = alpha * a_values[:, j] + beta * b_values[:, j]

        # coeficientes en nivel j+1 (para B)
        eta_j1 = alpha * a_values[:, j + 1]
        phi_j1 = 2.0 * alpha * a_values[:, j + 1] + beta * b_values[:, j + 1] - c_values[:, j + 1]
        psi_j1 = alpha * a_values[:, j + 1] + beta * b_values[:, j + 1]

        # diagonales de A (matriz que multiplica V_j)
        # tamaño M (interiores 1..M)
        A_diag = np.empty(M, dtype=float)
        A_lower = np.empty(M - 1, dtype=float)  # subdiagonal (i,i-1)
        A_upper = np.empty(M - 1, dtype=float)  # superdiagonal (i,i+1)

        # relleno A usando indices interiores i=1..M -> array index k=i-1
        for k in range(M):
            i = k + 1
            A_diag[k] = -gamma - theta * phi_j[i]
            if k > 0:
                A_lower[k - 1] = theta * eta_j[i]      # A_{i,i-1}
            if k < M - 1:
                A_upper[k] = theta * psi_j[i]         # A_{i,i+1}

        # construimos B (se usa para computar rhs = B @ V_{j+1})
        B_lower = np.empty(M - 1, dtype=float)
        B_diag = np.empty(M, dtype=float)
        B_upper = np.empty(M - 1, dtype=float)
        for k in range(M):
            i = k + 1
            B_diag[k] = -(gamma - (1.0 - theta) * phi_j1[i])
            if k > 0:
                B_lower[k - 1] = -(1.0 - theta) * eta_j1[i]
            if k < M - 1:
                B_upper[k] = -(1.0 - theta) * psi_j1[i]

        # término fuente ponderado F = theta * f_j + (1-theta) * f_{j+1}
        F = (theta * f_values[1:-1, j] + (1.0 - theta) * f_values[1:-1, j + 1]).astype(float)

        # vector C con contribuciones de condiciones de frontera (solo primeras y últimas ecuaciones)
        C = np.zeros(M, dtype=float)
        # primera ecuación interior i=1 (k=0) usa V[0, *]
        # A subdiagonal que multiplica V_j^0: A_{1,0} = theta * eta_j[1]
        # B subdiagonal similar en nivel j+1: B_{1,0} = -(1-theta) * eta_j1[1]
        C[0] = - (theta * eta_j[1]) * V[0, j] + (-(1.0 - theta) * eta_j1[1]) * V[0, j + 1]

        # última ecuación interior i=M (k=M-1) usa V[M+1, *]
        # A superdiagonal A_{M,M+1} = theta * psi_j[M]
        # B superdiagonal B_{M,M+1} = -(1-theta) * psi_j1[M]
        C[-1] = - (theta * psi_j[M]) * V[-1, j] + (-(1.0 - theta) * psi_j1[M]) * V[-1, j + 1]

        # calcular rhs = B @ y - F + C, donde y = V[1:-1, j+1]
        y = V[1:-1, j + 1]  # tamaño M

        # multiplicación tridiagonal B @ y eficiente
        rhs = np.empty(M, dtype=float)
        for k in range(M):
            val = B_diag[k] * y[k]
            if k > 0:
                val += B_lower[k - 1] * y[k - 1]
            if k < M - 1:
                val += B_upper[k] * y[k + 1]
            rhs[k] = val

        # incorporar -F + C
        rhs = rhs - F + C

        # resolver A x = rhs  (x = V[1:-1, j])
        # para Thomas necesitamos lower (len M-1), diag (M), upper (M-1)
        x = _thomas_solver(A_lower, A_diag, A_upper, rhs)

        V[1:-1, j] = x

    # devolver mallas con la convención: S en filas, t en columnas (como V)
    # S_mesh, t_mesh están en (M+2, N+2)
    return S_mesh, t_mesh, V


class VanillaOption:
    def __init__(self, a, b, c, f,
                 func_cond: callable,
                 payoff, func_S0, func_S_inf,
                 T: float = 1.0, 
                 N: int = 100, S_inf: float = 10.0, M: int = 100,
                 theta: float = 0.5,
                 option_name: str = "Vanilla Option"):
        
        self.option_name = option_name
        self.S, self.t, self.V = DF_EDP_parabolica(a=a, b=b, c=c, f=f, payoff=payoff, func_S0=func_S0, func_S_inf=func_S_inf, T=T, N=N, S_inf=S_inf, M=M, theta=theta)

    def plot(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.S, self.t, self.V, cmap='viridis')
        ax.set_xlabel('S')
        ax.set_ylabel('t')
        ax.set_zlabel('value')
        ax.set_title(self.option_name)
        
        return fig, ax




class CallEuropea(VanillaOption):
    def __init__(self, 
                 sigma=None, r=None, E=None, 
                 a=None, b=None, c=None, f=None,
                 payoff=None, func_S_inf=None, func_S0=None,
                 T=1,N=100, M=100,theta=0.5,option_name="Call Europea",
                 EDE="lognormal"):
        
        if sigma is None: 
            sigma = 0.25
        if r is None: 
            r = 0.05
        if E is None: 
            E = 5

        S_inf = 4*E

        if a is None:
            if EDE == "lognormal":
                a = lambda t, S: sigma**2 * S**2 / 2
            else:
                a = lambda t, S: sigma**2 * S**2 / 2
        if b is None:
            if EDE == "lognormal":
                b = lambda t, S: r * S
            else:
                b = lambda t, S: r * S
        if c is None:
            if EDE == "lognormal":
                c = lambda t, S: -r
            else:
                c = lambda t, S: -r
        if f is None:
            if EDE == "lognormal":
                f = lambda t, S: 0
            else:
                f = lambda t, S: 0

        if payoff is None:
            payoff = lambda S: np.maximum(S - E, 0)
        if func_S_inf is None:
            func_S_inf = lambda t: S_inf - E * np.exp(-r * (T - t))
        if func_S0 is None:
            func_S0 = lambda t: 0

        super().__init__(a=a, b=b, c=c, f=f,
                         payoff=payoff, func_S0=func_S0, func_S_inf=func_S_inf,
                         T=T, S_inf=S_inf, N=N, M=M, theta=theta, option_name=option_name)
        
class PutEuropea(VanillaOption):
    def __init__(self, 
                    sigma=None, r=None, E=None, 
                    a=None, b=None, c=None, f=None,
                    payoff=None, func_S_inf=None, func_S0=None,
                    T=1, N=100, M=100, theta=0.5, option_name="Put Europea",
                    EDE="lognormal"):
        
        if sigma is None: 
            sigma = 0.25
        if r is None: 
            r = 0.05
        if E is None: 
            E = 5

        S_inf = 4 * E

        if a is None:
            if EDE == "lognormal":
                a = lambda t, S: sigma**2 * S**2 / 2
            else:
                a = lambda t, S: sigma**2 * S**2 / 2
        if b is None:
            if EDE == "lognormal":
                b = lambda t, S: r * S
            else:
                b = lambda t, S: r * S
        if c is None:
            if EDE == "lognormal":
                c = lambda t, S: -r
            else:
                c = lambda t, S: -r
        if f is None:
            if EDE == "lognormal":
                f = lambda t, S: 0
            else:
                f = lambda t, S: 0

        if payoff is None:
            payoff = lambda S: np.maximum(E - S, 0)
        if func_S_inf is None:
            func_S_inf = lambda t: 0
        if func_S0 is None:
            func_S0 = lambda t: E * np.exp(-r * (T - t))

        super().__init__(a=a, b=b, c=c, f=f,
                            payoff=payoff, func_S0=func_S0, func_S_inf=func_S_inf,
                            T=T, S_inf=S_inf, N=N, M=M, theta=theta, option_name=option_name)
        

class BinaryCallEuropea(VanillaOption):
    def __init__(self, 
                    sigma=None, r=None, E=None, 
                    a=None, b=None, c=None, f=None,
                    payoff=None, func_S_inf=None, func_S0=None,
                    T=1, N=100, M=100, theta=0.5, option_name="Binary Call Europea",
                    EDE="lognormal"):
        
        if sigma is None: 
            sigma = 0.25
        if r is None: 
            r = 0.05
        if E is None: 
            E = 5

        S_inf = 4 * E

        if a is None:
            if EDE == "lognormal":
                a = lambda t, S: sigma**2 * S**2 / 2
            else:
                a = lambda t, S: sigma**2 * S**2 / 2
        if b is None:
            if EDE == "lognormal":
                b = lambda t, S: r * S
            else:
                b = lambda t, S: r * S
        if c is None:
            if EDE == "lognormal":
                c = lambda t, S: -r
            else:
                c = lambda t, S: -r
        if f is None:
            if EDE == "lognormal":
                f = lambda t, S: 0
            else:
                f = lambda t, S: 0

        if payoff is None:
            payoff = lambda S: np.where(S > E, 1.0, 0.0)
        if func_S_inf is None:
            func_S_inf = lambda t: 1.0
        if func_S0 is None:
            func_S0 = lambda t: 0.0

        super().__init__(a=a, b=b, c=c, f=f,
                            payoff=payoff, func_S0=func_S0, func_S_inf=func_S_inf,
                            T=T, S_inf=S_inf, N=N, M=M, theta=theta, option_name=option_name)
        
class BinaryPutEuropea(VanillaOption):
    def __init__(self, 
                    sigma=None, r=None, E=None, 
                    a=None, b=None, c=None, f=None,
                    payoff=None, func_S_inf=None, func_S0=None,
                    T=1, N=100, M=100, theta=0.5, option_name="Binary Put Europea",
                    EDE="lognormal"):
        
        if sigma is None: 
            sigma = 0.25
        if r is None: 
            r = 0.05
        if E is None: 
            E = 5

        S_inf = 4 * E

        if a is None:
            if EDE == "lognormal":
                a = lambda t, S: sigma**2 * S**2 / 2
            else:
                a = lambda t, S: sigma**2 * S**2 / 2
        if b is None:
            if EDE == "lognormal":
                b = lambda t, S: r * S
            else:
                b = lambda t, S: r * S
        if c is None:
            if EDE == "lognormal":
                c = lambda t, S: -r
            else:
                c = lambda t, S: -r
        if f is None:
            if EDE == "lognormal":
                f = lambda t, S: 0
            else:
                f = lambda t, S: 0

        if payoff is None:
            payoff = lambda S: np.where(S < E, 1.0, 0.0)
        if func_S_inf is None:
            func_S_inf = lambda t: 0.0
        if func_S0 is None:
            func_S0 = lambda t: 1.0

        super().__init__(a=a, b=b, c=c, f=f,
                            payoff=payoff, func_S0=func_S0, func_S_inf=func_S_inf,
                            T=T, S_inf=S_inf, N=N, M=M, theta=theta, option_name=option_name)



call_eur = CallEuropea(N=200, M=200)
fig_call_eur, ax_call_eur = call_eur.plot()

put_eur = PutEuropea(N=200, M=200)
fig_put_eur, ax_put_eur = put_eur.plot()

bin_call_eur = BinaryCallEuropea(N=200, M=200)
fig_bin_call_eur, ax_bin_call_eur = bin_call_eur.plot()

bin_put_eur = BinaryPutEuropea(N=200, M=200)
fig_bin_put_eur, ax_bin_put_eur = bin_put_eur.plot()



plt.show(block=True)


