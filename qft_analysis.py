"""
================================================================
  QFT / MECÁNICA CUÁNTICA APLICADA AL MERCADO  —  EXPERIMENTAL
================================================================

ADVERTENCIA: esto es exploratorio. No es un modelo de riesgo validado ni se usa
en un banco. Está separado del resto a propósito.

LA IDEA
-------
Tratar al retorno estandarizado del activo como una partícula cuántica en un pozo
de potencial. Si la densidad de retornos rho(x) es el estado fundamental
|psi_0|^2 de un hamiltoniano H = -1/2 d^2/dx^2 + V(x) (con hbar = m = 1), la
ecuación de Schrödinger estacionaria fija el potencial a partir de R = sqrt(rho):

    -1/2 R''(x) + V(x) R(x) = E_0 R(x)   =>   V(x) = E_0 + 1/2 · R''(x)/R(x)

y el **potencial cuántico de Bohm** es  Q(x) = -1/2 · R''(x)/R(x).

Para que las derivadas sean estables, la densidad se ajusta con una
**Student-t** (la que mejor describe los retornos diarios); con nu -> infinito
se recupera la gaussiana. Todo lo demás (V, Q, omega) sale en forma cerrada:

    R(x) = (1 + x^2/nu)^(-alpha)  con  alpha = (nu+1)/4
    R''(x)/R(x) = -2 alpha/u + 4 alpha(1+alpha) x^2/u^2   con  u = nu + x^2

QUÉ SE LEE (pozo estático)
--------------------------
- **omega** (rigidez del pozo): curvatura de V en el fondo, V ~ 1/2 omega^2 x^2.
  Alto = pozo angosto = reversión a la media fuerte.
- **paredes que se caen**: para la Student-t, V(x) -> 0 cuando x -> infinito
  (no hay pared que confine) — la firma de las colas gruesas: el precio puede
  "escapar" del pozo (salto / crash). Se mide con `confinamiento` (V real en las
  alas / parábola del oscilador): ~1 gaussiano, ~0 cola muy gruesa.
- **prob_mov_3sigma_vs_gauss**: cuántas veces más probable es un movimiento de
  3 sigma según el pozo (Student-t) que según una gaussiana.
- **posición en el pozo** (x_now) y la **fuerza cuántica** -dV/dx que empuja al
  centro. **niveles de energía** E_n = omega (n + 1/2) del oscilador equivalente.

EVOLUCIÓN TEMPORAL (EN DESARROLLO)
---------------------------------
`_propagador` diagonaliza H y evoluciona la posición actual como una difusión en
el pozo (ecuación de Smoluchowski = H en tiempo imaginario):

    P(x, t | x_0) = [phi_0(x)/phi_0(x_0)] · sum_n phi_n(x) phi_n(x_0) e^{-(E_n-E_0) t}

- **energia_gap** = E_1 - E_0: el modo más lento; su inversa es el tiempo de
  relajación. El reloj natural -> días se calibra con la vida media de un AR(1)
  ajustado al movimiento de 5 días (aproximado — de ahí lo de "en desarrollo").
- **pronostico**: media y banda (± 1 sigma) de la distribución a 1 / 5 / 20 días,
  que decae hacia el equilibrio del pozo.

Referencias: Baaquie, "Quantum Finance" (2004); Choustova, "Bohmian mechanics for
financial processes" (2007); Ye & Huang (2008); Risken, "The Fokker-Planck
Equation" (1989) para la conexión Schrödinger <-> Smoluchowski.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

__all__ = ["quantum_well_analysis"]


def _R_ratio(x: np.ndarray, nu: float) -> np.ndarray:
    """R''(x) / R(x) para R = (1 + x^2/nu)^(-alpha), alpha = (nu+1)/4."""
    alpha = (nu + 1.0) / 4.0
    u = nu + x**2
    return -2.0 * alpha / u + 4.0 * alpha * (1.0 + alpha) * x**2 / u**2


def _propagador(x, V, R, x_now, z, dx):
    """Evolución temporal del pozo (EN DESARROLLO).

    Diagonaliza H = -1/2 d^2/dx^2 + V(x) y evoluciona la posición actual como una
    difusión en el pozo (ecuación de Smoluchowski, que es H en tiempo imaginario):

        P(x, t | x_0) = [phi_0(x)/phi_0(x_0)] · sum_n phi_n(x) phi_n(x_0) e^{-(E_n-E_0) t}

    El "reloj" (tiempo natural -> días) se calibra contra la vida media de un
    AR(1) ajustado al movimiento acumulado de 5 días. Esa calibración es
    aproximada — de ahí lo de experimental.
    """
    from scipy import linalg

    n = x.size
    # Laplaciano por diferencias finitas (3 puntos)
    main = np.full(n, -2.0)
    off = np.ones(n - 1)
    lap = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / dx**2
    H = -0.5 * lap + np.diag(V)
    E, phi = linalg.eigh(H)
    # normalizar autofunciones: sum phi^2 dx = 1
    phi = phi / np.sqrt((phi**2).sum(axis=0) * dx)

    gap = float(E[1] - E[0])
    if gap <= 1e-6:
        return None

    i0 = int(np.argmin(np.abs(x - x_now)))
    phi0 = np.clip(np.abs(phi[:, 0]), 1e-9, None)

    # vida media empírica (días) del movimiento acumulado de 5 días
    if z.size > 60:
        m5 = np.convolve(z, np.ones(5), "valid") / np.sqrt(5)
        a, b = m5[:-1], m5[1:]
        phi_ar = float(np.clip(np.dot(a, b) / (np.dot(a, a) + 1e-12), 0.02, 0.98))
        half_dias = float(np.log(2) / -np.log(phi_ar))
    else:
        phi_ar, half_dias = 0.9, 6.6
    half_dias = float(np.clip(half_dias, 1.5, 60.0))

    t_half_nat = np.log(2) / gap
    k = half_dias / t_half_nat            # factor natural -> días

    def P_en(t_nat):
        w = np.exp(-(E - E[0]) * t_nat)
        P = (phi0 / phi0[i0]) * (phi * (phi[i0] * w)).sum(axis=1)
        P = np.clip(P, 0.0, None)
        area = P.sum() * dx
        return P / area if area > 0 else R**2 / ((R**2).sum() * dx)

    horizontes = []
    P_corto = P_largo = None
    for dias in (1.0, 5.0, 20.0):
        t_nat = dias / k
        P = P_en(t_nat)
        media = float((x * P).sum() * dx)
        var = float((x**2 * P).sum() * dx - media**2)
        horizontes.append({"dias": int(dias),
                           "media_sigma": round(media, 3),
                           "banda_sigma": round(float(np.sqrt(max(var, 0))), 3)})
        if dias == 1.0:
            P_corto = P
        if dias == 20.0:
            P_largo = P

    return {
        "energia_gap": round(gap, 4),
        "reversion_half_dias": round(half_dias, 1),
        "ar1_mov5d": round(phi_ar, 3),
        "pronostico": horizontes,
        "curvas_pronostico": {
            "dist_1d": [round(float(v), 5) for v in P_corto],
            "dist_20d": [round(float(v), 5) for v in P_largo],
            "equilibrio": [round(float(v), 5) for v in (R**2 / ((R**2).sum() * dx))],
        },
    }


def quantum_well_analysis(returns, grid_points: int = 221) -> dict:
    """Ajusta el pozo de potencial cuántico a la distribución de retornos.

    ``returns`` son retornos (log o simples); se estandarizan a unidades de sigma.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 120:
        raise ValueError(f"se necesitan >= 120 retornos; hay {r.size}")

    mu, sd = float(r.mean()), float(r.std(ddof=1))
    z = (r - mu) / sd

    # --- densidad: Student-t estandarizada (nu grande -> gaussiana) ---
    # t.fit sobre datos de varianza 1 devuelve scale ~ sqrt((nu-2)/nu), que ya
    # deja la t con varianza unitaria: ese scale ES la escala en unidades de sigma.
    nu_hat, _loc, scale = stats.t.fit(z, floc=0.0)
    nu = float(np.clip(nu_hat, 2.2, 250.0))
    s = float(scale)

    x = np.linspace(-6.0, 6.0, grid_points)
    dx = float(x[1] - x[0])
    xs = x / s  # coordenada interna de la t

    rho = stats.t.pdf(xs, df=nu) / s
    R = np.sqrt(np.clip(rho, 1e-12, None))
    ratio = _R_ratio(xs, nu) / s**2       # R''/R en la escala de x

    V = 0.5 * ratio
    V = V - float(V.min())
    Q = -0.5 * ratio

    # --- oscilador equivalente: curvatura de V en el fondo (forma cerrada) ---
    # V(x) ~ C + [alpha(3+2alpha) / (nu^2 s^4)] x^2  cerca de 0  =>  1/2 omega^2 = eso
    alpha = (nu + 1.0) / 4.0
    half_omega2 = alpha * (3.0 + 2.0 * alpha) / (nu**2 * s**4)
    omega = float(np.sqrt(2.0 * half_omega2))
    x0 = 0.0
    V_qho = 0.5 * omega**2 * x**2

    # --- confinamiento: V real vs parábola del oscilador a 3 sigma ---
    j = int(np.argmin(np.abs(np.abs(x) - 3.0)))
    confinamiento = float(V[j] / (V_qho[j] + 1e-9))

    # --- estado actual: movimiento acumulado de 5 días, estandarizado ---
    mov_5d = float(np.sum(z[-5:]) / np.sqrt(5)) if z.size >= 5 else float(z[-1])
    x_now = float(np.clip(mov_5d, x[0], x[-1]))
    dV = np.gradient(V, dx)
    fuerza = -float(np.interp(x_now, x, dV))
    energia_actual = float(np.interp(x_now, x, V))

    niveles = [round(omega * (n + 0.5), 3) for n in range(4)]

    # --- lecturas (omega ~ 0.5 gaussiano; datos reales suelen dar 1.5-3.5) ---
    rigidez = ("pozo muy angosto — reversión a la media muy fuerte" if omega > 2.5
               else "pozo angosto — reversión a la media fuerte" if omega > 1.5
               else "pozo moderado — reversión leve" if omega > 0.9
               else "pozo ancho — poca reversión, el precio deriva")

    if nu < 4:
        paredes = (f"nu = {nu:.1f}: colas MUY gruesas. El pozo casi no confina "
                   "en las alas — riesgo alto de salto/crash")
    elif nu < 8:
        paredes = f"nu = {nu:.1f}: colas gruesas moderadas; las paredes del pozo se ablandan"
    elif nu < 30:
        paredes = f"nu = {nu:.1f}: colas algo gruesas"
    else:
        paredes = f"nu = {nu:.0f}: distribución casi gaussiana"

    if abs(x_now) < 0.7:
        posicion = "el movimiento de 5 días está cerca del fondo del pozo (equilibrio)"
        señal = "NEUTRAL"
    elif abs(x_now) < 1.6:
        lado = "subió" if x_now > 0 else "bajó"
        señal = "REVERSIÓN BAJISTA" if x_now > 0 else "REVERSIÓN ALCISTA"
        posicion = (f"el precio {lado} {x_now:+.1f} sigma en 5 días; trepó una pared "
                    "del pozo y la fuerza cuántica lo empuja de vuelta")
    else:
        lado = "subió" if x_now > 0 else "bajó"
        señal = "REVERSIÓN BAJISTA (extremo)" if x_now > 0 else "REVERSIÓN ALCISTA (extremo)"
        posicion = (f"el precio {lado} {x_now:+.1f} sigma — muy lejos del equilibrio; "
                    "reversión fuerte esperada, o ruptura del pozo")

    vmax = max(float(np.max(V)), 0.5)
    V_qho_disp = np.clip(V_qho, 0.0, 2.5 * vmax)

    # --- probabilidad de un movimiento de 3 sigma: pozo vs gaussiano ---
    cola_pozo = float(2.0 * stats.t.cdf(-3.0 / s, df=nu))
    cola_gauss = float(2.0 * stats.norm.cdf(-3.0))
    prob_3sigma_x = round(cola_pozo / cola_gauss, 1)

    # --- evolución temporal (en desarrollo) ---
    try:
        prop = _propagador(x, V, R, x_now, z, dx)
    except Exception:
        prop = None

    out = {
        "n_obs": int(r.size),
        "sigma_diaria": round(sd, 5),
        "nu": round(nu, 2),
        "omega": round(omega, 4),
        "confinamiento": round(confinamiento, 3),
        "prob_mov_3sigma_vs_gauss": prob_3sigma_x,
        "x_now_sigma": round(x_now, 3),
        "fuerza_cuantica": round(fuerza, 4),
        "energia_actual": round(energia_actual, 4),
        "niveles_energia": niveles,
        "señal": señal,
        "lectura": {"rigidez": rigidez, "paredes": paredes, "posicion": posicion},
        "curvas": {
            "x": [round(float(v), 3) for v in x],
            "densidad": [round(float(v), 5) for v in rho],
            "potencial": [round(float(v), 4) for v in V],
            "potencial_qho": [round(float(v), 4) for v in V_qho_disp],
            "potencial_cuantico": [round(float(v), 4) for v in Q],
        },
    }
    if prop is not None:
        out["propagador"] = {k: v for k, v in prop.items() if k != "curvas_pronostico"}
        out["curvas"].update(prop["curvas_pronostico"])
    return out


if __name__ == "__main__":
    import yfinance as yf

    tk = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    px = yf.download(tk, period="2y", auto_adjust=True, progress=False)["Close"].dropna()
    ret = np.log(px / px.shift(1)).dropna().to_numpy().ravel()
    res = quantum_well_analysis(ret)
    print(f"\n{'=' * 62}\n  QFT (experimental) — {tk}  ·  {datetime.now():%Y-%m-%d}\n{'=' * 62}")
    print("  POZO ESTÁTICO")
    print(f"    nu (grados de libertad)     : {res['nu']}")
    print(f"    omega (rigidez del pozo)    : {res['omega']}")
    print(f"    confinamiento (1=gauss)     : {res['confinamiento']}")
    print(f"    P(mov 3 sigma) vs gaussiana : {res['prob_mov_3sigma_vs_gauss']}x")
    print(f"    posicion actual (mov 5d)    : {res['x_now_sigma']:+.2f} sigma")
    print(f"    niveles de energia E_n      : {res['niveles_energia']}")
    for v in res["lectura"].values():
        print(f"    - {v}")
    p = res.get("propagador")
    if p:
        print("\n  EVOLUCIÓN TEMPORAL (en desarrollo)")
        print(f"    energia gap (E1-E0)         : {p['energia_gap']}")
        print(f"    vida media de reversion     : ~{p['reversion_half_dias']} dias  (AR1={p['ar1_mov5d']})")
        for h in p["pronostico"]:
            print(f"      {h['dias']:>2}d  ->  media {h['media_sigma']:+.2f} sigma   banda +-{h['banda_sigma']:.2f} sigma")
    print(f"{'=' * 62}\n")
