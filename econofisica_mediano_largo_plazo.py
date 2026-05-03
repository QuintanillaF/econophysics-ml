"""
╔══════════════════════════════════════════════════════════════════╗
║     SISTEMA DE ECONOFÍSICA — MEDIANO Y LARGO PLAZO              ║
║     Stocks & Crypto │ Meses / Años                              ║
╚══════════════════════════════════════════════════════════════════╝

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    print("pip install yfinance")

from scipy import stats, signal
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from scipy.linalg import eigh
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller


# ══════════════════════════════════════════════════════════════
# MÓDULO 1: DATA LOADER — LARGO PLAZO
# ══════════════════════════════════════════════════════════════
class DataLoader:
    """
    Descarga datos históricos largos.

    Períodos recomendados:
        Mediano plazo:  '2y', '3y'
        Largo plazo:    '5y', '10y', 'max'
    """

    PERIODOS_VALIDOS = ['1y','2y','3y','5y','10y','max']

    def __init__(self, ticker: str, periodo: str = '5y'):
        self.ticker   = ticker
        self.periodo  = periodo
        self.datos    = None
        self.retornos = None
        self.precios  = None

    def descargar(self) -> pd.DataFrame:
        print(f"📥 Descargando {self.ticker} ({self.periodo})...")
        self.datos = yf.download(
            self.ticker, period=self.periodo,
            auto_adjust=True, progress=False
        )
        if self.datos.empty:
            raise ValueError(f"Sin datos para {self.ticker}")

        self.precios  = self.datos['Close'].squeeze()
        self.retornos = np.log(self.precios / self.precios.shift(1)).dropna()

        n = len(self.retornos)
        print(f"   ✓ {n} días  │  "
              f"{self.retornos.index[0].date()} → "
              f"{self.retornos.index[-1].date()}")

        if n < 252:
            print("   Menos de 1 año de datos — algunos análisis "
                  "pueden ser poco confiables.")
        return self.datos

    def retornos_mensuales(self) -> pd.Series:
        """Agrega retornos a frecuencia mensual (más estable para largo plazo)."""
        return self.precios.resample('ME').last().pct_change().dropna()

    def estadisticas(self) -> dict:
        r = self.retornos
        años = len(r) / 252
        return {
            'años_datos':        round(años, 1),
            'retorno_anual':     float((1 + r.mean())**252 - 1),
            'vol_anual':         float(r.std() * np.sqrt(252)),
            'sharpe':            float(r.mean() / r.std() * np.sqrt(252)),
            'asimetria':         float(r.skew()),
            'curtosis':          float(r.kurt()),
            'max_drawdown':      self._drawdown(),
            'calmar_ratio':      self._calmar(),
        }

    def _drawdown(self) -> float:
        peak = self.precios.cummax()
        return float(((self.precios - peak) / peak).min())

    def _calmar(self) -> float:
        """Calmar Ratio = Retorno anual / |Max Drawdown|."""
        dd = abs(self._drawdown())
        ret = float((1 + self.retornos.mean())**252 - 1)
        return ret / dd if dd > 0 else 0.0


# ══════════════════════════════════════════════════════════════
# MÓDULO 2: HURST ROBUSTO + DFA
# ══════════════════════════════════════════════════════════════
class AnalisisHurst:
    """
    Dos métodos complementarios para largo plazo:

    1. R/S clásico   — rápido, robusto para series largas
    2. DFA           — Detrended Fluctuation Analysis
                       Más preciso en presencia de tendencias no estacionarias
                       (muy común en stocks y crypto a largo plazo)

    INTERPRETACIÓN (igual que corto plazo pero más confiable con más datos):
        H < 0.45  → Mean-reverting  → estrategia de reversión
        H ≈ 0.50  → Aleatorio
        H > 0.55  → Persistente     → seguir tendencia
    """

    def __init__(self, retornos: pd.Series):
        self.retornos = np.array(retornos)
        self.H_rs  = None
        self.H_dfa = None

    # ── R/S Analysis ─────────────────────────────────────────
    def calcular_rs(self, n_escalas: int = 25) -> float:
        serie = self.retornos
        N     = len(serie)

        min_e = max(20, int(N * 0.01))
        max_e = int(N / 4)
        escalas = np.unique(
            np.logspace(np.log10(min_e), np.log10(max_e),
                        n_escalas).astype(int)
        )

        self._escalas_rs = []
        self._rs_vals    = []

        for n in escalas:
            n_bloques = N // n
            rs_list   = []
            for i in range(n_bloques):
                bloque = serie[i*n:(i+1)*n]
                std    = np.std(bloque, ddof=1)
                if std < 1e-10:
                    continue
                dev = np.cumsum(bloque - bloque.mean())
                rs_list.append((dev.max() - dev.min()) / std)
            if rs_list:
                self._escalas_rs.append(n)
                self._rs_vals.append(np.mean(rs_list))

        log_n  = np.log(self._escalas_rs)
        log_rs = np.log(self._rs_vals)
        self.H_rs, *_ = stats.linregress(log_n, log_rs)
        return self.H_rs

    # ── DFA (Detrended Fluctuation Analysis) ────────────────
    def calcular_dfa(self, n_escalas: int = 25) -> float:
       
        serie = np.cumsum(self.retornos - self.retornos.mean())
        N     = len(serie)

        min_e = max(10, int(N * 0.01))
        max_e = int(N / 5)
        escalas = np.unique(
            np.logspace(np.log10(min_e), np.log10(max_e),
                        n_escalas).astype(int)
        )

        self._escalas_dfa = []
        self._F_dfa       = []

        for n in escalas:
            n_seg    = N // n
            fluct    = []
            for i in range(n_seg):
                seg  = serie[i*n:(i+1)*n]
                x    = np.arange(n)
                # Regresión lineal local (detrending)
                coef = np.polyfit(x, seg, 1)
                tend = np.polyval(coef, x)
                fluct.append(np.sqrt(np.mean((seg - tend)**2)))

            if fluct:
                self._escalas_dfa.append(n)
                self._F_dfa.append(np.mean(fluct))

        log_n = np.log(self._escalas_dfa)
        log_F = np.log(self._F_dfa)
        self.H_dfa, *_ = stats.linregress(log_n, log_F)
        return self.H_dfa

    def calcular(self) -> dict:
        self.calcular_rs()
        self.calcular_dfa()
        H_consenso = (self.H_rs + self.H_dfa) / 2
        return {
            'H_rs':       float(self.H_rs),
            'H_dfa':      float(self.H_dfa),
            'H_consenso': float(H_consenso),
            'interpretacion': self._interpretar(H_consenso),
        }

    def _interpretar(self, H: float) -> str:
        if H < 0.40:
            return f"H={H:.3f} │ FUERTE REVERSIÓN — comprar mínimos, vender máximos"
        elif H < 0.48:
            return f"H={H:.3f} │ Leve reversión — ciclos de sobre/subvaloración"
        elif H < 0.52:
            return f"H={H:.3f} │ ALEATORIO — sin ventaja estadística clara"
        elif H < 0.62:
            return f"H={H:.3f} │ PERSISTENTE — tendencias duraderas (buy & hold)"
        else:
            return f"H={H:.3f} │ MUY PERSISTENTE — momentum de largo plazo fuerte"


# ══════════════════════════════════════════════════════════════
# MÓDULO 3: DETECTOR DE BURBUJAS — MODELO LPPLS (SORNETTE)
# ══════════════════════════════════════════════════════════════
class DetectorBurbuja:
    """
    Implementa el modelo Log-Periodic Power Law Singularity (LPPLS)
    de Didier Sornette para detectar burbujas especulativas.

    IDEA FÍSICA:
        En una burbuja, el precio sigue un crecimiento super-exponencial
        con oscilaciones log-periódicas que convergen hacia un tiempo
        crítico tc (crash inminente).

    FÓRMULA:
        ln P(t) = A + B(tc-t)^m + C(tc-t)^m * cos(ω*ln(tc-t) + φ)

    PARÁMETROS:
        tc  → tiempo crítico estimado (cuando colapsa)
        m   → exponente (0 < m < 1 en burbuja real)
        ω   → frecuencia log-periódica (típico: 6 < ω < 13)
        B   → debe ser negativo (crecimiento acelerado hacia tc)

    SEÑAL DE BURBUJA: B < 0, 0.1 < m < 0.9, 6 < ω < 13
    """

    def __init__(self, precios: pd.Series):
        self.precios = precios
        self.t       = np.arange(len(precios)) / len(precios)  # normalizado [0,1]
        self.log_p   = np.log(precios.values)
        self.params  = None
        self.confianza_burbuja = 0.0

    def _lppls(self, t, tc, m, omega, A, B, C, phi):
        """Función LPPLS completa."""
        dt = np.maximum(tc - t, 1e-8)
        return A + B * dt**m + C * dt**m * np.cos(omega * np.log(dt) + phi)

    def _residuo(self, params_nl):
        """
        Ajuste en dos pasos:
        1. Parámetros lineales (A, B, C) → mínimos cuadrados analítico
        2. Parámetros no lineales (tc, m, ω, φ) → optimización global
        """
        tc, m, omega, phi = params_nl
        t  = self.t
        lp = self.log_p

        if tc <= t[-1] or tc > 1.5:
            return 1e10
        if not (0.01 < m < 0.99):
            return 1e10
        if not (2 < omega < 25):
            return 1e10

        dt  = np.maximum(tc - t, 1e-8)
        f1  = dt**m
        f2  = f1 * np.cos(omega * np.log(dt) + phi)
        f3  = f1 * np.sin(omega * np.log(dt) + phi)

        # Mínimos cuadrados para A, B, C
        X   = np.column_stack([np.ones_like(t), f1, f2])
        try:
            coef, res, _, _ = np.linalg.lstsq(X, lp, rcond=None)
            fitted = X @ coef
            return float(np.sum((lp - fitted)**2))
        except Exception:
            return 1e10

    def ajustar(self, n_intentos: int = 50) -> dict:
        """
        Ajuste robusto usando optimización diferencial global.
        Múltiples intentos para evitar mínimos locales.
        """
        print(f"    Ajustando modelo LPPLS ({n_intentos} intentos)...")

        bounds = [
            (self.t[-1] + 0.001, self.t[-1] + 0.5),  # tc
            (0.1,  0.9),                                # m
            (4.0, 15.0),                                # omega
            (0.0,  2*np.pi),                            # phi
        ]

        resultado = differential_evolution(
            self._residuo, bounds,
            maxiter=n_intentos * 10,
            popsize=12,
            seed=42,
            tol=1e-6,
        )

        tc_opt, m_opt, omega_opt, phi_opt = resultado.x

        # Reconstruir parámetros lineales
        dt  = np.maximum(tc_opt - self.t, 1e-8)
        f1  = dt**m_opt
        f2  = f1 * np.cos(omega_opt * np.log(dt) + phi_opt)
        X   = np.column_stack([np.ones(len(self.t)), f1, f2])
        coef, _, _, _ = np.linalg.lstsq(X, self.log_p, rcond=None)
        A_opt, B_opt, C_opt = coef

        # Tiempo crítico en días desde hoy
        n_dias_total = len(self.precios)
        tc_dias = int((tc_opt - self.t[-1]) * n_dias_total)

        # Verificar condiciones de burbuja
        es_burbuja = (
            B_opt < 0 and
            0.1 < m_opt < 0.9 and
            6 < omega_opt < 13
        )

        # Score de confianza [0, 1]
        score = 0.0
        if B_opt < 0:          score += 0.35
        if 0.1 < m_opt < 0.9:  score += 0.30
        if 6 < omega_opt < 13: score += 0.25
        if tc_dias > 0:        score += 0.10
        self.confianza_burbuja = score

        self.params = {
            'tc_normalizado':  float(tc_opt),
            'tc_dias_desde_hoy': tc_dias,
            'A': float(A_opt), 'B': float(B_opt),
            'C': float(C_opt), 'm': float(m_opt),
            'omega': float(omega_opt), 'phi': float(phi_opt),
        }

        return {
            **self.params,
            'es_burbuja':         es_burbuja,
            'confianza_burbuja':  score,
            'interpretacion':     self._interpretar(es_burbuja, score, tc_dias),
        }

    def _interpretar(self, es_burbuja: bool, score: float, tc_dias: int) -> str:
        if not es_burbuja or score < 0.5:
            return "Sin señal clara de burbuja especulativa."
        elif score < 0.7:
            return (f"Señal débil de burbuja (confianza {score:.0%}). "
                    f"Tiempo crítico estimado: ~{tc_dias} días.")
        else:
            return (f" SEÑAL DE BURBUJA (confianza {score:.0%}). "
                    f"Tiempo crítico estimado: ~{tc_dias} días. "
                    "Considerar reducir exposición gradualmente.")

    def curva_ajustada(self) -> np.ndarray:
        """Retorna la curva LPPLS ajustada para graficar."""
        if self.params is None:
            return None
        p = self.params
        dt  = np.maximum(p['tc_normalizado'] - self.t, 1e-8)
        fitted = (p['A'] + p['B'] * dt**p['m'] +
                  p['C'] * dt**p['m'] *
                  np.cos(p['omega'] * np.log(dt) + p['phi']))
        return np.exp(fitted)


# ══════════════════════════════════════════════════════════════
# MÓDULO 4: ANÁLISIS DE CICLOS CON FFT
# ══════════════════════════════════════════════════════════════
class AnalisisCiclos:
    

    def __init__(self, precios: pd.Series):
        self.precios = precios
        self.ciclos  = None

    def analizar(self, n_ciclos_top: int = 5) -> dict:
        """
        Aplica FFT a la serie de precios desestacionalizada.

        Args:
            n_ciclos_top: Cuántos ciclos dominantes reportar
        """
        # Trabajar sobre retornos suavizados para reducir ruido
        serie = np.log(self.precios.values)

        # Remover tendencia lineal antes de FFT
        x     = np.arange(len(serie))
        coef  = np.polyfit(x, serie, 1)
        serie_detrend = serie - np.polyval(coef, x)

        # Aplicar ventana de Hanning (reduce artefactos de borde)
        ventana = np.hanning(len(serie_detrend))
        serie_ventaneada = serie_detrend * ventana

        # FFT
        fft_vals  = np.fft.rfft(serie_ventaneada)
        freqs     = np.fft.rfftfreq(len(serie_ventaneada))
        potencia  = np.abs(fft_vals)**2

        # Ignorar frecuencia 0 (componente DC)
        potencia[0] = 0

        # Top ciclos por potencia espectral
        idx_top = np.argsort(potencia)[::-1][:n_ciclos_top]

        ciclos = []
        for idx in idx_top:
            f = freqs[idx]
            if f > 0:
                periodo_dias = 1.0 / f
                ciclos.append({
                    'periodo_dias':  round(periodo_dias, 1),
                    'periodo_meses': round(periodo_dias / 21, 1),
                    'potencia_rel':  float(potencia[idx] / potencia.max()),
                })

        self.ciclos    = ciclos
        self.freqs     = freqs
        self.potencia  = potencia
        self.serie_detrend = serie_detrend

        return {
            'ciclos_dominantes': ciclos,
            'interpretacion':    self._interpretar(ciclos),
        }

    def _interpretar(self, ciclos: list) -> str:
        if not ciclos:
            return "No se detectaron ciclos significativos."
        c = ciclos[0]
        m = c['periodo_meses']
        if m < 1:
            return f"Ciclo dominante: {c['periodo_dias']:.0f} días (~{m:.1f} mes)"
        elif m < 6:
            return f"Ciclo dominante: ~{m:.0f} meses — ciclo trimestral/semestral"
        elif m < 18:
            return f"Ciclo dominante: ~{m:.0f} meses — ciclo anual"
        else:
            return f"Ciclo dominante: ~{m:.0f} meses (~{m/12:.1f} años) — ciclo largo"


# ══════════════════════════════════════════════════════════════
# MÓDULO 5: RANDOM MATRIX THEORY (RMT)
# ══════════════════════════════════════════════════════════════
class RMT_Correlaciones:
   

    def __init__(self, retornos_df: pd.DataFrame):
        self.retornos = retornos_df.dropna()
        self.T, self.N = self.retornos.shape
        self.corr_raw    = None
        self.corr_limpia = None
        self.eigenvalores = None

    def analizar(self) -> dict:
        """Aplica filtro RMT a la matriz de correlaciones."""
        # Matriz de correlación empírica
        self.corr_raw = self.retornos.corr().values

        # Eigendescomposición
        eigenvalores, eigenvectores = eigh(self.corr_raw)
        self.eigenvalores = eigenvalores

        # Límites de Marchenko-Pastur
        q       = self.T / self.N
        sigma2  = 1.0  # correlaciones normalizadas
        lambda_max = sigma2 * (1 + 1/q + 2*np.sqrt(1/q))
        lambda_min = sigma2 * (1 + 1/q - 2*np.sqrt(1/q))

        self.lambda_max = lambda_max
        self.lambda_min = max(lambda_min, 0)

        # Separar señal de ruido
        mask_señal = eigenvalores > lambda_max
        n_señal    = mask_señal.sum()

        # Reconstruir matriz limpia (solo eigenvalores de señal)
        evals_limpios = np.where(mask_señal, eigenvalores, 0)
        self.corr_limpia = (
            eigenvectores @
            np.diag(evals_limpios) @
            eigenvectores.T
        )
        # Normalizar diagonal a 1
        d = np.sqrt(np.diag(self.corr_limpia))
        d[d == 0] = 1
        self.corr_limpia = self.corr_limpia / np.outer(d, d)
        np.fill_diagonal(self.corr_limpia, 1.0)

        pct_ruido = float((~mask_señal).sum() / self.N * 100)

        return {
            'n_activos':        self.N,
            'n_observaciones':  self.T,
            'q_ratio':          round(q, 2),
            'lambda_max_mp':    round(lambda_max, 3),
            'n_eigenvalores_señal': int(n_señal),
            'pct_correlacion_ruido': round(pct_ruido, 1),
            'interpretacion':   (
                f"{pct_ruido:.0f}% de correlaciones son ruido estadístico. "
                f"{n_señal} eigenvalores contienen información real."
            )
        }


# ══════════════════════════════════════════════════════════════
# MÓDULO 6: OPTIMIZADOR ROBUSTO CON RESTRICCIÓN DE DRAWDOWN
# ══════════════════════════════════════════════════════════════
class OptimizadorRobust:
    

    def __init__(self, retornos_df: pd.DataFrame,
                 corr_limpia: np.ndarray = None):
        self.retornos     = retornos_df.dropna()
        self.tickers      = list(retornos_df.columns)
        self.n            = len(self.tickers)
        self.corr_limpia  = corr_limpia
        self.pesos        = None

    def _cov_robusta(self) -> np.ndarray:
        """Usa correlación RMT si está disponible, si no la empírica."""
        stds = self.retornos.std().values
        if self.corr_limpia is not None:
            return np.outer(stds, stds) * self.corr_limpia
        return self.retornos.cov().values

    def optimizar(self, lambda_riesgo: float = 2.0,
                  max_drawdown: float = 0.20) -> dict:
        """
        Args:
            lambda_riesgo: Aversión al riesgo (mayor = más conservador)
            max_drawdown:  Drawdown máximo tolerado (ej: 0.20 = 20%)
        """
        cov = self._cov_robusta()
        mu  = self.retornos.mean().values * 252   # retornos anualizados

        def objetivo(w):
            entropia = -np.sum(w * np.log(w + 1e-10))
            riesgo   = w @ cov @ w * 252
            return -entropia + lambda_riesgo * riesgo

        # Restricciones
        restricciones = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]
        limites = [(0.02, 0.50)] * self.n

        w0 = np.ones(self.n) / self.n
        res = minimize(objetivo, w0, method='SLSQP',
                       bounds=limites, constraints=restricciones,
                       options={'maxiter': 2000, 'ftol': 1e-10})
        self.pesos = res.x

        # Métricas
        ret_p   = float(np.sum(mu * self.pesos))
        vol_p   = float(np.sqrt(self.pesos @ cov @ self.pesos * 252))
        sharpe  = ret_p / vol_p if vol_p > 0 else 0
        dd_est  = vol_p * np.sqrt(2 / np.pi)  # estimación analítica drawdown

        return {
            'pesos':               dict(zip(self.tickers, self.pesos)),
            'retorno_anual_est':   ret_p,
            'volatilidad_anual':   vol_p,
            'sharpe_ratio':        sharpe,
            'drawdown_estimado':   dd_est,
            'cumple_dd_constraint': dd_est <= max_drawdown,
            'entropia':            float(-np.sum(self.pesos *
                                                 np.log(self.pesos + 1e-10))),
        }


# ══════════════════════════════════════════════════════════════
# MÓDULO 7: COINTEGRACIÓN (RELACIONES DE LARGO PLAZO)
# ══════════════════════════════════════════════════════════════
class AnalisisCoIntegracion:
    

    def __init__(self, precios_df: pd.DataFrame):
        """
        Args:
            precios_df: DataFrame con columnas = activos, índice = fechas
        """
        self.precios  = precios_df.dropna()
        self.tickers  = list(precios_df.columns)
        self.pares    = []

    def analizar_todos_los_pares(self, alpha: float = 0.05) -> list:
        """Testea cointegración para todos los pares posibles."""
        n = len(self.tickers)
        resultados = []

        for i in range(n):
            for j in range(i+1, n):
                t1 = self.tickers[i]
                t2 = self.tickers[j]
                res = self._testear_par(t1, t2, alpha)
                resultados.append(res)

        # Ordenar por p-valor (más significativos primero)
        resultados.sort(key=lambda x: x['p_valor'])
        self.pares = resultados
        return resultados

    def _testear_par(self, t1: str, t2: str, alpha: float) -> dict:
        """Test de cointegración de Engle-Granger para un par."""
        s1 = np.log(self.precios[t1].values)
        s2 = np.log(self.precios[t2].values)

        # Test ADF individual
        adf1 = adfuller(s1, autolag='AIC')
        adf2 = adfuller(s2, autolag='AIC')

        # Test de cointegración
        score, p_valor, crit = coint(s1, s2)

        # Calcular spread y half-life de reversión
        modelo = sm.OLS(s1, sm.add_constant(s2)).fit()
        hedge_ratio = modelo.params[1]
        spread      = s1 - hedge_ratio * s2
        spread_s    = pd.Series(spread)
        half_life   = self._half_life(spread_s)

        return {
            'par':           f"{t1}/{t2}",
            't1':            t1,
            't2':            t2,
            'p_valor':       float(p_valor),
            'cointegrados':  p_valor < alpha,
            'hedge_ratio':   float(hedge_ratio),
            'half_life_dias': int(half_life) if half_life > 0 else None,
            'spread_zscore': float((spread[-1] - spread.mean()) / spread.std()),
            'señal_trading': self._señal_spread(
                (spread[-1] - spread.mean()) / spread.std(),
                p_valor < alpha, half_life
            ),
        }

    def _half_life(self, spread: pd.Series) -> float:
        """Half-life de reversión a la media (días)."""
        spread_lag  = spread.shift(1).dropna()
        spread_ret  = (spread - spread.shift(1)).dropna()
        spread_lag  = spread_lag[-len(spread_ret):]
        modelo      = sm.OLS(spread_ret, sm.add_constant(spread_lag)).fit()
        lam = modelo.params.iloc[1]
        if lam >= 0:
            return float('inf')
        return float(-np.log(2) / lam)

    def _señal_spread(self, z: float, cointegrado: bool,
                      hl: float) -> str:
        if not cointegrado:
            return "Sin señal — par no cointegrado"
        if hl > 252:
            return "Half-life muy largo (>1 año) — poco útil para trading"
        if z > 2.0:
            return f"VENDER {self.tickers[0]}, COMPRAR {self.tickers[1]} (z={z:.2f})"
        elif z < -2.0:
            return f"COMPRAR {self.tickers[0]}, VENDER {self.tickers[1]} (z={z:.2f})"
        else:
            return f"Spread neutral (z={z:.2f}) — esperar divergencia"


# ══════════════════════════════════════════════════════════════
# MÓDULO 8: DASHBOARD COMPLETO
# ══════════════════════════════════════════════════════════════
class Dashboard:
    

    def __init__(self, ticker: str, periodo: str = '5y'):
        self.ticker  = ticker
        self.periodo = periodo
        self.loader  = None
        self.hurst   = None
        self.burbuja = None
        self.ciclos  = None

    def analizar(self):
        self.loader = DataLoader(self.ticker, self.periodo)
        self.loader.descargar()

        r = self.loader.retornos
        p = self.loader.precios

        print("  Calculando Hurst (R/S + DFA)...")
        self.hurst = AnalisisHurst(r)

        print("   🫧  Ajustando modelo LPPLS (burbujas)...")
        self.burbuja = DetectorBurbuja(p)

        print("   〰️  Analizando ciclos (FFT)...")
        self.ciclos = AnalisisCiclos(p)

        return self

    def imprimir_reporte(self):
        sep = "─" * 62
        print(f"\n{'═'*62}")
        print(f"  REPORTE MEDIANO/LARGO PLAZO: {self.ticker}  ({self.periodo})")
        print(f"{'═'*62}")

        # Estadísticas
        st = self.loader.estadisticas()
        print(f"\n ESTADÍSTICAS ({st['años_datos']} años de datos)")
        print(sep)
        print(f"  Retorno anual (CAGR):  {st['retorno_anual']*100:+.1f}%")
        print(f"  Volatilidad anual:     {st['vol_anual']*100:.1f}%")
        print(f"  Sharpe ratio:          {st['sharpe']:.2f}")
        print(f"  Max Drawdown:          {st['max_drawdown']*100:.1f}%")
        print(f"  Calmar Ratio:          {st['calmar_ratio']:.2f}")

        # Hurst
        h = self.hurst.calcular()
        print(f"\n EXPONENTE DE HURST (consenso R/S + DFA)")
        print(sep)
        print(f"  H_RS  = {h['H_rs']:.3f}")
        print(f"  H_DFA = {h['H_dfa']:.3f}")
        print(f"  ▶ {h['interpretacion']}")

        # Burbujas
        b = self.burbuja.ajustar()
        print(f"\n🫧  MODELO LPPLS — DETECCIÓN DE BURBUJA")
        print(sep)
        print(f"  ▶ {b['interpretacion']}")
        if b['es_burbuja']:
            print(f"  m={b['m']:.2f}, ω={b['omega']:.2f}, B={b['B']:.4f}")

        # Ciclos
        c = self.ciclos.analizar()
        print(f"\n〰️  ANÁLISIS DE CICLOS (FFT)")
        print(sep)
        print(f"  ▶ {c['interpretacion']}")
        for i, ciclo in enumerate(c['ciclos_dominantes'][:3], 1):
            print(f"  {i}. {ciclo['periodo_dias']:.0f} días "
                  f"({ciclo['periodo_meses']:.1f} meses)  "
                  f"— potencia relativa: {ciclo['potencia_rel']:.2f}")

        # Señal integrada
        print(f"\n SEÑAL INTEGRADA (LARGO PLAZO)")
        print(sep)
        self._señal_integrada(h, b, st)
        print(f"{'═'*62}\n")

    def _señal_integrada(self, h: dict, b: dict, st: dict):
        puntos = 0
        notas  = []

        H = h['H_consenso']
        if H > 0.55:
            puntos += 2
            notas.append(f" Hurst={H:.2f}: tendencias persistentes (favorable buy & hold)")
        elif H < 0.45:
            puntos -= 1
            notas.append(f"↩️  Hurst={H:.2f}: reversión a la media — DCA conveniente")
        else:
            notas.append(f"➖ Hurst={H:.2f}: comportamiento aleatorio")

        if b['es_burbuja'] and b['confianza_burbuja'] > 0.65:
            puntos -= 2
            notas.append(f"  BURBUJA detectada (confianza {b['confianza_burbuja']:.0%})")
        else:
            puntos += 1
            notas.append(" Sin señal de burbuja")

        if st['calmar_ratio'] > 0.5:
            puntos += 1
            notas.append(f" Calmar={st['calmar_ratio']:.2f}: buen balance retorno/drawdown")
        elif st['calmar_ratio'] < 0:
            puntos -= 1
            notas.append("  Calmar negativo — retorno insuficiente para el drawdown sufrido")

        for n in notas:
            print(f"  {n}")

        if puntos >= 3:
            print(f"\n  ▶ SESGO LARGO PLAZO: ALCISTA FUERTE (+{puntos})")
        elif puntos >= 1:
            print(f"\n  ▶ SESGO LARGO PLAZO: LEVEMENTE ALCISTA (+{puntos}) — DCA recomendado")
        elif puntos == 0:
            print(f"\n  ▶ SESGO LARGO PLAZO: NEUTRAL — esperar confirmación")
        else:
            print(f"\n  ▶ SESGO LARGO PLAZO: CAUTELOSO ({puntos}) — reducir exposición")

    def graficar(self):
        p  = self.loader.precios
        r  = self.loader.retornos

        fig = plt.figure(figsize=(18, 11))
        fig.patch.set_facecolor('#0d1117')
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

        BG   = '#161b22'
        AZUL = '#58a6ff'
        ROJO = '#f78166'
        VERDE= '#3fb950'
        GRIS = '#8b949e'

        def estilo(ax, titulo):
            ax.set_facecolor(BG)
            ax.set_title(titulo, color='white', fontsize=10, pad=7)
            ax.tick_params(colors=GRIS, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#30363d')

        # ── 1. Precio + LPPLS fit ─────────────────────────────
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor(BG)
        ax1.plot(p.index, p.values, color=AZUL, linewidth=1.3, label='Precio')

        curva = self.burbuja.curva_ajustada()
        if curva is not None:
            color_fit = ROJO if self.burbuja.confianza_burbuja > 0.6 else VERDE
            ax1.plot(p.index, curva, '--', color=color_fit,
                     linewidth=1.5, alpha=0.8,
                     label=f'LPPLS (conf. burbuja: '
                           f'{self.burbuja.confianza_burbuja:.0%})')

        ax1.set_title(f'{self.ticker} — Precio & Modelo LPPLS',
                      color='white', fontsize=11, pad=7)
        ax1.legend(facecolor=BG, labelcolor='white', fontsize=9)
        ax1.tick_params(colors=GRIS)
        for sp in ax1.spines.values():
            sp.set_edgecolor('#30363d')

        # ── 2. DFA plot ────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        estilo(ax2, 'DFA — Detrended Fluctuation Analysis')

        if self.hurst.H_dfa is not None:
            x = np.log(self.hurst._escalas_dfa)
            y = np.log(self.hurst._F_dfa)
            ax2.scatter(x, y, color=AZUL, s=18, alpha=0.8)
            x_l = np.linspace(x.min(), x.max(), 100)
            y_l = self.hurst.H_dfa * (x_l - x.mean()) + np.mean(y)
            ax2.plot(x_l, y_l, '--', color=ROJO, linewidth=1.5,
                     label=f'H_DFA={self.hurst.H_dfa:.3f}')
            ax2.legend(facecolor=BG, labelcolor='white', fontsize=9)
        ax2.set_xlabel('log(n)', color=GRIS, fontsize=8)
        ax2.set_ylabel('log(F(n))', color=GRIS, fontsize=8)

        # ── 3. Espectro de potencia (ciclos) ──────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        estilo(ax3, 'Espectro de Potencia (Ciclos Dominantes)')

        freqs   = self.ciclos.freqs[1:]   # quitar freq 0
        pot     = self.ciclos.potencia[1:]
        periodos = 1.0 / (freqs + 1e-10)
        mask    = periodos < len(p) * 0.8
        ax3.semilogy(periodos[mask], pot[mask], color=AZUL,
                     linewidth=1.0, alpha=0.8)
        # Marcar picos
        for ciclo in self.ciclos.ciclos[:3]:
            ax3.axvline(ciclo['periodo_dias'], color=ROJO,
                        alpha=0.6, linewidth=1.2, linestyle='--')
        ax3.set_xlabel('Período (días)', color=GRIS, fontsize=8)
        ax3.set_ylabel('Potencia', color=GRIS, fontsize=8)

        # ── 4. Retornos mensuales ─────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        estilo(ax4, 'Retornos Mensuales')

        ret_m = self.loader.retornos_mensuales()
        colores_barra = [VERDE if v > 0 else ROJO for v in ret_m.values]
        ax4.bar(range(len(ret_m)), ret_m.values * 100,
                color=colores_barra, alpha=0.8, width=0.8)
        ax4.axhline(0, color=GRIS, linewidth=0.8)
        ax4.set_xlabel('Mes', color=GRIS, fontsize=8)
        ax4.set_ylabel('Retorno (%)', color=GRIS, fontsize=8)

        # ── 5. Drawdown histórico ─────────────────────────────
        ax5 = fig.add_subplot(gs[1, 2])
        estilo(ax5, 'Drawdown Histórico')

        peak = p.cummax()
        dd   = (p - peak) / peak * 100
        ax5.fill_between(dd.index, dd.values, 0,
                         color=ROJO, alpha=0.4)
        ax5.plot(dd.index, dd.values, color=ROJO, linewidth=0.8)
        ax5.set_ylabel('Drawdown (%)', color=GRIS, fontsize=8)

        fig.suptitle(
            f'ANÁLISIS ECONOFÍSICO LARGO PLAZO — {self.ticker}  │  {self.periodo}',
            color='white', fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════
# FUNCIONES DE ALTO NIVEL
# ══════════════════════════════════════════════════════════════
def analizar_activo(ticker: str, periodo: str = '5y'):
    """Análisis completo de un activo para mediano/largo plazo."""
    d = Dashboard(ticker, periodo)
    d.analizar()
    d.imprimir_reporte()
    d.graficar()
    return d


def analizar_portafolio(tickers: list, periodo: str = '5y',
                        max_drawdown: float = 0.25):
    """
    Análisis de portafolio con RMT + optimización robusta + cointegración.
    """
    print(f"\n{'═'*62}")
    print(f"  PORTAFOLIO LARGO PLAZO: {', '.join(tickers)}")
    print(f"{'═'*62}\n")

    # Descargar precios y retornos
    precios_dict  = {}
    retornos_dict = {}
    for t in tickers:
        try:
            loader = DataLoader(t, periodo)
            loader.descargar()
            precios_dict[t]  = loader.precios
            retornos_dict[t] = loader.retornos
        except Exception as e:
            print(f"    {t}: {e}")

    if len(retornos_dict) < 2:
        print("  Se necesitan al menos 2 activos.")
        return

    retornos_df = pd.DataFrame(retornos_dict).dropna()
    precios_df  = pd.DataFrame(precios_dict).dropna()

    # 1. RMT
    print(" Random Matrix Theory — filtrando correlaciones...")
    rmt = RMT_Correlaciones(retornos_df)
    res_rmt = rmt.analizar()
    print(f"   {res_rmt['interpretacion']}")

    # 2. Optimización robusta
    print("\n  Optimización de portafolio (Entropía + RMT)...")
    opt = OptimizadorRobust(retornos_df, corr_limpia=rmt.corr_limpia)
    res_opt = opt.optimizar(lambda_riesgo=2.5, max_drawdown=max_drawdown)

    print("\n  PESOS ÓPTIMOS:")
    print("  " + "─"*40)
    for ticker, peso in res_opt['pesos'].items():
        barra = '█' * int(peso * 35)
        print(f"  {ticker:<14} {peso*100:5.1f}%  {barra}")
    print(f"\n  Retorno anual est.:  {res_opt['retorno_anual_est']*100:+.1f}%")
    print(f"  Volatilidad anual:   {res_opt['volatilidad_anual']*100:.1f}%")
    print(f"  Sharpe:              {res_opt['sharpe_ratio']:.2f}")
    print(f"  Drawdown estimado:   {res_opt['drawdown_estimado']*100:.1f}%")

    # 3. Cointegración
    print(f"\n🔗 Cointegración entre pares...")
    coint_anal = AnalisisCoIntegracion(precios_df)
    pares = coint_anal.analizar_todos_los_pares()

    pares_coint = [p for p in pares if p['cointegrados']]
    if pares_coint:
        print(f"  {len(pares_coint)} par(es) cointegrado(s):\n")
        for par in pares_coint[:5]:
            hl = par['half_life_dias']
            hl_str = f"{hl} días" if hl else "N/A"
            print(f"  {par['par']:<18} p={par['p_valor']:.3f}  "
                  f"half-life={hl_str}")
            print(f"   {par['señal_trading']}")
    else:
        print("  Sin pares cointegrados significativos.")

    print(f"\n{'═'*62}\n")
    return {'rmt': res_rmt, 'portafolio': res_opt, 'pares': pares}


# ══════════════════════════════════════════════════════════════
# EJECUCIÓN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Análisis individual ───────────────────────────────────
    analizar_activo("AAPL",    periodo="5y")
    analizar_activo("BTC-USD", periodo="5y")
    analizar_activo("SPY",     periodo="10y")

    # ── Portafolio largo plazo ────────────────────────────────
    analizar_portafolio(
        tickers=["SPY", "AAPL", "NVDA", "BTC-USD", "GLD"],
        periodo="5y",
        max_drawdown=0.25,
    )
