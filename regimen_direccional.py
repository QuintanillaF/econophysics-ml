"""
╔══════════════════════════════════════════════════════════════════╗
║   RÉGIMEN DIRECCIONAL + CAMBIO DE ESTRATEGIA                    ║
║   BULL / BEAR / SIDEWAYS → estrategia distinta en cada uno      ║
╚══════════════════════════════════════════════════════════════════╝

QUÉ AGREGA AL SISTEMA:

  ANTES:
    Régimen = CALMA / ESTRÉS  (solo volatilidad, sin dirección)
    Los agentes usaban los MISMOS pesos siempre
    El régimen solo ajustaba el tamaño de posición

  AHORA:
    Régimen = BULL / BEAR / SIDEWAYS  (dirección + fuerza)
    Los agentes cambian sus PESOS según el régimen
    Cada régimen tiene su propia estrategia:

      BULL      → momentum / trend following
                  comprar fuerza, aguantar posiciones, Kelly completo
      BEAR      → defensiva
                  solo sobreventa extrema, Kelly reducido, umbral alto
      SIDEWAYS  → mean reversion
                  comprar bandas bajas, vender bandas altas, salir rápido

  Además detecta el RÉGIMEN DEL MERCADO GENERAL (SPY para stocks,
  BTC para crypto) y lo usa como filtro: si el mercado está en BEAR,
  se aplica cautela extra aunque un activo suelto se vea alcista.

INDICADORES USADOS PARA CLASIFICAR:
  1. Pendiente de regresión log-lineal → dirección y velocidad
  2. R² de esa regresión               → ¿es tendencia limpia o ruido?
  3. SMA 50 vs SMA 200                 → estructura de tendencia
  4. ADX + DI+/DI-                     → fuerza de la tendencia
  5. Drawdown desde el máximo          → ¿estamos en corrección?
  6. Hurst                             → persistente vs reversivo

INSTALACIÓN:
  pip install numpy scipy pandas requests yfinance

USO:
  from regimen_direccional import SistemaAgentesAdaptativo

  sistema = SistemaAgentesAdaptativo(
      tickers=['BTCUSDT','ETHUSDT','AAPL','NVDA'],
      capital_inicial=100_000,
  )
  sistema.analizar_todos()
"""

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional
import warnings; warnings.filterwarnings('ignore')

from scipy import stats

from data_layer import get_data_layer, normalizar_ticker
from sistema_agentes import (
    ContextoMercado, ArgumentoAgente, DecisionFinal, KillSwitch
)

logger = logging.getLogger('RegimenDireccional')
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s │ %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(h)


# ══════════════════════════════════════════════════════════════
# MÓDULO 1: RESULTADO DE LA CLASIFICACIÓN
# ══════════════════════════════════════════════════════════════
@dataclass
class RegimenMercado:
    """
    Resultado completo de la clasificación de régimen.

    regimen:     'BULL' | 'BEAR' | 'SIDEWAYS'
    score:       -100 (bear extremo) a +100 (bull extremo)
    fuerza:      0 a 100 — qué tan definida está la tendencia
                 (< 25 significa lateral aunque el score sea alto)
    confianza:   0 a 1 — qué tan seguros estamos de la clasificación
    """
    ticker:          str
    regimen:         str
    score:           float
    fuerza:          float
    confianza:       float
    # Indicadores individuales (para auditar la decisión)
    tendencia_anual: float   # % anualizado de la regresión
    r2:              float   # calidad de la tendencia (0-1)
    sma_estructura:  str     # 'ALCISTA' | 'BAJISTA' | 'MIXTA'
    adx:             float   # fuerza de tendencia (>25 = fuerte)
    di_diff:         float   # DI+ menos DI- (dirección del ADX)
    drawdown:        float   # % desde el máximo reciente
    hurst:           float
    vol_anual:       float
    detalle:         list = field(default_factory=list)
    timestamp:       str = field(default_factory=lambda: datetime.now().isoformat())

    def resumen(self) -> str:
        emoji = {'BULL': '🐂', 'BEAR': '🐻', 'SIDEWAYS': '↔️'}[self.regimen]
        return (f"{emoji} {self.regimen} │ score={self.score:+.0f} │ "
                f"fuerza={self.fuerza:.0f} │ conf={self.confianza:.0%}")

    def to_texto(self) -> str:
        lines = [
            f"RÉGIMEN — {self.ticker}",
            "─" * 46,
            f"  Clasificación:    {self.resumen()}",
            f"  Tendencia anual:  {self.tendencia_anual:+.1f}%  (R²={self.r2:.2f})",
            f"  Estructura SMA:   {self.sma_estructura}",
            f"  ADX:              {self.adx:.1f}  (DI+ − DI− = {self.di_diff:+.1f})",
            f"  Drawdown:         {self.drawdown:.1f}%",
            f"  Hurst:            {self.hurst:.3f}",
            f"  Volatilidad:      {self.vol_anual:.1f}%",
        ]
        for d in self.detalle:
            lines.append(f"    {d}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# MÓDULO 2: DETECTOR DE RÉGIMEN DIRECCIONAL
# ══════════════════════════════════════════════════════════════
class DetectorRegimenDireccional:
    """
    Clasifica el mercado en BULL / BEAR / SIDEWAYS.

    LÓGICA EN DOS PASOS:

      Paso 1 — ¿HAY tendencia?
        Se mide con ADX y con el R² de la regresión.
        ADX < 20 y R² < 0.3  →  no hay tendencia  →  SIDEWAYS
        (sin importar hacia dónde apunte el score)

      Paso 2 — Si hay tendencia, ¿hacia dónde?
        Se suma un score de -100 a +100 combinando:
        pendiente, estructura SMA, DI+/DI−, drawdown y Hurst.
        score > +20  →  BULL
        score < -20  →  BEAR
        en medio     →  SIDEWAYS (transición)

    El régimen se cachea por 6 horas: no cambia minuto a minuto y
    recalcularlo en cada ciclo del bot sería un desperdicio de API.
    """

    CACHE_HORAS = 6

    def __init__(self, periodo: str = '2y'):
        self.dl      = get_data_layer()
        self.periodo = periodo
        self._cache: dict = {}

    # ── Cache ─────────────────────────────────────────────────
    def _cache_ok(self, ticker: str) -> bool:
        if ticker not in self._cache:
            return False
        ts, _ = self._cache[ticker]
        return (datetime.now() - ts) < timedelta(hours=self.CACHE_HORAS)

    # ── Indicadores ───────────────────────────────────────────
    def _regresion(self, precios: pd.Series, dias: int = 126) -> tuple:
        """
        Regresión log-lineal sobre los últimos N días.

        La pendiente da la dirección Y la velocidad de la tendencia.
        El R² dice si es una tendencia limpia o un movimiento errático:
          R² > 0.7  → tendencia muy definida
          R² < 0.3  → movimiento sin dirección (lateral)
        """
        p = precios.tail(dias).dropna()
        if len(p) < 30:
            return 0.0, 0.0

        y = np.log(p.values)
        x = np.arange(len(y))
        slope, _, r_value, _, _ = stats.linregress(x, y)

        # Anualizar la pendiente diaria
        tendencia_anual = (np.exp(slope * 252) - 1) * 100
        return float(tendencia_anual), float(r_value ** 2)

    def _estructura_sma(self, precios: pd.Series) -> tuple:
        """
        Relación entre precio, SMA50 y SMA200.

        precio > SMA50 > SMA200  →  estructura alcista clásica
        precio < SMA50 < SMA200  →  estructura bajista clásica
        cualquier otra            →  mixta (transición)
        """
        if len(precios) < 60:
            return 'MIXTA', 0.0

        sma50  = precios.rolling(50).mean()
        n200   = min(200, len(precios) - 1)
        sma200 = precios.rolling(n200).mean()

        p   = float(precios.iloc[-1])
        s50 = float(sma50.iloc[-1]) if not np.isnan(sma50.iloc[-1]) else p
        s200= float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else p

        if p > s50 > s200:
            return 'ALCISTA', 1.0
        if p < s50 < s200:
            return 'BAJISTA', -1.0
        # Estructuras parciales
        if p > s50 and p > s200:
            return 'ALCISTA', 0.5
        if p < s50 and p < s200:
            return 'BAJISTA', -0.5
        return 'MIXTA', 0.0

    def _adx(self, df: pd.DataFrame, n: int = 14) -> tuple:
        """
        ADX (Average Directional Index) — mide FUERZA de tendencia,
        no dirección. DI+ y DI− dan la dirección.

          ADX > 25  → tendencia fuerte (trend following funciona)
          ADX < 20  → sin tendencia (mean reversion funciona)

        Si no hay columnas high/low (fallback de CoinGecko), se
        aproxima usando solo el cierre.
        """
        if 'high' not in df.columns or 'low' not in df.columns:
            c = df['close']
            h = c.rolling(2).max().fillna(c)
            l = c.rolling(2).min().fillna(c)
        else:
            h, l, c = df['high'], df['low'], df['close']

        if len(c) < n * 3:
            return 0.0, 0.0

        up   = h.diff()
        down = -l.diff()
        plus_dm  = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)

        tr = pd.concat([
            h - l,
            (h - c.shift()).abs(),
            (l - c.shift()).abs()
        ], axis=1).max(axis=1)

        atr      = tr.ewm(alpha=1/n, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-10)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx      = dx.ewm(alpha=1/n, adjust=False).mean()

        try:
            return float(adx.iloc[-1]), float(plus_di.iloc[-1] - minus_di.iloc[-1])
        except Exception:
            return 0.0, 0.0

    def _drawdown(self, precios: pd.Series, dias: int = 252) -> float:
        """% de caída desde el máximo del último año."""
        p = precios.tail(dias)
        if len(p) < 2:
            return 0.0
        return float((p.iloc[-1] / p.max() - 1) * 100)

    # ── Clasificación principal ───────────────────────────────
    def detectar(self, ticker: str, forzar: bool = False) -> Optional[RegimenMercado]:
        """
        Clasifica el régimen del ticker.

        Args:
            ticker: cualquier formato (BTCUSDT, BTC-USD, AAPL)
            forzar: ignora el cache y recalcula
        """
        if not forzar and self._cache_ok(ticker):
            return self._cache[ticker][1]

        df = self.dl.get_ohlcv(ticker, horizonte='mediano', periodo=self.periodo)
        if df.empty or len(df) < 60:
            df = self.dl.get_ohlcv(ticker, horizonte='largo', periodo='2y')
        if df.empty or len(df) < 60:
            logger.warning(f"Datos insuficientes para clasificar {ticker}")
            return None

        precios = df['close'].squeeze()
        retornos = np.log(precios / precios.shift(1)).dropna()

        # ── Indicadores ───────────────────────────────────────
        tend_anual, r2   = self._regresion(precios, dias=126)
        estructura, s_sc = self._estructura_sma(precios)
        adx, di_diff     = self._adx(df)
        dd               = self._drawdown(precios)
        H                = self.dl._hurst(retornos.values)
        vol              = float(retornos.std() * np.sqrt(252) * 100)

        detalle = []

        # ── PASO 1: ¿Hay tendencia? ───────────────────────────
        # ADX y R² se combinan en una medida de "fuerza"
        fuerza_adx = np.clip(adx / 40 * 100, 0, 100)      # ADX 40 = 100
        fuerza_r2  = np.clip(r2 / 0.7 * 100, 0, 100)      # R² 0.7 = 100
        fuerza     = float(0.5 * fuerza_adx + 0.5 * fuerza_r2)

        sin_tendencia = (adx < 20 and r2 < 0.30)
        if sin_tendencia:
            detalle.append(f"Sin tendencia definida (ADX={adx:.0f} < 20, R²={r2:.2f} < 0.30)")

        # ── PASO 2: Dirección — score de -100 a +100 ──────────
        score = 0.0

        # Pendiente anualizada (peso 35)
        # ±40% anual satura el componente
        comp_tend = np.clip(tend_anual / 40, -1, 1) * 35
        score += comp_tend
        detalle.append(f"Tendencia {tend_anual:+.0f}%/año → {comp_tend:+.0f} pts")

        # Estructura SMA (peso 25)
        comp_sma = s_sc * 25
        score += comp_sma
        detalle.append(f"Estructura SMA {estructura} → {comp_sma:+.0f} pts")

        # Dirección del ADX (peso 20)
        comp_di = np.clip(di_diff / 25, -1, 1) * 20
        score += comp_di
        detalle.append(f"DI+ − DI− = {di_diff:+.1f} → {comp_di:+.0f} pts")

        # Drawdown (peso 15) — cerca de máximos es alcista
        # 0% dd → +15 ; -30% dd → -15
        comp_dd = np.clip((dd + 15) / 15, -1, 1) * 15
        score += comp_dd
        detalle.append(f"Drawdown {dd:.0f}% → {comp_dd:+.0f} pts")

        # Hurst (peso 5) — solo refuerza, no define dirección
        comp_h = (H - 0.5) * 2 * 5 * np.sign(comp_tend if comp_tend != 0 else 1)
        score += comp_h
        detalle.append(f"Hurst {H:.2f} → {comp_h:+.0f} pts")

        score = float(np.clip(score, -100, 100))

        # ── Clasificación final ───────────────────────────────
        if sin_tendencia or fuerza < 25:
            regimen = 'SIDEWAYS'
        elif score > 20:
            regimen = 'BULL'
        elif score < -20:
            regimen = 'BEAR'
        else:
            regimen = 'SIDEWAYS'
            detalle.append(f"Score {score:+.0f} en zona neutral (-20 a +20)")

        # Confianza: qué tan lejos del límite estamos
        if regimen == 'SIDEWAYS':
            confianza = float(np.clip(1 - abs(score) / 60, 0.3, 1.0))
        else:
            confianza = float(np.clip(
                0.4 + (abs(score) - 20) / 80 * 0.4 + fuerza / 100 * 0.2, 0.3, 1.0))

        res = RegimenMercado(
            ticker=ticker, regimen=regimen, score=round(score, 1),
            fuerza=round(fuerza, 1), confianza=round(confianza, 3),
            tendencia_anual=round(tend_anual, 2), r2=round(r2, 3),
            sma_estructura=estructura, adx=round(adx, 1),
            di_diff=round(di_diff, 1), drawdown=round(dd, 2),
            hurst=round(H, 3), vol_anual=round(vol, 1), detalle=detalle,
        )
        self._cache[ticker] = (datetime.now(), res)
        return res

    def regimen_mercado_general(self, es_crypto: bool) -> Optional[RegimenMercado]:
        """
        Régimen del mercado en general, no de un activo suelto.

        Stocks → SPY (índice S&P 500)
        Crypto → BTCUSDT (BTC arrastra al resto del mercado)

        Se usa como FILTRO: si el mercado general está en BEAR,
        el sistema aplica cautela extra a todos los activos.
        """
        proxy = 'BTCUSDT' if es_crypto else 'SPY'
        return self.detectar(proxy)


# ══════════════════════════════════════════════════════════════
# MÓDULO 3: PARÁMETROS DE ESTRATEGIA POR RÉGIMEN
# ══════════════════════════════════════════════════════════════
@dataclass
class ParametrosEstrategia:
    """Configuración que cambia según el régimen detectado."""
    nombre:          str
    descripcion:     str
    kelly_mult:      float   # multiplicador sobre el Kelly base
    min_confianza:   float   # umbral para operar
    max_posiciones:  int     # cuántas posiciones simultáneas
    permitir_compra: bool
    permitir_venta:  bool
    # Pesos de los agentes (multiplicadores sobre el peso base)
    peso_momentum:   float   # momentum y Hurst
    peso_reversion:  float   # RSI y Bollinger
    peso_riesgo:     float   # cuánto pesa el Bear
    # Umbrales de RSI adaptados al régimen
    rsi_compra:      float
    rsi_venta:       float


class EstrategiaPorRegimen:
    """
    Devuelve los parámetros de estrategia para cada régimen.

    LA IDEA CENTRAL:
      El mismo indicador significa cosas distintas en cada régimen.

      RSI=65 en BULL      → tendencia sana, seguir comprando
      RSI=65 en SIDEWAYS  → cerca del techo del rango, vender
      RSI=65 en BEAR      → rebote de gato muerto, no entrar

      Momentum +8% en BULL     → confirmación, comprar
      Momentum +8% en SIDEWAYS → sobreextendido, esperar reversión
    """

    ESTRATEGIAS = {
        'BULL': ParametrosEstrategia(
            nombre='MOMENTUM / TREND FOLLOWING',
            descripcion='Comprar fuerza, aguantar la tendencia, tamaño completo',
            kelly_mult=1.0, min_confianza=0.55, max_posiciones=6,
            permitir_compra=True, permitir_venta=True,
            peso_momentum=1.5, peso_reversion=0.6, peso_riesgo=0.8,
            rsi_compra=55, rsi_venta=82,   # en bull se compra fuerza
        ),
        'BEAR': ParametrosEstrategia(
            nombre='DEFENSIVA',
            descripcion='Solo sobreventa extrema, tamaño reducido, umbral alto',
            kelly_mult=0.4, min_confianza=0.72, max_posiciones=2,
            permitir_compra=True, permitir_venta=True,
            peso_momentum=0.4, peso_reversion=1.2, peso_riesgo=1.6,
            rsi_compra=25, rsi_venta=55,   # solo comprar caídas fuertes
        ),
        'SIDEWAYS': ParametrosEstrategia(
            nombre='MEAN REVERSION',
            descripcion='Comprar banda baja, vender banda alta, salir rápido',
            kelly_mult=0.6, min_confianza=0.62, max_posiciones=4,
            permitir_compra=True, permitir_venta=True,
            peso_momentum=0.3, peso_reversion=1.8, peso_riesgo=1.0,
            rsi_compra=35, rsi_venta=65,   # rango clásico
        ),
    }

    @classmethod
    def get(cls, regimen: str) -> ParametrosEstrategia:
        return cls.ESTRATEGIAS.get(regimen, cls.ESTRATEGIAS['SIDEWAYS'])

    @classmethod
    def ajustar_por_mercado(cls, params: ParametrosEstrategia,
                            regimen_mercado: str) -> ParametrosEstrategia:
        """
        Aplica el filtro del mercado general.

        Si el mercado está en BEAR pero el activo se ve BULL,
        no ignoramos la señal — la achicamos y subimos el umbral.
        Un activo rara vez sobrevive contra su mercado.
        """
        if regimen_mercado == 'BEAR':
            return ParametrosEstrategia(
                nombre=params.nombre + ' (mercado BEAR)',
                descripcion=params.descripcion + ' — cautela por mercado bajista',
                kelly_mult=params.kelly_mult * 0.5,
                min_confianza=min(params.min_confianza + 0.08, 0.85),
                max_posiciones=max(1, params.max_posiciones // 2),
                permitir_compra=params.permitir_compra,
                permitir_venta=params.permitir_venta,
                peso_momentum=params.peso_momentum * 0.7,
                peso_reversion=params.peso_reversion,
                peso_riesgo=params.peso_riesgo * 1.3,
                rsi_compra=params.rsi_compra - 5,
                rsi_venta=params.rsi_venta - 5,
            )
        if regimen_mercado == 'BULL':
            return ParametrosEstrategia(
                nombre=params.nombre + ' (mercado BULL)',
                descripcion=params.descripcion + ' — viento a favor',
                kelly_mult=min(params.kelly_mult * 1.15, 1.2),
                min_confianza=max(params.min_confianza - 0.03, 0.50),
                max_posiciones=params.max_posiciones,
                permitir_compra=params.permitir_compra,
                permitir_venta=params.permitir_venta,
                peso_momentum=params.peso_momentum,
                peso_reversion=params.peso_reversion,
                peso_riesgo=params.peso_riesgo * 0.9,
                rsi_compra=params.rsi_compra,
                rsi_venta=params.rsi_venta,
            )
        return params


# ══════════════════════════════════════════════════════════════
# MÓDULO 4: AGENTES ADAPTATIVOS
# Mismos agentes, pero con pesos que cambian según el régimen
# ══════════════════════════════════════════════════════════════
class AgenteAlcistaAdaptativo:
    """
    Bull con pesos variables.

    En BULL      → premia momentum, Hurst alto, precio sobre medias
    En SIDEWAYS  → premia RSI bajo y banda inferior; PENALIZA momentum alto
    En BEAR      → solo se activa con sobreventa extrema
    """
    ROL = 'BULL'

    def analizar(self, ctx: ContextoMercado,
                 params: ParametrosEstrategia,
                 reg: RegimenMercado) -> ArgumentoAgente:
        pts, max_pts, args = 0.0, 0.0, []

        # ── Componente momentum (peso variable) ───────────────
        w_mom = params.peso_momentum

        max_pts += 2 * w_mom
        if ctx.hurst > 0.60:
            pts += 2 * w_mom
            args.append(f"✅ Hurst={ctx.hurst:.3f}: tendencia persistente")
        elif ctx.hurst > 0.53:
            pts += 1 * w_mom
            args.append(f"✅ Hurst={ctx.hurst:.3f}: leve persistencia")
        else:
            args.append(f"➖ Hurst={ctx.hurst:.3f}: sin persistencia")

        max_pts += 1.5 * w_mom
        if reg.regimen == 'SIDEWAYS':
            # En lateral, momentum alto = sobreextendido = MALO para comprar
            if ctx.retorno_7d > 5:
                args.append(f"⚠️  Momentum 7d {ctx.retorno_7d:+.1f}% en lateral: sobreextendido")
            elif ctx.retorno_7d < -5:
                pts += 1.5 * w_mom
                args.append(f"✅ Caída 7d {ctx.retorno_7d:+.1f}% en lateral: zona de compra")
        else:
            if ctx.retorno_7d > 3:
                pts += 1.5 * w_mom
                args.append(f"✅ Momentum 7d: {ctx.retorno_7d:+.1f}%")
            elif ctx.retorno_7d < -5:
                args.append(f"⚠️  Caída 7d: {ctx.retorno_7d:+.1f}%")

        # ── Componente reversión (peso variable) ──────────────
        w_rev = params.peso_reversion

        max_pts += 2 * w_rev
        if ctx.rsi < params.rsi_compra - 10:
            pts += 2 * w_rev
            args.append(f"✅ RSI={ctx.rsi:.0f} muy por debajo del umbral {params.rsi_compra:.0f}")
        elif ctx.rsi < params.rsi_compra:
            pts += 1.2 * w_rev
            args.append(f"✅ RSI={ctx.rsi:.0f} < umbral de compra {params.rsi_compra:.0f}")
        elif ctx.rsi > params.rsi_venta:
            args.append(f"⚠️  RSI={ctx.rsi:.0f} > umbral de venta {params.rsi_venta:.0f}")
        else:
            args.append(f"➖ RSI={ctx.rsi:.0f} en zona media")

        max_pts += 1.5 * w_rev
        if ctx.bb_posicion < -0.6:
            pts += 1.5 * w_rev
            args.append("✅ Precio en banda inferior de Bollinger")
        elif ctx.bb_posicion < 0:
            pts += 0.5 * w_rev
            args.append("➖ Precio bajo la media de Bollinger")
        elif ctx.bb_posicion > 0.7 and reg.regimen == 'SIDEWAYS':
            args.append("⚠️  Precio en banda superior — techo del rango")

        # ── Componentes que no dependen del régimen ───────────
        max_pts += 2
        if ctx.señal_ml == 1:
            pts += 2 * ctx.prob_ml
            args.append(f"✅ ML: COMPRAR (prob={ctx.prob_ml:.0%})")
        elif ctx.señal_ml == -1:
            args.append("❌ ML: VENDER")

        max_pts += 1.5
        if ctx.whales_señal in ('ALCISTA', 'COMPRADORA'):
            pts += 1.5
            args.append(f"✅ Flujo comprador: {ctx.whales_detalle}")
        elif ctx.whales_señal in ('BAJISTA', 'VENDEDORA'):
            args.append("❌ Flujo vendedor")

        max_pts += 1
        if ctx.regimen == 'CALMA':
            pts += 1
            args.append("✅ Volatilidad bajo control")
        else:
            args.append("⚠️  Volatilidad elevada")

        # ── Bonus/penalización por régimen direccional ────────
        max_pts += 2
        if reg.regimen == 'BULL':
            bonus = 2 * (reg.confianza)
            pts += bonus
            args.append(f"✅ Régimen BULL (score {reg.score:+.0f}) → +{bonus:.1f} pts")
        elif reg.regimen == 'BEAR':
            args.append(f"❌ Régimen BEAR (score {reg.score:+.0f}) → sin bonus alcista")
        else:
            pts += 0.6
            args.append(f"➖ Régimen SIDEWAYS → operar solo extremos del rango")

        c = pts / max_pts if max_pts > 0 else 0
        return ArgumentoAgente(rol=self.ROL, posicion=1 if c > 0.5 else 0,
                               confianza=round(float(c), 3), argumentos=args)


class AgenteBajistaAdaptativo:
    """
    Bear con peso variable.

    En BEAR      → peso alto, encuentra razones para no comprar
    En BULL      → peso reducido, no frena la tendencia sin motivo
    En SIDEWAYS  → peso medio, alerta sobre techos del rango
    """
    ROL = 'BEAR'

    def analizar(self, ctx: ContextoMercado,
                 params: ParametrosEstrategia,
                 reg: RegimenMercado) -> ArgumentoAgente:
        pts, max_pts, args = 0.0, 0.0, []
        w = params.peso_riesgo

        max_pts += 2 * w
        if ctx.regimen == 'ESTRÉS':
            pts += 2 * w
            args.append("🔴 Volatilidad en régimen de ESTRÉS")
        else:
            args.append("✅ Volatilidad normal")

        max_pts += 2 * w
        if ctx.rsi > params.rsi_venta:
            pts += 2 * w
            args.append(f"🔴 RSI={ctx.rsi:.0f} > umbral de venta {params.rsi_venta:.0f}")
        elif ctx.rsi > params.rsi_venta - 10:
            pts += 0.8 * w
            args.append(f"⚠️  RSI={ctx.rsi:.0f} acercándose al techo")
        else:
            args.append(f"➖ RSI={ctx.rsi:.0f} sin señal bajista")

        max_pts += 1.5 * w
        if ctx.volatilidad > 80:
            pts += 1.5 * w
            args.append(f"🔴 Volatilidad {ctx.volatilidad:.0f}%: extrema")
        elif ctx.volatilidad > 50:
            pts += 0.7 * w
            args.append(f"⚠️  Volatilidad {ctx.volatilidad:.0f}%: elevada")

        max_pts += 2
        if ctx.señal_ml == -1:
            pts += 2 * ctx.prob_ml
            args.append(f"🔴 ML: VENDER (prob={ctx.prob_ml:.0%})")
        elif ctx.señal_ml == 1:
            args.append("✅ ML dice comprar")

        max_pts += 1.5
        if ctx.whales_señal in ('BAJISTA', 'VENDEDORA'):
            pts += 1.5
            args.append(f"🔴 Flujo vendedor: {ctx.whales_detalle}")
        elif ctx.whales_señal in ('ALCISTA', 'COMPRADORA'):
            args.append("✅ Flujo comprador")

        max_pts += 1
        if ctx.bb_posicion > 0.6:
            pts += 1
            args.append("🔴 Precio en banda superior de Bollinger")

        # ── Régimen direccional ───────────────────────────────
        max_pts += 3
        if reg.regimen == 'BEAR':
            bonus = 3 * reg.confianza
            pts += bonus
            args.append(f"🔴 Régimen BEAR (score {reg.score:+.0f}, "
                        f"tendencia {reg.tendencia_anual:+.0f}%/año) → +{bonus:.1f} pts")
        elif reg.regimen == 'SIDEWAYS':
            pts += 1.0
            args.append(f"⚠️  Régimen SIDEWAYS → tendencias no duran")
        else:
            args.append(f"✅ Régimen BULL → sin señal estructural bajista")

        # Drawdown profundo
        max_pts += 1
        if reg.drawdown < -25:
            pts += 1
            args.append(f"🔴 Drawdown {reg.drawdown:.0f}% desde máximos")

        c = pts / max_pts if max_pts > 0 else 0
        return ArgumentoAgente(rol=self.ROL, posicion=-1 if c > 0.5 else 0,
                               confianza=round(float(c), 3), argumentos=args)


class RiskManagerAdaptativo:
    """
    Risk Manager con vetos que se endurecen según el régimen.

    En BEAR los umbrales de veto bajan: es más fácil que vete.
    En BULL se relajan un poco, pero nunca desaparecen.
    """
    ROL = 'RISK'

    def __init__(self, max_vol_base: float = 100.0, min_consenso: float = 0.15):
        self.max_vol_base = max_vol_base
        self.min_consenso = min_consenso

    def evaluar(self, ctx: ContextoMercado, bull: ArgumentoAgente,
                bear: ArgumentoAgente, params: ParametrosEstrategia,
                reg: RegimenMercado, reg_mercado: Optional[RegimenMercado],
                capital: float = None, capital_0: float = None,
                n_posiciones: int = 0) -> ArgumentoAgente:
        vetos, args = [], []

        # Umbral de volatilidad ajustado al régimen
        max_vol = self.max_vol_base
        if reg.regimen == 'BEAR':
            max_vol *= 0.7
        elif reg.regimen == 'SIDEWAYS':
            max_vol *= 0.85

        if ctx.volatilidad > max_vol:
            vetos.append(f"Volatilidad {ctx.volatilidad:.0f}% > límite "
                         f"{max_vol:.0f}% (ajustado por régimen {reg.regimen})")

        if ctx.regimen == 'ESTRÉS' and ctx.volatilidad > 60:
            vetos.append("Régimen de ESTRÉS con volatilidad > 60%")

        # Drawdown del portfolio
        if capital and capital_0:
            dd = (capital_0 - capital) / capital_0
            if dd > 0.15:
                vetos.append(f"Portfolio en drawdown {dd:.0%}")
            if dd > 0.20:
                vetos.append(f"DRAWDOWN CRÍTICO {dd:.0%} — veto total")

        # Consenso entre agentes
        dif = abs(bull.confianza - bear.confianza)
        if dif < self.min_consenso:
            vetos.append(f"Señales equilibradas (Bull={bull.confianza:.2f} "
                         f"Bear={bear.confianza:.2f})")

        # Límite de posiciones simultáneas del régimen
        if n_posiciones >= params.max_posiciones and bull.posicion == 1:
            vetos.append(f"Ya hay {n_posiciones} posiciones abiertas "
                         f"(máximo {params.max_posiciones} en {reg.regimen})")

        # Comprar contra un BEAR estructural fuerte
        if (bull.posicion == 1 and reg.regimen == 'BEAR'
                and reg.confianza > 0.7 and ctx.rsi > 40):
            vetos.append(f"Compra contra BEAR fuerte (score {reg.score:+.0f}) "
                         f"sin sobreventa (RSI={ctx.rsi:.0f} > 40)")

        # Mercado general en BEAR + activo volátil
        if (reg_mercado and reg_mercado.regimen == 'BEAR'
                and reg_mercado.confianza > 0.65
                and bull.posicion == 1 and ctx.volatilidad > 60):
            vetos.append(f"Mercado general en BEAR y activo con "
                         f"volatilidad {ctx.volatilidad:.0f}%")

        # RSI en extremos absolutos
        if bull.posicion == 1 and ctx.rsi > 82:
            vetos.append(f"RSI={ctx.rsi:.0f}: sobrecompra extrema")
        if bear.posicion == -1 and ctx.rsi < 18:
            vetos.append(f"RSI={ctx.rsi:.0f}: sobreventa extrema")

        hay_veto = len(vetos) > 0
        if hay_veto:
            args.append("🛑 VETO ACTIVADO")
            for v in vetos:
                args.append(f"  → {v}")
        else:
            args.append(f"✅ Sin objeciones │ estrategia: {params.nombre}")
            args.append(f"  Diferencia de confianza: {dif:.2f} "
                        f"(mínimo {self.min_consenso})")
            args.append(f"  Posiciones: {n_posiciones}/{params.max_posiciones}")

        return ArgumentoAgente(rol=self.ROL, posicion=0, confianza=0.0,
                               argumentos=args, veto=hay_veto)


# ══════════════════════════════════════════════════════════════
# MÓDULO 5: MOTOR DE DEBATE ADAPTATIVO
# ══════════════════════════════════════════════════════════════
class MotorDebateAdaptativo:
    """
    Orquesta el debate con los agentes adaptativos.

    FLUJO:
      1. Detectar régimen del activo
      2. Detectar régimen del mercado general
      3. Elegir estrategia según ambos
      4. Bull y Bear analizan con los pesos de esa estrategia
      5. Risk Manager evalúa con umbrales de esa estrategia
      6. Kelly se multiplica por el kelly_mult del régimen
    """

    def __init__(self, kelly_base: float = 0.25, min_edge: float = 0.15):
        self.bull      = AgenteAlcistaAdaptativo()
        self.bear      = AgenteBajistaAdaptativo()
        self.risk      = RiskManagerAdaptativo()
        self.kelly_base = kelly_base
        self.min_edge  = min_edge

    def debatir(self, ctx: ContextoMercado, reg: RegimenMercado,
                reg_mercado: Optional[RegimenMercado] = None,
                capital: float = None, capital_inicial: float = None,
                n_posiciones: int = 0) -> tuple:
        """
        Returns:
            (DecisionFinal, ParametrosEstrategia)
        """
        # Estrategia según régimen del activo, ajustada por el mercado
        params = EstrategiaPorRegimen.get(reg.regimen)
        if reg_mercado:
            params = EstrategiaPorRegimen.ajustar_por_mercado(
                params, reg_mercado.regimen)

        bull = self.bull.analizar(ctx, params, reg)
        bear = self.bear.analizar(ctx, params, reg)
        risk = self.risk.evaluar(ctx, bull, bear, params, reg, reg_mercado,
                                 capital, capital_inicial, n_posiciones)

        razon = self._razonamiento(ctx, reg, reg_mercado, params, bull, bear, risk)

        if risk.veto:
            return DecisionFinal(
                ticker=ctx.ticker, accion='ESPERAR', posicion=0, confianza=0.0,
                kelly_fraccion=0.0, razonamiento=razon, vetado=True), params

        edge = bull.confianza - bear.confianza

        # El umbral de confianza depende del régimen
        umbral_efectivo = max(self.min_edge, params.min_confianza - 0.40)

        if abs(edge) < umbral_efectivo:
            return DecisionFinal(
                ticker=ctx.ticker, accion='ESPERAR', posicion=0,
                confianza=round(abs(edge), 3), kelly_fraccion=0.0,
                razonamiento=razon, vetado=False), params

        pos    = 1 if edge > 0 else -1
        accion = 'COMPRAR' if pos == 1 else 'VENDER'

        if pos == 1 and not params.permitir_compra:
            return DecisionFinal(
                ticker=ctx.ticker, accion='ESPERAR', posicion=0,
                confianza=round(abs(edge), 3), kelly_fraccion=0.0,
                razonamiento=razon + "\n  Compras deshabilitadas en este régimen",
                vetado=False), params

        kelly = self._kelly(abs(edge), ctx.volatilidad, ctx.regimen, params)

        return DecisionFinal(
            ticker=ctx.ticker, accion=accion, posicion=pos,
            confianza=round(float(abs(edge)), 3),
            kelly_fraccion=round(float(kelly), 4),
            razonamiento=razon, vetado=False), params

    def _kelly(self, conf: float, vol: float, reg_vol: str,
               params: ParametrosEstrategia) -> float:
        """Kelly fraccionario ajustado por volatilidad y por régimen."""
        p = 0.5 + conf * 0.5
        k = max(0.0, p - (1 - p)) * self.kelly_base

        if vol > 80:   k *= 0.4
        elif vol > 60: k *= 0.6
        elif vol > 40: k *= 0.8

        if reg_vol == 'ESTRÉS':
            k *= 0.5

        # Multiplicador del régimen direccional
        k *= params.kelly_mult

        return float(np.clip(k, 0.0, 0.20))

    def _razonamiento(self, ctx, reg, reg_mercado, params,
                      bull, bear, risk) -> str:
        lines = [
            f"\n{'═'*52}",
            f"DEBATE ADAPTATIVO: {ctx.ticker}  │  {ctx.timestamp}",
            f"{'═'*52}",
            reg.to_texto(),
        ]
        if reg_mercado:
            lines.append(f"\n  MERCADO GENERAL ({reg_mercado.ticker}): "
                         f"{reg_mercado.resumen()}")
        lines += [
            f"\n🎯 ESTRATEGIA SELECCIONADA: {params.nombre}",
            "─" * 46,
            f"  {params.descripcion}",
            f"  Kelly ×{params.kelly_mult:.2f} │ "
            f"umbral {params.min_confianza:.0%} │ "
            f"máx {params.max_posiciones} posiciones",
            f"  Pesos → momentum ×{params.peso_momentum:.1f}  "
            f"reversión ×{params.peso_reversion:.1f}  "
            f"riesgo ×{params.peso_riesgo:.1f}",
            f"  RSI → compra < {params.rsi_compra:.0f}  "
            f"venta > {params.rsi_venta:.0f}",
            ctx.to_texto(),
            f"\n🐂 BULL (confianza: {bull.confianza:.0%})",
            "─" * 46,
        ]
        lines += [f"  {a}" for a in bull.argumentos]
        lines += [f"\n🐻 BEAR (confianza: {bear.confianza:.0%})", "─" * 46]
        lines += [f"  {a}" for a in bear.argumentos]
        lines += [f"\n⚖️  RISK MANAGER", "─" * 46]
        lines += [f"  {a}" for a in risk.argumentos]
        edge = bull.confianza - bear.confianza
        lines += [
            f"\n📊 RESULTADO", "─" * 46,
            f"  Edge neto (Bull − Bear): {edge:+.3f}",
            f"  Veto: {'SÍ' if risk.veto else 'NO'}",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# MÓDULO 6: SISTEMA COMPLETO ADAPTATIVO
# Reemplaza a SistemaAgentes en el bot
# ══════════════════════════════════════════════════════════════
class SistemaAgentesAdaptativo:
    """
    Reemplazo directo de SistemaAgentes con régimen direccional.

    Expone la misma interfaz (analizar_todos, analizar_ticker, kill)
    así que el paper_trading_bot.py funciona sin cambios estructurales:
    solo hay que cambiar qué clase se instancia.
    """

    def __init__(self, tickers: list, capital_inicial: float = 100_000,
                 kelly_fraccion: float = 0.25, max_drawdown: float = 0.20,
                 periodo_regimen: str = '2y'):
        self.tickers    = tickers
        self.capital_0  = capital_inicial
        self.capital    = capital_inicial
        self.dl         = get_data_layer()
        self.detector   = DetectorRegimenDireccional(periodo=periodo_regimen)
        self.motor      = MotorDebateAdaptativo(kelly_base=kelly_fraccion)
        self.kill       = KillSwitch(max_drawdown=max_drawdown)
        self.kill.inicializar(capital_inicial)
        self.decisiones = []
        self.regimenes  = {}          # ticker → RegimenMercado
        self.estrategias = {}         # ticker → ParametrosEstrategia
        self.n_posiciones = 0         # el bot lo actualiza antes del ciclo
        self.log_file   = 'debate_log.txt'

        logger.info(f"SistemaAgentesAdaptativo iniciado")
        logger.info(f"  Tickers: {', '.join(tickers)}")
        logger.info(f"  Capital: ${capital_inicial:,.0f} │ "
                    f"Kelly base: {kelly_fraccion:.0%}")

    def actualizar_posiciones(self, n: int):
        """El bot llama a esto con el número de posiciones abiertas."""
        self.n_posiciones = n

    def analizar_ticker(self, ticker: str,
                        verbose: bool = True) -> Optional[DecisionFinal]:
        activo, razon = self.kill.verificar(self.capital)
        if activo:
            logger.warning(f"⛔ {ticker}: {razon}")
            return None

        # Contexto (mismo DataLayer de siempre)
        ctx_dict = self.dl.get_contexto_agentes(ticker)
        if not ctx_dict:
            logger.error(f"Sin contexto para {ticker}")
            return None
        ctx = ContextoMercado(**ctx_dict)

        # Régimen del activo y del mercado general
        reg = self.detector.detectar(ticker)
        if reg is None:
            logger.warning(f"Sin régimen para {ticker} — se omite")
            return None

        info = normalizar_ticker(ticker)
        reg_mercado = self.detector.regimen_mercado_general(info['es_crypto'])

        # Debate
        decision, params = self.motor.debatir(
            ctx, reg, reg_mercado,
            capital=self.capital, capital_inicial=self.capital_0,
            n_posiciones=self.n_posiciones,
        )

        self.regimenes[ticker]   = reg
        self.estrategias[ticker] = params

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(decision.razonamiento + "\n\n")

        if verbose:
            emoji_a = ('🟢' if decision.accion == 'COMPRAR' else
                       '🔴' if decision.accion == 'VENDER' else '⚪')
            emoji_r = {'BULL': '🐂', 'BEAR': '🐻', 'SIDEWAYS': '↔️'}[reg.regimen]
            veto = ' [VETADO]' if decision.vetado else ''
            print(f"  {emoji_a} {ticker:<12} {decision.accion:<8} │ "
                  f"{emoji_r} {reg.regimen:<9} │ "
                  f"conf={decision.confianza:.2f} │ "
                  f"kelly={decision.kelly_fraccion:.1%} │ "
                  f"{params.nombre.split(' /')[0][:14]}{veto}")

        self.decisiones.append(decision)
        return decision

    def analizar_todos(self, verbose: bool = True) -> list:
        print(f"\n{'═'*72}")
        print(f"  DEBATE ADAPTATIVO — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

        # Régimen de los dos mercados generales
        hay_crypto = any(normalizar_ticker(t)['es_crypto'] for t in self.tickers)
        hay_stocks = any(not normalizar_ticker(t)['es_crypto'] for t in self.tickers)

        if hay_crypto:
            rc = self.detector.regimen_mercado_general(True)
            if rc: print(f"  MERCADO CRYPTO (BTC):  {rc.resumen()}")
        if hay_stocks:
            rs = self.detector.regimen_mercado_general(False)
            if rs: print(f"  MERCADO STOCKS (SPY):  {rs.resumen()}")

        if hay_crypto and self.dl.coingecko.disponible:
            m  = self.dl.get_macro_crypto()
            fg = m['fear_greed']
            print(f"  Fear&Greed={fg['valor']} ({fg['nombre']}) │ "
                  f"BTC Dom={m['dominancia_btc']:.0f}%")

        print(f"{'═'*72}\n")

        decisiones = []
        for t in self.tickers:
            try:
                d = self.analizar_ticker(t, verbose)
                if d: decisiones.append(d)
            except Exception as e:
                logger.error(f"Error en {t}: {e}")
            time.sleep(0.25)

        c  = sum(1 for d in decisiones if d.accion == 'COMPRAR')
        v  = sum(1 for d in decisiones if d.accion == 'VENDER')
        vt = sum(1 for d in decisiones if d.vetado)

        # Distribución de regímenes
        dist = {}
        for r in self.regimenes.values():
            dist[r.regimen] = dist.get(r.regimen, 0) + 1

        print(f"\n  {'─'*62}")
        print(f"  {c} compras │ {v} ventas │ "
              f"{len(decisiones)-c-v} esperas │ {vt} vetados")
        if dist:
            print(f"  Regímenes: " + " │ ".join(
                f"{k}={v2}" for k, v2 in sorted(dist.items())))
        print(f"  Log completo en: {self.log_file}\n")

        return decisiones

    def reporte_regimenes(self):
        """Tabla con el régimen de cada activo."""
        if not self.regimenes:
            print("Ejecutá analizar_todos() primero")
            return

        print(f"\n{'═'*76}")
        print("  REGÍMENES DETECTADOS")
        print(f"{'═'*76}")
        print(f"  {'Ticker':<12} {'Régimen':<10} {'Score':>7} {'Fuerza':>7} "
              f"{'Tend/año':>9} {'ADX':>6}  Estrategia")
        print("  " + "─" * 72)
        for t, r in self.regimenes.items():
            p = self.estrategias.get(t)
            est = p.nombre.split(' /')[0][:20] if p else '—'
            emoji = {'BULL': '🐂', 'BEAR': '🐻', 'SIDEWAYS': '↔️'}[r.regimen]
            print(f"  {t:<12} {emoji} {r.regimen:<8} {r.score:>+7.0f} "
                  f"{r.fuerza:>7.0f} {r.tendencia_anual:>+8.0f}% "
                  f"{r.adx:>6.0f}  {est}")
        print(f"{'═'*76}\n")


# ══════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':

    # ── Test 1: Detección de régimen suelta ───────────────────
    print("\n" + "═"*60)
    print("  TEST 1 — DETECCIÓN DE RÉGIMEN")
    print("═"*60)

    det = DetectorRegimenDireccional()
    for ticker in ['BTCUSDT', 'ETHUSDT', 'SPY', 'NVDA']:
        reg = det.detectar(ticker)
        if reg:
            print(f"\n{reg.to_texto()}")
            params = EstrategiaPorRegimen.get(reg.regimen)
            print(f"  → Estrategia: {params.nombre}")
            print(f"    {params.descripcion}")

    # ── Test 2: Sistema completo ──────────────────────────────
    print("\n\n" + "═"*60)
    print("  TEST 2 — SISTEMA ADAPTATIVO COMPLETO")
    print("═"*60)

    sistema = SistemaAgentesAdaptativo(
        tickers=['BTCUSDT', 'ETHUSDT', 'SOLUSDT',
                 'AAPL', 'NVDA', 'SPY'],
        capital_inicial=100_000,
        kelly_fraccion=0.25,
        max_drawdown=0.20,
    )
    sistema.analizar_todos()
    sistema.reporte_regimenes()
