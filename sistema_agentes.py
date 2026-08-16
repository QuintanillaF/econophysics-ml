"""
╔══════════════════════════════════════════════════════════════════╗
║   SISTEMA DE AGENTES — Bull / Bear / Risk Manager               ║
║   v2.0 — Usa DataLayer (Binance + CoinGecko + yfinance)         ║
╚══════════════════════════════════════════════════════════════════╝

INSTALACIÓN:
  pip install numpy scipy pandas yfinance requests scikit-learn
"""

import time, logging
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import warnings; warnings.filterwarnings('ignore')

from data_layer import get_data_layer, normalizar_ticker, TOP_10_BINANCE

logger = logging.getLogger('Agentes')
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s │ %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(h)


# ══════════════════════════════════════════════════════════════
# CONTEXTO DE MERCADO
# ══════════════════════════════════════════════════════════════
@dataclass
class ContextoMercado:
    ticker:          str
    precio_actual:   float
    retorno_1h:      float
    retorno_24h:     float
    retorno_7d:      float
    volatilidad:     float
    hurst:           float
    regimen:         str
    rsi:             float
    bb_posicion:     float
    volumen_ratio:   float
    señal_ml:        int
    prob_ml:         float
    whales_señal:    str
    whales_detalle:  str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_texto(self) -> str:
        return f"""
CONTEXTO — {self.ticker} — {self.timestamp}
{'='*50}
PRECIO:      ${self.precio_actual:,.4f}
RETORNOS:    1h={self.retorno_1h:+.2f}%  24h={self.retorno_24h:+.2f}%  7d={self.retorno_7d:+.2f}%
VOLATILIDAD: {self.volatilidad:.1f}% anual
HURST:       {self.hurst:.3f} ({'PERSISTENTE' if self.hurst>0.55 else 'REVERSIVO' if self.hurst<0.45 else 'ALEATORIO'})
RÉGIMEN:     {self.regimen}
RSI:         {self.rsi:.1f} ({'SOBRECOMPRADO' if self.rsi>70 else 'SOBREVENDIDO' if self.rsi<30 else 'NEUTRO'})
BB POS:      {self.bb_posicion:+.2f}
VOLUMEN:     {self.volumen_ratio:.1f}x
ML:          {'COMPRAR' if self.señal_ml==1 else 'VENDER' if self.señal_ml==-1 else 'ESPERAR'} (prob={self.prob_ml:.0%})
BALLENAS:    {self.whales_señal} — {self.whales_detalle}"""


# ══════════════════════════════════════════════════════════════
# ARGUMENTOS
# ══════════════════════════════════════════════════════════════
@dataclass
class ArgumentoAgente:
    rol:        str
    posicion:   int
    confianza:  float
    argumentos: list
    veto:       bool = False


# ══════════════════════════════════════════════════════════════
# AGENTE BULL
# ══════════════════════════════════════════════════════════════
class AgenteAlcista:
    ROL = 'BULL'

    def analizar(self, ctx: ContextoMercado) -> ArgumentoAgente:
        pts, max_pts, args = 0.0, 0.0, []

        max_pts += 2
        if ctx.hurst > 0.60:   pts+=2; args.append(f"✅ Hurst={ctx.hurst:.3f}: tendencia fuerte")
        elif ctx.hurst > 0.53: pts+=1; args.append(f"✅ Hurst={ctx.hurst:.3f}: leve persistencia")
        else: args.append(f" Hurst={ctx.hurst:.3f}: sin tendencia")

        max_pts += 1.5
        if ctx.regimen == 'CALMA': pts+=1.5; args.append("✅ Régimen calmo")
        else: args.append("  Régimen de estrés")

        max_pts += 2
        if ctx.rsi < 30:   pts+=2; args.append(f"✅ RSI={ctx.rsi:.1f}: sobrevendido → rebote probable")
        elif ctx.rsi < 45: pts+=1; args.append(f"✅ RSI={ctx.rsi:.1f}: zona de valor")
        elif ctx.rsi > 70: args.append(f"⚠️  RSI={ctx.rsi:.1f}: sobrecomprado")
        else: args.append(f" RSI={ctx.rsi:.1f}: neutral")

        max_pts += 1.5
        if ctx.bb_posicion < -0.6:   pts+=1.5; args.append("✅ Precio en banda inferior BB")
        elif ctx.bb_posicion < 0:    pts+=0.5; args.append("➖ Precio bajo media BB")

        max_pts += 2
        if ctx.señal_ml == 1:   pts+=2*ctx.prob_ml; args.append(f"✅ ML: COMPRAR (prob={ctx.prob_ml:.0%})")
        elif ctx.señal_ml == -1: args.append(f"❌ ML: VENDER")

        max_pts += 1.5
        if ctx.whales_señal == 'ALCISTA':  pts+=1.5; args.append(f"✅ Ballenas: {ctx.whales_detalle}")
        elif ctx.whales_señal == 'BAJISTA': args.append(f"❌ Ballenas bajistas")

        max_pts += 0.5
        if ctx.volumen_ratio > 1.5: pts+=0.5; args.append(f"✅ Volumen {ctx.volumen_ratio:.1f}x")

        max_pts += 1
        if ctx.retorno_7d > 3:    pts+=1; args.append(f"✅ Momentum 7d: {ctx.retorno_7d:+.1f}%")
        elif ctx.retorno_7d < -5: args.append(f"⚠️  Caída 7d: {ctx.retorno_7d:+.1f}%")

        c = pts/max_pts if max_pts > 0 else 0
        return ArgumentoAgente(rol=self.ROL, posicion=1 if c>0.5 else 0, confianza=round(c,3), argumentos=args)


# ══════════════════════════════════════════════════════════════
# AGENTE BEAR
# ══════════════════════════════════════════════════════════════
class AgenteBajista:
    ROL = 'BEAR'

    def analizar(self, ctx: ContextoMercado) -> ArgumentoAgente:
        pts, max_pts, args = 0.0, 0.0, []

        max_pts += 2
        if ctx.regimen == 'ESTRÉS': pts+=2; args.append("🔴 Régimen ESTRÉS")
        else: args.append("✅ Régimen calmo")

        max_pts += 2
        if ctx.rsi > 70:   pts+=2; args.append(f"🔴 RSI={ctx.rsi:.1f}: sobrecomprado")
        elif ctx.rsi > 60: pts+=0.8; args.append(f"⚠️  RSI={ctx.rsi:.1f}: precaución")
        else: args.append(f"➖ RSI={ctx.rsi:.1f}: sin señal bajista")

        max_pts += 1.5
        if ctx.volatilidad > 80:   pts+=1.5; args.append(f"🔴 Volatilidad {ctx.volatilidad:.0f}%: extrema")
        elif ctx.volatilidad > 50: pts+=0.7; args.append(f"⚠️  Volatilidad {ctx.volatilidad:.0f}%: elevada")

        max_pts += 1.5
        if ctx.hurst < 0.42:   pts+=1.5; args.append(f"🔴 Hurst={ctx.hurst:.3f}: mercado reversivo")
        elif ctx.hurst < 0.48: pts+=0.5; args.append(f"⚠️  Hurst={ctx.hurst:.3f}: leve reversión")

        max_pts += 2
        if ctx.señal_ml == -1:  pts+=2*ctx.prob_ml; args.append(f"🔴 ML: VENDER (prob={ctx.prob_ml:.0%})")
        elif ctx.señal_ml == 1: args.append(f"✅ ML dice comprar")

        max_pts += 1.5
        if ctx.whales_señal == 'BAJISTA':  pts+=1.5; args.append(f"🔴 Ballenas: {ctx.whales_detalle}")
        elif ctx.whales_señal == 'ALCISTA': args.append(f"✅ Ballenas alcistas")

        max_pts += 1
        if ctx.retorno_24h < -3: pts+=1; args.append(f"🔴 Caída 24h: {ctx.retorno_24h:+.1f}%")
        elif ctx.retorno_24h > 5: pts+=0.3; args.append(f"⚠️  Subida fuerte 24h: posible agotamiento")

        max_pts += 1
        if ctx.bb_posicion > 0.6: pts+=1; args.append("🔴 Precio en banda superior BB")

        c = pts/max_pts if max_pts > 0 else 0
        return ArgumentoAgente(rol=self.ROL, posicion=-1 if c>0.5 else 0, confianza=round(c,3), argumentos=args)


# ══════════════════════════════════════════════════════════════
# RISK MANAGER
# ══════════════════════════════════════════════════════════════
class AgenteRiskManager:
    ROL = 'RISK'

    def __init__(self, max_vol: float = 100.0, min_consenso: float = 0.15):
        self.max_vol = max_vol; self.min_consenso = min_consenso

    def evaluar(self, ctx: ContextoMercado, bull: ArgumentoAgente,
                bear: ArgumentoAgente, capital: float = None,
                capital_0: float = None) -> ArgumentoAgente:
        vetos, args = [], []

        if ctx.volatilidad > self.max_vol:
            vetos.append(f"Volatilidad {ctx.volatilidad:.0f}% > límite {self.max_vol:.0f}%")

        if ctx.regimen == 'ESTRÉS' and ctx.volatilidad > 60:
            vetos.append("Régimen ESTRÉS + vol > 60%")

        if capital and capital_0:
            dd = (capital_0 - capital) / capital_0
            if dd > 0.15: vetos.append(f"Portfolio drawdown {dd:.0%}")
            if dd > 0.20: vetos.append(f"DRAWDOWN CRÍTICO {dd:.0%} — VETO TOTAL")

        dif = abs(bull.confianza - bear.confianza)
        if dif < self.min_consenso:
            vetos.append(f"Señales equilibradas Bull={bull.confianza:.2f} Bear={bear.confianza:.2f}")

        if bull.posicion == 1 and ctx.rsi > 78:
            vetos.append(f"Bull quiere comprar pero RSI={ctx.rsi:.0f}: extremo")
        if bear.posicion == -1 and ctx.rsi < 22:
            vetos.append(f"Bear quiere vender pero RSI={ctx.rsi:.0f}: extremo")

        hay_veto = len(vetos) > 0
        if hay_veto:
            args.append("🛑 VETO ACTIVADO")
            for v in vetos: args.append(f"  → {v}")
        else:
            args.append(f"✅ Sin objeciones (dif confianza={dif:.2f})")

        return ArgumentoAgente(rol=self.ROL, posicion=0, confianza=0.0, argumentos=args, veto=hay_veto)


# ══════════════════════════════════════════════════════════════
# DECISIÓN FINAL
# ══════════════════════════════════════════════════════════════
@dataclass
class DecisionFinal:
    ticker:         str
    accion:         str
    posicion:       int
    confianza:      float
    kelly_fraccion: float
    razonamiento:   str
    vetado:         bool
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ══════════════════════════════════════════════════════════════
# MOTOR DE DEBATE
# ══════════════════════════════════════════════════════════════
class MotorDebate:
    def __init__(self, kelly_fraccion_base: float = 0.25, min_confianza: float = 0.15):
        self.bull    = AgenteAlcista()
        self.bear    = AgenteBajista()
        self.risk    = AgenteRiskManager()
        self.kelly_f = kelly_fraccion_base
        self.min_c   = min_confianza

    def debatir(self, ctx: ContextoMercado, capital: float = None,
                capital_inicial: float = None) -> DecisionFinal:
        bull = self.bull.analizar(ctx)
        bear = self.bear.analizar(ctx)
        risk = self.risk.evaluar(ctx, bull, bear, capital, capital_inicial)

        razon = self._razonamiento(ctx, bull, bear, risk)

        if risk.veto:
            return DecisionFinal(ticker=ctx.ticker, accion='ESPERAR', posicion=0,
                                  confianza=0.0, kelly_fraccion=0.0, razonamiento=razon, vetado=True)

        edge = bull.confianza - bear.confianza
        if abs(edge) < self.min_c:
            return DecisionFinal(ticker=ctx.ticker, accion='ESPERAR', posicion=0,
                                  confianza=abs(edge), kelly_fraccion=0.0, razonamiento=razon, vetado=False)

        pos    = 1 if edge > 0 else -1
        conf   = abs(edge)
        kelly  = self._kelly(conf, ctx.volatilidad, ctx.regimen)
        accion = 'COMPRAR' if pos == 1 else 'VENDER'

        return DecisionFinal(ticker=ctx.ticker, accion=accion, posicion=pos,
                              confianza=round(conf,3), kelly_fraccion=round(kelly,4),
                              razonamiento=razon, vetado=False)

    def _kelly(self, conf: float, vol: float, reg: str) -> float:
        p = 0.5 + conf*0.5; q = 1-p
        k = max(0, (p-q)) * self.kelly_f
        if vol > 80: k *= 0.4
        elif vol > 60: k *= 0.6
        elif vol > 40: k *= 0.8
        if reg == 'ESTRÉS': k *= 0.5
        return float(np.clip(k, 0, 0.20))

    def _razonamiento(self, ctx, bull, bear, risk) -> str:
        lines = [f"\n{'═'*50}", f"DEBATE: {ctx.ticker}", f"{'═'*50}",
                 ctx.to_texto(), f"\n🐂 BULL (confianza: {bull.confianza:.0%})", "─"*40]
        lines += [f"  {a}" for a in bull.argumentos]
        lines += [f"\n🐻 BEAR (confianza: {bear.confianza:.0%})", "─"*40]
        lines += [f"  {a}" for a in bear.argumentos]
        lines += [f"\n⚖️  RISK MANAGER", "─"*40]
        lines += [f"  {a}" for a in risk.argumentos]
        edge = bull.confianza - bear.confianza
        lines += [f"\n📊 RESULTADO", "─"*40, f"  Edge neto: {edge:+.3f}",
                  f"  Veto: {'SÍ' if risk.veto else 'NO'}"]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
# KILL SWITCH
# ══════════════════════════════════════════════════════════════
class KillSwitch:
    def __init__(self, max_drawdown: float = 0.20, max_perdidas_seq: int = 5, max_trades_hora: int = 20):
        self.max_dd    = max_drawdown
        self.max_perd  = max_perdidas_seq
        self.max_th    = max_trades_hora
        self.activado  = False; self.razon = ""
        self.perd_seq  = 0; self.trades   = []
        self.capital_0 = None

    def inicializar(self, capital: float): self.capital_0 = capital

    def verificar(self, capital: float, ganó: bool = None) -> tuple:
        if self.activado: return True, self.razon
        if self.capital_0:
            dd = (self.capital_0-capital)/self.capital_0
            if dd >= self.max_dd: self._activar(f"Drawdown {dd:.0%} ≥ {self.max_dd:.0%}"); return True, self.razon
        if ganó is not None:
            self.perd_seq = 0 if ganó else self.perd_seq+1
        if self.perd_seq >= self.max_perd: self._activar(f"{self.perd_seq} pérdidas consecutivas"); return True, self.razon
        ahora = datetime.now()
        self.trades = [t for t in self.trades if (ahora-t).seconds < 3600]
        if len(self.trades) >= self.max_th: self._activar(f"{len(self.trades)} trades/hora"); return True, self.razon
        return False, ""

    def registrar_trade(self): self.trades.append(datetime.now())
    def resetear(self): self.activado=False; self.razon=""; self.perd_seq=0; logger.info("Kill switch reseteado")
    def _activar(self, r): self.activado=True; self.razon=r; logger.critical(f"🛑 KILL SWITCH: {r}")


# ══════════════════════════════════════════════════════════════
# SISTEMA PRINCIPAL
# ══════════════════════════════════════════════════════════════
class SistemaAgentes:
    """
    Sistema completo: DataLayer → Contexto → Debate → Decisión.
    Funciona con cualquier ticker: BTCUSDT, BTC-USD, AAPL, etc.
    """

    def __init__(self, tickers: list, capital_inicial: float = 100_000,
                 kelly_fraccion: float = 0.25, max_drawdown: float = 0.20):
        self.tickers   = tickers
        self.capital_0 = capital_inicial
        self.capital   = capital_inicial
        self.dl        = get_data_layer()
        self.motor     = MotorDebate(kelly_fraccion_base=kelly_fraccion)
        self.kill      = KillSwitch(max_drawdown=max_drawdown)
        self.kill.inicializar(capital_inicial)
        self.decisiones = []
        self.log_file   = 'debate_log.txt'

        fuentes = []
        for t in tickers:
            info = normalizar_ticker(t)
            if info['es_crypto'] and self.dl.binance.disponible: fuentes.append('Binance')
            else: fuentes.append('yfinance')
        if self.dl.coingecko.disponible: fuentes.append('CoinGecko')

        logger.info(f"SistemaAgentes iniciado → {', '.join(set(fuentes))}")
        logger.info(f"Tickers: {', '.join(tickers)}")
        logger.info(f"Capital: ${capital_inicial:,.0f}  Kelly: {kelly_fraccion:.0%}  MaxDD: {max_drawdown:.0%}")

    def analizar_ticker(self, ticker: str, verbose: bool = True) -> Optional[DecisionFinal]:
        activo, razon = self.kill.verificar(self.capital)
        if activo:
            logger.warning(f"⛔ {ticker}: {razon}"); return None

        ctx_dict = self.dl.get_contexto_agentes(ticker)
        if not ctx_dict:
            logger.error(f"Sin contexto para {ticker}"); return None

        ctx      = ContextoMercado(**ctx_dict)
        decision = self.motor.debatir(ctx, self.capital, self.capital_0)

        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(decision.razonamiento + "\n\n")

        if verbose:
            emoji = '🟢' if decision.accion=='COMPRAR' else '🔴' if decision.accion=='VENDER' else '⚪'
            info  = normalizar_ticker(ticker)
            fuente = 'Binance' if (info['es_crypto'] and self.dl.binance.disponible) else 'yfinance'
            veto   = ' [VETADO]' if decision.vetado else ''
            print(f"  {emoji} {ticker:<14} {decision.accion:<8} │ "
                  f"conf={decision.confianza:.2f} │ kelly={decision.kelly_fraccion:.1%} │ {fuente}{veto}")

        self.decisiones.append(decision)
        self.kill.registrar_trade()
        return decision

    def analizar_todos(self, verbose: bool = True) -> list:
        print(f"\n{'═'*55}")
        print(f"  DEBATE MULTI-AGENTE — {datetime.now().strftime('%H:%M:%S')}")

        if any(normalizar_ticker(t)['es_crypto'] for t in self.tickers) and self.dl.coingecko.disponible:
            m  = self.dl.get_macro_crypto(); fg = m['fear_greed']
            print(f"  F&G={fg['valor']} ({fg['nombre']}) │ BTC={m['dominancia_btc']:.0f}% │ {m['sentimiento']}")
        print(f"{'═'*55}\n")

        decisiones = []
        for t in self.tickers:
            try:
                d = self.analizar_ticker(t, verbose)
                if d: decisiones.append(d)
            except Exception as e:
                logger.error(f"Error {t}: {e}")
            time.sleep(0.3)

        c = sum(1 for d in decisiones if d.accion=='COMPRAR')
        v = sum(1 for d in decisiones if d.accion=='VENDER')
        vt = sum(1 for d in decisiones if d.vetado)
        print(f"\n  {'─'*45}")
        print(f"  {c} compras │ {v} ventas │ {len(decisiones)-c-v} esperas │ {vt} vetados")
        print(f"  Log guardado en: {self.log_file}\n")
        return decisiones

    def reporte(self) -> dict:
        if not self.decisiones: return {'total':0}
        c = len(self.decisiones); vt = sum(1 for d in self.decisiones if d.vetado)
        cp = float(np.mean([d.confianza for d in self.decisiones if not d.vetado])) if c>vt else 0
        return {'total':c,'compras':sum(1 for d in self.decisiones if d.accion=='COMPRAR'),
                'ventas':sum(1 for d in self.decisiones if d.accion=='VENDER'),
                'vetados':vt,'confianza_prom':round(cp,3),'kill_activo':self.kill.activado}


if __name__ == '__main__':
    sistema = SistemaAgentes(
        tickers         = TOP_10_BINANCE + ['AAPL','NVDA','TSLA','SPY','MSFT','AMZN','GOOGL','META','AMD','PLTR'],
        capital_inicial = 100_000,
        kelly_fraccion  = 0.25,
        max_drawdown    = 0.20,
    )
    sistema.analizar_todos()
    rep = sistema.reporte()
    print("📊 REPORTE:")
    for k,v in rep.items(): print(f"  {k}: {v}")
