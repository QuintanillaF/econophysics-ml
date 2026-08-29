"""
╔══════════════════════════════════════════════════════════════════╗
║   PAPER TRADING BOT — v2.1 CORREGIDO                            ║
║   Alpaca (stocks) + Binance Testnet (crypto)                    ║
╚══════════════════════════════════════════════════════════════════╝

CAMBIOS v2.1:
  - Kill Switch arreglado: solo cuenta trades reales, no análisis
  - Kill Switch límite subido a 200 trades/hora (era 20)
  - MIN_CONFIANZA bajada a 0.55 (era 0.60) para más señales
  - Logs más claros: muestra exactamente qué pasa en cada ciclo
  - Mercado cerrado: avisa en lugar de fallar silenciosamente

INSTALACIÓN:
  pip install requests schedule yfinance numpy pandas scikit-learn
"""

import time, json, logging, requests, schedule, numpy as np, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')
from regimen_direccional import SistemaAgentesAdaptativo
from data_layer import get_data_layer, normalizar_ticker, TOP_10_BINANCE
from sistema_agentes import SistemaAgentes

logger = logging.getLogger('PaperBot')
logger.setLevel(logging.INFO)
if not logger.handlers:
    fh = logging.FileHandler('paper_trading.log', encoding='utf-8')
    ch = logging.StreamHandler()
    fmt = logging.Formatter('[%(asctime)s] %(levelname)-8s │ %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt); ch.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(ch)


# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
CONFIG = {
    # ── Pegá tus API keys acá ─────────────────────────────────
    'ALPACA_API_KEY':    '--',
    'ALPACA_SECRET_KEY': '--',
    'ALPACA_BASE_URL':   'https://paper-api.alpaca.markets',

    'BINANCE_API_KEY':    '--',
    'BINANCE_SECRET_KEY': '--',
    'BINANCE_TESTNET':    True,

    # ── Parámetros de trading ─────────────────────────────────
    'CAPITAL_PAPER':     15_527,
    'CAPITAL_BINANCE':   50,
    'MAX_POSICION_PCT':  0.10,      # 10% del capital por posición
    'STOP_LOSS_PCT':     0.05,      # stop loss 5%
    'TAKE_PROFIT_PCT':   0.10,      # take profit 10%
    'MIN_CONFIANZA':     0.55,      # BAJADO de 0.60 a 0.55 → más señales
    'COOLDOWN_HORAS':    4,         # esperar 4h entre trades del mismo activo

    # ── Activos a monitorear ──────────────────────────────────
    'STOCKS_PAPER':  ['AAPL', 'NVDA', 'TSLA', 'SPY', 'MSFT',
                      'AMZN', 'GOOGL', 'META', 'AMD', 'PLTR', 'YPF'],
    'CRYPTO_PAPER':  ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
                      'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'LINKUSDT', 'DOTUSDT'],

    'LOG_FILE':    'paper_trading.log',
    'TRADES_FILE': 'trades_log.json',
}


# ══════════════════════════════════════════════════════════════
# KILL SWITCH CORREGIDO
# ══════════════════════════════════════════════════════════════
class KillSwitchBot:
    """
    Versión corregida del Kill Switch para el bot.

    DIFERENCIA CON LA VERSIÓN ANTERIOR:
      Antes: contaba cada análisis como un "trade" → bloqueaba todo
      Ahora: solo cuenta órdenes realmente ejecutadas

    LÍMITES:
      max_trades_hora: 200 (antes era 20, demasiado restrictivo)
      max_drawdown:    20% del portfolio
      max_perdidas_seq: 8 pérdidas consecutivas
    """

    def __init__(self, capital_inicial: float,
                 max_drawdown: float = 0.20,
                 max_perdidas_seq: int = 8,
                 max_trades_hora: int = 200):   # ← subido de 20 a 200
        self.capital_0    = capital_inicial
        self.max_dd       = max_drawdown
        self.max_perd     = max_perdidas_seq
        self.max_th       = max_trades_hora
        self.activado     = False
        self.razon        = ""
        self.perd_seq     = 0
        self.trades_reales = []   # ← solo trades ejecutados, no análisis

    def verificar(self, capital_actual: float) -> tuple:
        """Retorna (bloqueado: bool, razon: str)"""
        if self.activado:
            return True, self.razon

        # Condición 1: Drawdown máximo del portfolio
        if self.capital_0 > 0:
            dd = (self.capital_0 - capital_actual) / self.capital_0
            if dd >= self.max_dd:
                self._activar(f"Drawdown {dd:.0%} alcanzó límite {self.max_dd:.0%}")
                return True, self.razon

        # Condición 2: Pérdidas consecutivas
        if self.perd_seq >= self.max_perd:
            self._activar(f"{self.perd_seq} pérdidas consecutivas")
            return True, self.razon

        # Condición 3: Demasiados trades ejecutados en 1 hora
        ahora = datetime.now()
        self.trades_reales = [
            t for t in self.trades_reales
            if (ahora - t).seconds < 3600
        ]
        if len(self.trades_reales) >= self.max_th:
            self._activar(f"{len(self.trades_reales)} trades en 1 hora — posible bug")
            return True, self.razon

        return False, ""

    def registrar_trade_ejecutado(self, ganó: bool = None):
        """
        Llamar SOLO cuando se ejecuta una orden real.
        NO llamar durante el análisis.
        """
        self.trades_reales.append(datetime.now())
        if ganó is not None:
            self.perd_seq = 0 if ganó else self.perd_seq + 1

    def resetear(self):
        self.activado = False
        self.razon = ""
        self.perd_seq = 0
        logger.info("⚡ Kill switch reseteado")

    def _activar(self, razon: str):
        self.activado = True
        self.razon = razon
        logger.critical(f"\n🛑 KILL SWITCH ACTIVADO: {razon}\n")

    def estado(self) -> str:
        if self.activado:
            return f"🛑 ACTIVO — {self.razon}"
        trades_1h = len(self.trades_reales)
        return (f"✅ Normal │ "
                f"Trades última hora: {trades_1h}/{self.max_th} │ "
                f"Pérdidas consecutivas: {self.perd_seq}/{self.max_perd}")


# ══════════════════════════════════════════════════════════════
# CONECTOR ALPACA
# ══════════════════════════════════════════════════════════════
class AlpacaConnector:
    def __init__(self, api_key, secret_key, base_url):
        self.headers = {
            'APCA-API-KEY-ID':     api_key,
            'APCA-API-SECRET-KEY': secret_key,
            'Content-Type':        'application/json'
        }
        self.base = base_url.rstrip('/')
        self.base_data = 'https://data.alpaca.markets'

    def _get(self, ep):
        try:
            r = requests.get(f"{self.base}{ep}", headers=self.headers, timeout=10)
            r.raise_for_status(); return r.json()
        except Exception as e:
            logger.error(f"Alpaca GET {ep}: {e}"); return {}

    def _post(self, ep, data):
        try:
            r = requests.post(f"{self.base}{ep}", headers=self.headers, json=data, timeout=10)
            r.raise_for_status(); return r.json()
        except Exception as e:
            logger.error(f"Alpaca POST {ep}: {e}"); return {}

    def _delete(self, ep):
        try:
            return requests.delete(f"{self.base}{ep}", headers=self.headers, timeout=10).status_code in [200, 204]
        except:
            return False

    def get_cuenta(self) -> dict:
        d = self._get('/v2/account')
        if not d: return {}
        return {
            'capital':       float(d.get('equity', 0)),
            'cash':          float(d.get('cash', 0)),
            'buying_power':  float(d.get('buying_power', 0)),
            'portfolio_val': float(d.get('portfolio_value', 0)),
            'status':        d.get('status', 'unknown'),
        }

    def mercado_abierto(self) -> bool:
        """Verifica si el mercado de stocks está abierto ahora."""
        d = self._get('/v2/clock')
        return bool(d.get('is_open', False))

    def get_posiciones(self) -> list:
        data = self._get('/v2/positions')
        if not isinstance(data, list): return []
        return [{
            'symbol':  p.get('symbol'),
            'qty':     float(p.get('qty', 0)),
            'side':    p.get('side'),
            'valor':   float(p.get('market_value', 0)),
            'pnl':     float(p.get('unrealized_pl', 0)),
            'pnl_pct': float(p.get('unrealized_plpc', 0)) * 100,
        } for p in data]

    def tiene_posicion(self, symbol: str) -> bool:
        return any(p['symbol'] == symbol for p in self.get_posiciones())

    def get_precio_actual(self, symbol: str) -> float:
        """Obtiene el precio actual apuntando al servidor de Market Data."""
        try:
            # Usamos la URL base de datos externa para que no de 404
            url = f"{self.base_data}/v2/stocks/{symbol}/quotes/latest"
            r = requests.get(url, headers=self.headers, timeout=5)
            r.raise_for_status()
            
            data = r.json()
            # Estructura oficial de Alpaca Data v2: data['quote']['ap'] (Ask Price)
            precio = float(data.get('quote', {}).get('ap', 0))
            
            if precio == 0:
                # Si viene en 0 por falta de liquidez instantánea, probamos con el Bid Price (bp)
                precio = float(data.get('quote', {}).get('bp', 0))
                
            return precio
        except Exception as e:
            logger.error(f"Error al obtener precio Alpaca para {symbol}: {e}")
            return 0.0

    def enviar_bracket_order(self, symbol, qty, side, sl_pct, tp_pct) -> dict:
        """Envía orden con stop loss y take profit automáticos (Versión Blindada)."""
        precio = self.get_precio_actual(symbol)
        if precio == 0:
            logger.error(f"No se pudo obtener precio de {symbol}")
            return {}

        # ── BLINDAJE 1: Forzar enteros o máximo 2 decimales para acciones ──
        # Si la acción vale más de $100, compramos solo unidades enteras para evitar problemas de fracciones en Alpaca
        if precio > 100:
            qty_limpia = int(qty)
        else:
            qty_limpia = round(qty, 2)

        # Si el redondeo da cero (porque el capital asignado era muy chico), compramos mínimo 1 acción
        if qty_limpia <= 0:
            qty_limpia = 1

        # ── BLINDAJE 2: Recalcular P&L con los datos reales limpios ──
        sl = round(precio * (1 - sl_pct) if side == 'buy' else precio * (1 + sl_pct), 2)
        tp = round(precio * (1 + tp_pct) if side == 'buy' else precio * (1 - tp_pct), 2)

        orden = {
            'symbol':        symbol,
            'qty':           str(qty_limpia),  # Pasamos el número limpio convertido a texto
            'side':          side,
            'type':          'market',
            'time_in_force': 'gtc',
            'order_class':   'bracket',
            'stop_loss':     {'stop_price': str(sl)},
            'take_profit':   {'limit_price': str(tp)},
        }

        monto_estimado = qty_limpia * precio
        logger.info(f"📤 [CORREGIDA] BRACKET {side.upper()} {qty_limpia} {symbol} "
                    f"@ ${precio:.2f} │ Total Est: ${monto_estimado:.2f} │ SL=${sl} TP=${tp}")
        
        return self._post('/v2/orders', orden)

    def cerrar_posicion(self, symbol: str) -> bool:
        logger.info(f"🔒 Cerrando posición: {symbol}")
        return self._delete(f'/v2/positions/{symbol}')


# ══════════════════════════════════════════════════════════════
# CONECTOR BINANCE
# ══════════════════════════════════════════════════════════════
class BinanceConnector:
    BASE_TESTNET = 'https://testnet.binance.vision/api'
    BASE_REAL    = 'https://api.binance.com/api'

    def __init__(self, api_key, secret_key, testnet=True):
        import hmac, hashlib
        self._hmac = hmac; self._sha = hashlib
        self.api_key = api_key; self.secret = secret_key.encode()
        self.base = self.BASE_TESTNET if testnet else self.BASE_REAL
        self.headers = {'X-MBX-APIKEY': api_key}

    def _sign(self, params):
        params['timestamp'] = int(time.time() * 1000)
        q = '&'.join(f"{k}={v}" for k, v in params.items())
        params['signature'] = self._hmac.new(self.secret, q.encode(), self._sha.sha256).hexdigest()
        return params

    def _get(self, ep, params=None, signed=False):
        p = params or {}
        if signed: p = self._sign(p)
        try:
            r = requests.get(f"{self.base}{ep}", headers=self.headers, params=p, timeout=10)
            r.raise_for_status(); return r.json()
        except Exception as e:
            logger.error(f"Binance GET {ep}: {e}"); return {}

    def _post(self, ep, params):
        p = self._sign(params)
        try:
            r = requests.post(f"{self.base}{ep}", headers=self.headers, params=p, timeout=10)
            r.raise_for_status(); return r.json()
        except Exception as e:
            logger.error(f"Binance POST {ep}: {e}"); return {}

    def get_cuenta(self) -> dict:
        d = self._get('/v3/account', signed=True)
        return {b['asset']: float(b['free']) for b in d.get('balances', []) if float(b['free']) > 0}

    def get_precio(self, symbol: str) -> float:
        d = self._get('/v3/ticker/price', {'symbol': symbol})
        return float(d.get('price', 0)) if d else 0.0

    def enviar_orden_market(self, symbol, lado, usdt_cantidad) -> dict:
        precio = self.get_precio(symbol)
        if precio <= 0: return {}
        qty = round(usdt_cantidad / precio, 5)
        logger.info(f" BINANCE {lado} {qty:.5f} {symbol} ≈ ${usdt_cantidad:.2f} USDT")
        return self._post('/v3/order', {
            'symbol':   symbol,
            'side':     lado,
            'type':     'MARKET',
            'quantity': qty,
        })

    def get_posicion(self, symbol: str) -> float:
        asset = symbol.replace('USDT', '')
        return self.get_cuenta().get(asset, 0.0)


# ══════════════════════════════════════════════════════════════
# GESTOR DE RIESGO
# ══════════════════════════════════════════════════════════════
class GestorRiesgo:
    def __init__(self, capital_total, max_pct=0.10, stop_global=0.20, cooldown_horas=4):
        self.capital   = capital_total
        self.capital_0 = capital_total
        self.max_pct   = max_pct
        self.stop      = stop_global
        self.cooldown  = timedelta(hours=cooldown_horas)
        self.ultimo_trade = {}

    def actualizar_capital(self, c): self.capital = c

    def puede_operar(self, ticker) -> tuple:
        # Stop global del portfolio
        if self.capital_0 > 0:
            dd = (self.capital_0 - self.capital) / self.capital_0
            if dd >= self.stop:
                return False, f"Stop global activado (DD={dd:.0%})"

        # Cooldown por activo
        if ticker in self.ultimo_trade:
            rest = self.cooldown - (datetime.now() - self.ultimo_trade[ticker])
            if rest.total_seconds() > 0:
                horas = rest.total_seconds() / 3600
                return False, f"Cooldown activo ({horas:.1f}h restantes)"

        # Capital mínimo
        if self.capital < 100:
            return False, "Capital insuficiente (< $100)"

        return True, "OK"

    def calcular_size(self, ticker, precio, volatilidad=None) -> float:
        m = self.capital * self.max_pct
        if volatilidad and volatilidad > 0:
            ajuste = np.clip(0.20 / volatilidad, 0.2, 2.0)
            m *= ajuste
        return min(m, self.capital * self.max_pct)

    def registrar_trade(self, ticker):
        self.ultimo_trade[ticker] = datetime.now()


# ══════════════════════════════════════════════════════════════
# REGISTRO DE TRADES
# ══════════════════════════════════════════════════════════════
class RegistroTrades:
    def __init__(self, archivo):
        self.archivo = archivo
        self.trades = json.load(open(archivo)) if Path(archivo).exists() else []

    def _guardar(self):
        with open(self.archivo, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)

    def registrar(self, ticker, accion, cantidad, precio, monto, prob, broker='alpaca'):
        t = {
            'timestamp':   datetime.now().isoformat(),
            'ticker':      ticker,
            'accion':      accion,
            'cantidad':    cantidad,
            'precio':      precio,
            'monto':       monto,
            'prob_modelo': prob,
            'broker':      broker,
        }
        self.trades.append(t)
        self._guardar()
        logger.info(f"📋 Trade registrado: {accion} {ticker} ${monto:.2f} @ ${precio:.2f}")

    def resumen(self) -> str:
        if not self.trades: return "Sin trades registrados"
        total = len(self.trades)
        compras = sum(1 for t in self.trades if t['accion'] == 'BUY')
        ventas  = sum(1 for t in self.trades if t['accion'] == 'SELL')
        return f"Total: {total} │ Compras: {compras} │ Ventas: {ventas}"


# ══════════════════════════════════════════════════════════════
# BOT PRINCIPAL
# ══════════════════════════════════════════════════════════════
class PaperTradingBot:
    def __init__(self, config=CONFIG):
        self.cfg     = config
        self.alpaca  = AlpacaConnector(
            config['ALPACA_API_KEY'],
            config['ALPACA_SECRET_KEY'],
            config['ALPACA_BASE_URL']
        )
        self.binance = BinanceConnector(
            config['BINANCE_API_KEY'],
            config['BINANCE_SECRET_KEY'],
            testnet=config['BINANCE_TESTNET']
        )
        self.dl       = get_data_layer()
        self.riesgo   = GestorRiesgo(
            config['CAPITAL_PAPER'],
            config['MAX_POSICION_PCT'],
            cooldown_horas=config['COOLDOWN_HORAS']
        )
        self.registro       = RegistroTrades(config['TRADES_FILE'])
        self.kill           = KillSwitchBot(config['CAPITAL_PAPER'])  # ← Kill Switch corregido
        self.sistema_agentes = None

    def _init_agentes(self):
        todos = self.cfg['STOCKS_PAPER'] + self.cfg['CRYPTO_PAPER']
        self.sistema_agentes = SistemaAgentesAdaptativo(
            todos,
            capital_inicial = self.cfg['CAPITAL_PAPER'],
            kelly_fraccion  = 0.25,
            max_drawdown    = 0.20,
        )
        # IMPORTANTE: resetear el kill switch interno del SistemaAgentes
        # para que no cuente los análisis como trades
        self.sistema_agentes.kill.max_th = 999999

    def ciclo_completo(self):
        """
        Ciclo principal — se ejecuta cada N minutos.

        FLUJO:
          1. Verificar kill switch
          2. Analizar todos los activos con agentes
          3. Por cada señal activa → verificar riesgo → ejecutar orden
          4. Loggear resultado
        """
        hora = datetime.now().strftime('%H:%M:%S')
        logger.info(f"\n{'─'*50}")
        logger.info(f" CICLO — {hora}")
        logger.info(f"{'─'*50}")

        # Verificar kill switch del bot
        cuenta_alpaca = self.alpaca.get_cuenta()
        capital_actual = cuenta_alpaca.get('portfolio_val', self.cfg['CAPITAL_PAPER'])
        self.riesgo.actualizar_capital(capital_actual)

        bloqueado, razon = self.kill.verificar(capital_actual)
        if bloqueado:
            logger.warning(f" Bot bloqueado: {razon}")
            return

        # Verificar horario de mercado para stocks
        mercado_abierto = self.alpaca.mercado_abierto()
        if not mercado_abierto:
            logger.info(" Mercado de stocks cerrado — crypto sigue activo")

        # Inicializar agentes si es necesario
        if not self.sistema_agentes:
            self._init_agentes()
        n_pos = len(self.alpaca.get_posiciones())
        try:
            balances = self.binance.get_cuenta()
            n_pos += sum(1 for a, c in balances.items()
                         if a != 'USDT' and c > 0.001)
        except Exception:
            pass
        if hasattr(self.sistema_agentes, 'actualizar_posiciones'):
            self.sistema_agentes.actualizar_posiciones(n_pos)
            logger.info(f" Posiciones abiertas: {n_pos}")

        # Obtener señales de los agentes
        logger.info(" Analizando activos...")
        decisiones = self.sistema_agentes.analizar_todos(verbose=True)

        # Contadores para el resumen
        ejecutadas = 0
        rechazadas = 0
        sin_señal  = 0

        for d in decisiones:
            # Saltar esperas y vetados
            if d.vetado or d.accion == 'ESPERAR':
                sin_señal += 1
                continue

            # Filtrar por confianza mínima
            if d.confianza < self.cfg['MIN_CONFIANZA']:
                logger.debug(f"  {d.ticker}: confianza {d.confianza:.2f} < {self.cfg['MIN_CONFIANZA']} → descartado")
                rechazadas += 1
                continue

            info = normalizar_ticker(d.ticker)

            # Verificar horario para stocks
            if not info['es_crypto'] and not mercado_abierto:
                logger.info(f"  {d.ticker}: mercado cerrado — orden pendiente para apertura")
                continue

            # Verificar gestor de riesgo
            puede, razon_riesgo = self.riesgo.puede_operar(d.ticker)
            if not puede:
                logger.info(f"  {d.ticker}: {razon_riesgo}")
                rechazadas += 1
                continue

            # Ejecutar orden
            logger.info(f"  ▶ Ejecutando: {d.accion} {d.ticker} (conf={d.confianza:.2f})")
            if info['es_crypto']:
                exito = self._ejecutar_crypto(d)
            else:
                exito = self._ejecutar_stock(d)

            if exito:
                ejecutadas += 1
                self.kill.registrar_trade_ejecutado()  # ← solo si realmente se ejecutó
            else:
                rechazadas += 1

        # Resumen del ciclo
        logger.info(f"\n  RESUMEN CICLO:")
        logger.info(f"  Ejecutadas: {ejecutadas}")
        logger.info(f"  Sin señal:  {sin_señal}")
        logger.info(f"   Rechazadas: {rechazadas}")
        logger.info(f"  Kill Switch:  {self.kill.estado()}")
        logger.info(f"  Trades total: {self.registro.resumen()}")

    def _ejecutar_stock(self, d) -> bool:
        """Ejecuta una orden en Alpaca. Retorna True si fue exitosa."""
        tiene   = self.alpaca.tiene_posicion(d.ticker)
        precio  = self.alpaca.get_precio_actual(d.ticker)
        if precio == 0:
            logger.error(f"  {d.ticker}: precio 0, no se puede operar")
            return False

        # Calcular el tamaño en dinero permitido por tu Gestor de Riesgo
        monto_permitido = self.riesgo.calcular_size(d.ticker, precio)
        
        # CORRECCIÓN 1: Forzamos a que el tamaño de la orden respete estrictamente el precio actual
        qty = monto_permitido / precio
        
        # CORRECCIÓN 2: Redondear a 2 decimales máximo. 
        # Alpaca rechaza 4 decimales (422) en la mayoría de las acciones comunes.
        qty = round(qty, 2) 
        
        # Recalculamos el monto real final tras el redondeo de piezas
        monto_real = qty * precio

        if qty <= 0:
            logger.warning(f"  {d.ticker}: Cantidad calculada es 0 tras redondeo. Monto permitido insuficiente.")
            return False

        if d.accion == 'COMPRAR' and not tiene:
            # Enviamos la orden utilizando la cantidad corregida
            r = self.alpaca.enviar_bracket_order(
                d.ticker, qty, 'buy',
                self.cfg['STOP_LOSS_PCT'],
                self.cfg['TAKE_PROFIT_PCT']
            )
            if r.get('id'):
                self.riesgo.registrar_trade(d.ticker)
                self.registro.registrar(d.ticker, 'BUY', qty, precio, monto_real, d.confianza, 'alpaca')
                return True
            else:
                logger.error(f"  {d.ticker}: orden rechazada por Alpaca → {r}")
                return False

        elif d.accion == 'VENDER' and tiene:
            ok = self.alpaca.cerrar_posicion(d.ticker)
            if ok:
                self.riesgo.registrar_trade(d.ticker)
                self.registro.registrar(d.ticker, 'SELL', 0, precio, 0, d.confianza, 'alpaca')
                return True
            return False

        else:
            if d.accion == 'COMPRAR' and tiene:
                logger.info(f"  {d.ticker}: ya tengo posición, no compro de nuevo")
            elif d.accion == 'VENDER' and not tiene:
                logger.info(f"  {d.ticker}: no tengo posición para vender")
            return False

    def _ejecutar_crypto(self, d) -> bool:
        """Ejecuta una orden en Binance usando su propio capital. Retorna True si fue exitosa."""
        info     = normalizar_ticker(d.ticker)
        symbol_b = info['binance']
        if not symbol_b: return False

        precio = self.binance.get_precio(symbol_b)
        if precio <= 0:
            logger.error(f"  {d.ticker}: precio 0 en Binance")
            return False

        # 1. Obtener balance real de la cuenta o usar el configurado por si la API tarda
        cuenta = self.binance.get_cuenta()
        usdt_disponible = cuenta.get('USDT', self.cfg.get('CAPITAL_BINANCE', 1000))

        # 2. Calcular el tamaño de la posición basándonos ESTRICTAMENTE en el capital de Binance
        capital_crypto = self.cfg.get('CAPITAL_BINANCE', usdt_disponible)
        monto = capital_crypto * self.cfg['MAX_POSICION_PCT']
        
        # Nos aseguramos de no intentar gastar más del USDT que realmente hay libre en la billetera
        monto = min(monto, usdt_disponible)

        tiene = self.binance.get_posicion(symbol_b) > 0.001

        if d.accion == 'COMPRAR' and not tiene:
            if monto < 10:
                logger.warning(f"  {d.ticker}: monto ${monto:.2f} < $10 mínimo de Binance")
                return False
            
            # Redondeamos el monto a 2 decimales para evitar problemas de precisión en USDT
            monto = round(monto, 2)
            qty = round(monto / precio, 5) # Cantidad de crypto (ej: 0.0025 BTC)

            r = self.binance.enviar_orden_market(symbol_b, 'BUY', monto)
            if r.get('orderId'):
                self.riesgo.registrar_trade(d.ticker)
                self.registro.registrar(d.ticker, 'BUY', qty, precio, monto, d.confianza, 'binance')
                return True
            else:
                logger.error(f"  {d.ticker}: orden rechazada por Binance → {r}")
                return False

        elif d.accion == 'VENDER' and tiene:
            qty = self.binance.get_posicion(symbol_b)
            if qty > 0:
                # En Binance Spot, para vender usás la cantidad exacta de crypto que poseés
                r = self.binance.enviar_orden_market(symbol_b, 'SELL', qty * precio)
                if r.get('orderId'):
                    self.riesgo.registrar_trade(d.ticker)
                    self.registro.registrar(d.ticker, 'SELL', qty, precio, qty * precio, d.confianza, 'binance')
                    return True
            return False

        else:
            if d.accion == 'COMPRAR' and tiene:
                logger.info(f"  {d.ticker}: ya tengo posición en {d.ticker}")
            elif d.accion == 'VENDER' and not tiene:
                logger.info(f"  {d.ticker}: no tengo posición para vender en {d.ticker}")
            return False

    def reporte_posiciones(self):
        """Muestra el estado actual del portfolio."""
        logger.info(f"\n{'═'*50}")
        logger.info("📊 REPORTE DE POSICIONES")
        logger.info(f"{'═'*50}")

        # Alpaca
        cuenta = self.alpaca.get_cuenta()
        if cuenta:
            logger.info(f"ALPACA PAPER:")
            logger.info(f"  Portfolio: ${cuenta.get('portfolio_val',0):,.2f}")
            logger.info(f"  Cash:      ${cuenta.get('cash',0):,.2f}")
            logger.info(f"  Status:    {cuenta.get('status','?')}")
            posiciones = self.alpaca.get_posiciones()
            if posiciones:
                logger.info("  Posiciones:")
                for p in posiciones:
                    e = '🟢' if p['pnl'] >= 0 else '🔴'
                    logger.info(f"    {e} {p['symbol']}: ${p['valor']:,.2f} │ "
                                f"P&L: ${p['pnl']:+.2f} ({p['pnl_pct']:+.1f}%)")
            else:
                logger.info("  Sin posiciones abiertas")
        else:
            logger.warning("  No se pudo conectar a Alpaca")

        # Binance
        cuenta_b = self.binance.get_cuenta()
        if cuenta_b:
            logger.info(f"BINANCE TESTNET:")
            for asset, cantidad in cuenta_b.items():
                logger.info(f"  {asset}: {cantidad:.6f}")
        else:
            logger.warning("  No se pudo conectar a Binance")

        logger.info(f"\n  {self.registro.resumen()}")
        logger.info(f"  Kill Switch: {self.kill.estado()}")

    def modo_señales_solamente(self):
        """Muestra señales sin ejecutar órdenes. Para probar sin API keys."""
        logger.info("\n👁️  MODO SOLO SEÑALES — No se ejecutan órdenes")
        if not self.sistema_agentes:
            self._init_agentes()
        decisiones = self.sistema_agentes.analizar_todos(verbose=True)
        activas = [d for d in decisiones if not d.vetado and d.accion != 'ESPERAR']
        logger.info(f"\nSeñales activas (confianza > {self.cfg['MIN_CONFIANZA']}):")
        for d in activas:
            if d.confianza >= self.cfg['MIN_CONFIANZA']:
                emoji = '🟢' if d.accion == 'COMPRAR' else '🔴'
                logger.info(f"  {emoji} {d.ticker}: {d.accion} (conf={d.confianza:.2f})")
        if not activas:
            logger.info("  Sin señales activas hoy — confianza insuficiente en todos los activos")
        return decisiones

    def correr(self, intervalo_min: int = 60):
        """
        Inicia el bot en loop infinito.

        HORARIOS IMPORTANTES (hora Argentina):
          Mercado stocks USA:
            Verano:   10:30am — 5:00pm
            Invierno: 11:30am — 6:00pm
          Crypto (Binance): 24/7

        El bot analiza cada 'intervalo_min' minutos.
        Para stocks, las órdenes solo se ejecutan cuando el mercado está abierto.
        Para crypto, siempre opera.
        """
        logger.info(f"\n{'═'*55}")
        logger.info(f"🚀 PAPER TRADING BOT INICIADO")
        logger.info(f"   Intervalo análisis: cada {intervalo_min} minutos")
        logger.info(f"   Stocks: {', '.join(self.cfg['STOCKS_PAPER'])}")
        logger.info(f"   Crypto: {', '.join(self.cfg['CRYPTO_PAPER'])}")
        logger.info(f"   MIN_CONFIANZA: {self.cfg['MIN_CONFIANZA']}")
        logger.info(f"   MAX_POSICION: {self.cfg['MAX_POSICION_PCT']*100:.0f}% del capital")
        logger.info(f"   Ctrl+C para detener")
        logger.info(f"{'═'*55}\n")

        # Programar ciclos
        schedule.every(intervalo_min).minutes.do(self.ciclo_completo)
        schedule.every(4).hours.do(self.reporte_posiciones)

        # Ejecutar inmediatamente al arrancar
        self.ciclo_completo()
        self.reporte_posiciones()

        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("\n⏹️  Bot detenido por el usuario")
            self.reporte_posiciones()


# ══════════════════════════════════════════════════════════════
# ARRANCAR
# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    bot = PaperTradingBot(CONFIG)

    # ── OPCIÓN 1: Solo señales (descomentar para probar sin ejecutar) ──
    # bot.modo_señales_solamente()

    # ── OPCIÓN 2: Bot completo con ejecución automática ───────────────
    bot.correr(intervalo_min=60)
