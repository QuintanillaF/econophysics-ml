"""
╔══════════════════════════════════════════════════════════════════╗
║   DATA LAYER — CAPA UNIFICADA DE DATOS                          ║
║   Binance REST + CoinGecko + yfinance (fallback)                ║
╚══════════════════════════════════════════════════════════════════╝

LÓGICA DE SELECCIÓN:
  Crypto corto/mediano → Binance REST  (sin delay, preciso)
  Crypto largo plazo   → yfinance      (más historia)
  Stocks               → yfinance      (Binance no tiene stocks)
  Macro crypto         → CoinGecko     (Fear&Greed, dominancia)

INSTALACIÓN:
  pip install requests pandas numpy scipy yfinance
"""

import time, json, logging
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime
from typing import Optional

logger = logging.getLogger('DataLayer')
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s │ %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(h)


# ── Mapeos ────────────────────────────────────────────────────
BINANCE_A_YF = {
    'BTCUSDT':'BTC-USD','ETHUSDT':'ETH-USD','BNBUSDT':'BNB-USD',
    'SOLUSDT':'SOL-USD','XRPUSDT':'XRP-USD','DOGEUSDT':'DOGE-USD',
    'ADAUSDT':'ADA-USD','AVAXUSDT':'AVAX-USD','LINKUSDT':'LINK-USD','DOTUSDT':'DOT-USD',
}
YF_A_BINANCE   = {v: k for k, v in BINANCE_A_YF.items()}
YF_A_COINGECKO = {
    'BTC-USD':'bitcoin','ETH-USD':'ethereum','BNB-USD':'binancecoin',
    'SOL-USD':'solana','XRP-USD':'ripple','DOGE-USD':'dogecoin',
    'ADA-USD':'cardano','AVAX-USD':'avalanche-2','LINK-USD':'chainlink','DOT-USD':'polkadot',
}
TOP_10_BINANCE = list(BINANCE_A_YF.keys())
TOP_10_YF      = list(BINANCE_A_YF.values())

# ── Watchlist: 20 acciones + 20 criptos que analiza el dashboard ──
WATCHLIST_ACCIONES = [
    'AAPL', 'NVDA', 'MSFT', 'GOOGL', 'AMZN', 'META', 'AVGO', 'TSLA', 'BRK-B', 'LLY',
    'JPM', 'V', 'WMT', 'XOM', 'UNH', 'MA', 'COST', 'AMD', 'NFLX', 'PLTR',
]
WATCHLIST_CRYPTO = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD',
    'AVAX-USD', 'LINK-USD', 'DOT-USD', 'TRX-USD', 'TON-USD', 'SHIB-USD', 'LTC-USD',
    'BCH-USD', 'ATOM-USD', 'XLM-USD', 'NEAR-USD', 'HBAR-USD', 'ETC-USD',
]
# Intercalado 1:1 para que ambos tipos aparezcan al scrollear la barra de señales.
WATCHLIST = [t for par in zip(WATCHLIST_ACCIONES, WATCHLIST_CRYPTO) for t in par]


def normalizar_ticker(ticker: str) -> dict:
    t = ticker.upper().strip()
    if t in BINANCE_A_YF:
        yf = BINANCE_A_YF[t]
        return {'original':ticker,'binance':t,'yfinance':yf,'coingecko':YF_A_COINGECKO.get(yf,''),'es_crypto':True}
    if t in YF_A_BINANCE:
        return {'original':ticker,'binance':YF_A_BINANCE[t],'yfinance':t,'coingecko':YF_A_COINGECKO.get(t,''),'es_crypto':True}
    return {'original':ticker,'binance':None,'yfinance':t,'coingecko':None,'es_crypto':False}


# ── Fuente yfinance ───────────────────────────────────────────
class FuenteYFinance:
    def __init__(self):
        try:
            import yfinance as yf
            self.yf = yf; self.disponible = True
        except ImportError:
            self.disponible = False

    def get_ohlcv(self, ticker: str, periodo: str = '1y') -> pd.DataFrame:
        if not self.disponible: return pd.DataFrame()
        try:
            df = self.yf.download(ticker, period=periodo, auto_adjust=True, progress=False)
            if df.empty: return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            cols = [c for c in ['open','high','low','close','volume'] if c in df.columns]
            return df[cols]
        except Exception as e:
            logger.error(f"yfinance {ticker}: {e}"); return pd.DataFrame()


# ── Fuente Binance REST ───────────────────────────────────────
class FuenteBinance:
    BASE  = 'https://api.binance.com/api/v3'
    FBASE = 'https://fapi.binance.com'          # mercado de futuros perpetuos

    def __init__(self):
        import requests
        self.session = requests.Session()
        self.disponible = self._ping()

    def _ping(self) -> bool:
        try:
            import requests
            return requests.get(f"{self.BASE}/ping", timeout=5).status_code == 200
        except: return False

    def _get(self, ep: str, params: dict = None):
        try:
            r = self.session.get(f"{self.BASE}{ep}", params=params or {}, timeout=15)
            r.raise_for_status(); return r.json()
        except Exception as e:
            logger.debug(f"Binance {ep}: {e}"); return None

    def _fget(self, ep: str, params: dict = None):
        try:
            r = self.session.get(f"{self.FBASE}{ep}", params=params or {}, timeout=15)
            r.raise_for_status(); return r.json()
        except Exception as e:
            logger.debug(f"Binance futures {ep}: {e}"); return None

    def get_klines(self, symbol: str, interval: str = '1d', limit: int = 500) -> pd.DataFrame:
        data = self._get('/klines', {'symbol':symbol,'interval':interval,'limit':min(limit,1000)})
        if not data: return pd.DataFrame()
        try:
            df = pd.DataFrame(data, columns=['ts','open','high','low','close','volume','ct','qv','nt','tbv','tbqv','ignore'])
            df['ts'] = pd.to_datetime(df['ts'], unit='ms')
            df.set_index('ts', inplace=True); df.index.name = 'timestamp'
            for c in ['open','high','low','close','volume']: df[c] = df[c].astype(float)
            return df[['open','high','low','close','volume']]
        except: return pd.DataFrame()

    def get_precio(self, symbol: str) -> float:
        d = self._get('/ticker/price', {'symbol':symbol})
        return float(d.get('price',0)) if d else 0.0

    def get_ticker_24h(self, symbol: str) -> dict:
        d = self._get('/ticker/24hr', {'symbol':symbol})
        if not d: return {}
        return {'precio':float(d.get('lastPrice',0)),'cambio_24h':float(d.get('priceChangePercent',0)),
                'high':float(d.get('highPrice',0)),'low':float(d.get('lowPrice',0)),'volumen':float(d.get('quoteVolume',0))}

    def get_orderbook_ratio(self, symbol: str) -> float:
        d = self._get('/depth', {'symbol':symbol,'limit':20})
        if not d: return 1.0
        try:
            bids = sum(float(p)*float(q) for p,q in d.get('bids',[]))
            asks = sum(float(p)*float(q) for p,q in d.get('asks',[]))
            return round(bids/(asks+1e-10),3)
        except: return 1.0

    def get_trades_grandes(self, symbol: str, umbral: float = 100_000) -> dict:
        data = self._get('/trades', {'symbol':symbol,'limit':100})
        if not isinstance(data, list): return {'señal':'NEUTRAL','detalle':'Sin datos','n':0}
        ballenas = [{'valor':float(t['price'])*float(t['qty']),'compra':not t['isBuyerMaker']}
                    for t in data if float(t['price'])*float(t['qty']) >= umbral]
        if not ballenas: return {'señal':'NEUTRAL','detalle':f'Sin trades > ${umbral:,.0f}','n':0}
        vc = sum(b['valor'] for b in ballenas if b['compra'])
        vv = sum(b['valor'] for b in ballenas if not b['compra'])
        r  = vc/(vv+1e-10)
        señal = 'ALCISTA' if r>1.5 else 'BAJISTA' if r<0.67 else 'NEUTRAL'
        return {'señal':señal,'detalle':f'Compras=${vc:,.0f} Ventas=${vv:,.0f} ({len(ballenas)} ballenas)','n':len(ballenas),'ratio':round(r,2)}

    # ── Microestructura de futuros perpetuos ─────────────────────
    def get_funding_rate(self, symbol: str) -> dict:
        """Funding rate actual + media de los últimos 7 días (anualizada).

        El funding es lo que los longs pagan a los shorts (o viceversa) cada 8h en
        un perpetuo. Positivo y alto = demanda de apalancamiento long saturada →
        señal contrarian bajista clásica.
        """
        pi = self._fget('/fapi/v1/premiumIndex', {'symbol': symbol})
        hist = self._fget('/fapi/v1/fundingRate', {'symbol': symbol, 'limit': 21})  # ~7 días
        if not pi:
            return {}
        actual = float(pi.get('lastFundingRate', 0))
        rates = [float(h['fundingRate']) for h in hist] if isinstance(hist, list) else []
        media_7d = float(np.mean(rates)) if rates else actual
        return {
            'actual': round(actual, 6),
            'media_7d': round(media_7d, 6),
            'anualizado_pct': round(media_7d * 3 * 365 * 100, 2),  # 3 pagos/día
            'proximo': pi.get('nextFundingTime'),
            'señal': ('LONGS SATURADOS' if media_7d > 0.0005 else
                      'SHORTS SATURADOS' if media_7d < -0.0003 else 'NEUTRAL'),
        }

    def get_basis(self, symbol: str) -> dict:
        """Basis del perpetuo: (mark - index) / index.

        Positivo = el perpetuo cotiza sobre el spot (contango, apalancamiento
        long); negativo = backwardation (presión short / miedo).
        """
        pi = self._fget('/fapi/v1/premiumIndex', {'symbol': symbol})
        if not pi:
            return {}
        mark, index = float(pi.get('markPrice', 0)), float(pi.get('indexPrice', 0))
        if index <= 0:
            return {}
        b = (mark - index) / index
        return {
            'mark': mark, 'index': index,
            'basis_pct': round(b * 100, 4),
            'estado': 'CONTANGO' if b > 0 else 'BACKWARDATION',
        }

    def get_open_interest(self, symbol: str) -> dict:
        """Open interest actual + cambio a 7 días.

        OI subiendo con precio subiendo = dinero nuevo apalancado entrando; OI
        subiendo con precio cayendo = shorts agresivos. OI cayendo fuerte =
        liquidaciones / deleveraging.
        """
        cur = self._fget('/fapi/v1/openInterest', {'symbol': symbol})
        hist = self._fget('/futures/data/openInterestHist',
                          {'symbol': symbol, 'period': '1d', 'limit': 8})
        if not cur:
            return {}
        oi_now = float(cur.get('openInterest', 0))
        cambio_7d = None
        if isinstance(hist, list) and len(hist) >= 2:
            oi_prev = float(hist[0].get('sumOpenInterest', 0))
            if oi_prev > 0:
                cambio_7d = round((oi_now / oi_prev - 1) * 100, 2)
        return {'contratos': oi_now, 'cambio_7d_pct': cambio_7d}

    def get_long_short_ratio(self, symbol: str) -> dict:
        """Ratio cuentas long / short (todas las cuentas de Binance Futures)."""
        d = self._fget('/futures/data/globalLongShortAccountRatio',
                       {'symbol': symbol, 'period': '1d', 'limit': 1})
        if not isinstance(d, list) or not d:
            return {}
        e = d[-1]
        ratio = float(e.get('longShortRatio', 1))
        return {
            'ratio': round(ratio, 3),
            'long_pct': round(float(e.get('longAccount', 0)) * 100, 1),
            'short_pct': round(float(e.get('shortAccount', 0)) * 100, 1),
            'sesgo': 'LARGOS' if ratio > 1.5 else 'CORTOS' if ratio < 0.8 else 'EQUILIBRADO',
        }

    def get_microestructura_cripto(self, symbol: str) -> dict:
        """Bundle: funding + basis + open interest + long/short, con una lectura."""
        funding = self.get_funding_rate(symbol)
        basis   = self.get_basis(symbol)
        oi      = self.get_open_interest(symbol)
        ls      = self.get_long_short_ratio(symbol)
        if not funding and not basis:
            return {'disponible': False}

        # Lectura combinada.
        f_alto = funding.get('media_7d', 0) > 0.0005
        oi_sube = (oi.get('cambio_7d_pct') or 0) > 5
        if f_alto and oi_sube:
            lectura = 'Funding alto + OI subiendo: trade long apalancado saturado → riesgo de long squeeze'
        elif funding.get('media_7d', 0) < -0.0003:
            lectura = 'Funding negativo: shorts pagando → posible short squeeze si el precio gira'
        elif (oi.get('cambio_7d_pct') or 0) < -10:
            lectura = 'OI cayendo fuerte: deleveraging / liquidaciones en curso'
        else:
            lectura = 'Microestructura sin extremos'

        return {
            'disponible': True, 'symbol': symbol,
            'funding': funding, 'basis': basis,
            'open_interest': oi, 'long_short': ls,
            'lectura': lectura,
        }


# ── Fuente CoinGecko ──────────────────────────────────────────
class FuenteCoinGecko:
    BASE = 'https://api.coingecko.com/api/v3'
    FNG  = 'https://api.alternative.me/fng/?limit=1'

    def __init__(self):
        import requests
        self.session = requests.Session()
        self._cache = {}; self._ttl = 300
        self.disponible = self._ping()

    def _ping(self) -> bool:
        try:
            import requests
            return requests.get(f"{self.BASE}/ping", timeout=5).status_code == 200
        except: return False

    def _get(self, ep: str, params: dict = None):
        key = f"{ep}_{json.dumps(params or {}, sort_keys=True)}"
        if key in self._cache:
            ts, d = self._cache[key]
            if time.time()-ts < self._ttl: return d
        try:
            r = self.session.get(f"{self.BASE}{ep}", params=params or {}, timeout=15)
            r.raise_for_status(); d = r.json()
            self._cache[key] = (time.time(), d); return d
        except Exception as e:
            logger.debug(f"CoinGecko {ep}: {e}"); return None

    def fear_and_greed(self) -> dict:
        try:
            import requests
            d = requests.get(self.FNG, timeout=10).json()['data'][0]
            v = int(d['value'])
            return {'valor':v,'nombre':d['value_classification'],
                    'señal':('COMPRAR' if v<25 else 'ALCISTA' if v<45 else 'NEUTRAL' if v<55 else 'BAJISTA' if v<75 else 'VENDER')}
        except: return {'valor':50,'nombre':'Neutral','señal':'NEUTRAL'}

    def global_data(self) -> dict:
        d = self._get('/global')
        if not d: return {}
        g = d.get('data',{}); dom = g.get('market_cap_percentage',{})
        return {'dominancia_btc':round(dom.get('btc',50),2),'dominancia_eth':round(dom.get('eth',20),2),
                'market_cap_usd':g.get('total_market_cap',{}).get('usd',0),'volumen_24h':g.get('total_volume',{}).get('usd',0)}

    def top10(self) -> list:
        ids  = ','.join(YF_A_COINGECKO.values())
        data = self._get('/coins/markets',{'vs_currency':'usd','ids':ids,'order':'market_cap_desc',
                          'per_page':10,'price_change_percentage':'1h,24h,7d'})
        if not isinstance(data, list): return []
        return [{'nombre':c.get('name',''),'symbol':c.get('symbol','').upper(),'precio':c.get('current_price',0),
                 'ranking':c.get('market_cap_rank',0),'cambio_1h':c.get('price_change_percentage_1h_in_currency') or 0,
                 'cambio_24h':c.get('price_change_percentage_24h') or 0,'cambio_7d':c.get('price_change_percentage_7d_in_currency') or 0,
                 'vol_24h':c.get('total_volume',0),'market_cap':c.get('market_cap',0)} for c in data]


# ══════════════════════════════════════════════════════════════
# DATA LAYER — INTERFAZ PRINCIPAL
# ══════════════════════════════════════════════════════════════
class DataLayer:
    _TTL = {'corto':60,'mediano':300,'largo':3600}

    def __init__(self):
        self.yf        = FuenteYFinance()
        self.binance   = FuenteBinance()
        self.coingecko = FuenteCoinGecko()
        self._cache: dict = {}
        self._iniciado = False

    def iniciar(self):
        if self._iniciado: return
        fuentes = ([f for f,ok in [('yfinance',self.yf.disponible),
                   ('Binance REST',self.binance.disponible),
                   ('CoinGecko',self.coingecko.disponible)] if ok])
        logger.info(f"DataLayer listo → {', '.join(fuentes)}")
        self._iniciado = True

    def _ok(self, k, h): return k in self._cache and (time.time()-self._cache[k][0]) < self._TTL.get(h,300)
    def _set(self, k, d): self._cache[k] = (time.time(), d)

    def get_ohlcv(self, ticker: str, horizonte: str = 'mediano', periodo: str = None) -> pd.DataFrame:
        info = normalizar_ticker(ticker)
        key  = f"ohlcv_{ticker}_{horizonte}_{periodo}"
        if self._ok(key, horizonte): return self._cache[key][1]

        p  = periodo or {'corto':'3mo','mediano':'1y','largo':'5y'}.get(horizonte,'1y')
        df = pd.DataFrame()

        if info['es_crypto']:
            if horizonte == 'largo':
                df = self.yf.get_ohlcv(info['yfinance'], p)
            elif self.binance.disponible and info['binance']:
                if horizonte == 'corto':
                    dias = {'1mo':30,'3mo':90,'6mo':180,'1y':365}.get(p,90)
                    raw  = self.binance.get_klines(info['binance'], '1h', min(dias*24,1000))
                    if not raw.empty:
                        df = raw.resample('D').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
                else:
                    lim = {'1y':365,'2y':730,'3y':1000,'5y':1000}.get(p,365)
                    df  = self.binance.get_klines(info['binance'], '1d', lim)
            if df.empty:
                df = self.yf.get_ohlcv(info['yfinance'], p)
        else:
            df = self.yf.get_ohlcv(info['yfinance'], p)

        if not df.empty: self._set(key, df)
        return df

    def get_retornos(self, ticker: str, horizonte: str = 'mediano', periodo: str = None) -> pd.Series:
        df = self.get_ohlcv(ticker, horizonte, periodo)
        if df.empty: return pd.Series(dtype=float)
        c = df['close'].squeeze()
        return np.log(c/c.shift(1)).dropna()

    def get_precio_actual(self, ticker: str) -> float:
        info = normalizar_ticker(ticker)
        if info['binance'] and self.binance.disponible:
            p = self.binance.get_precio(info['binance'])
            if p > 0: return p
        if self.yf.disponible:
            try:
                import yfinance as yf
                return float(yf.Ticker(info['yfinance']).info.get('regularMarketPrice') or 0)
            except: pass
        return 0.0

    def get_macro_crypto(self) -> dict:
        key = 'macro'
        if self._ok(key,'mediano'): return self._cache[key][1]
        fg = self.coingecko.fear_and_greed()
        gl = self.coingecko.global_data()
        t10 = self.coingecko.top10()
        n_sub = sum(1 for c in t10 if c.get('cambio_24h',0) > 0)
        d = {'fear_greed':fg,'dominancia_btc':gl.get('dominancia_btc',50),'dominancia_eth':gl.get('dominancia_eth',20),
             'market_cap_usd':gl.get('market_cap_usd',0),'volumen_24h':gl.get('volumen_24h',0),
             'top10_subiendo':n_sub,'top10_bajando':10-n_sub,
             'sentimiento':('ALCISTA' if n_sub>7 else 'BAJISTA' if n_sub<3 else 'MIXTO'),
             'top10':t10,'timestamp':datetime.now().isoformat()}
        self._set(key, d); return d

    def get_contexto_agentes(self, ticker: str) -> dict:
        info = normalizar_ticker(ticker)
        df = self.get_ohlcv(ticker, 'corto', '3mo')
        if df.empty: df = self.get_ohlcv(ticker, 'mediano', '6mo')
        if df.empty: return {}

        c = df['close'].squeeze(); r = np.log(c/c.shift(1)).dropna()
        precio = self.get_precio_actual(ticker) or float(c.iloc[-1])
        ret_24h = float(r.tail(1).sum()*100); ret_7d = float(r.tail(5).sum()*100)
        vol  = float(r.std()*np.sqrt(252)*100)
        H    = self._hurst(r.values)
        v21  = r.rolling(21).std()*np.sqrt(252)
        reg  = 'ESTRÉS' if float(v21.iloc[-1])>float(v21.median()) else 'CALMA'
        d2=c.diff(); g=d2.clip(lower=0).rolling(14).mean(); p2=(-d2.clip(upper=0)).rolling(14).mean()
        rsi  = float(100-(100/(1+float(g.iloc[-1])/(float(p2.iloc[-1])+1e-10))))
        m20=c.rolling(20).mean(); s20=c.rolling(20).std()
        bb   = float((c.iloc[-1]-float(m20.iloc[-1]-2*s20.iloc[-1]))/(4*float(s20.iloc[-1])+1e-10)-0.5)
        vr   = 1.0
        if 'volume' in df.columns:
            vm = df['volume'].rolling(20).mean().iloc[-1]
            if vm > 0: vr = float(df['volume'].iloc[-1]/vm)

        ws, wd = 'NEUTRAL','Sin datos de ballenas'
        if info['es_crypto'] and info['binance'] and self.binance.disponible:
            t24 = self.binance.get_ticker_24h(info['binance'])
            if t24.get('cambio_24h'): ret_24h = t24['cambio_24h']
            ob  = self.binance.get_orderbook_ratio(info['binance'])
            ws  = 'COMPRADORA' if ob>1.2 else 'VENDEDORA' if ob<0.8 else 'NEUTRAL'
            wd  = f"Orderbook ratio={ob:.2f}"
            bw  = self.binance.get_trades_grandes(info['binance'])
            if bw.get('n',0)>0: ws=bw['señal']; wd=bw['detalle']
            if self.coingecko.disponible:
                fg=self.coingecko.fear_and_greed(); wd+=f" │ F&G={fg.get('valor',50)} ({fg.get('nombre','')})"

        return {'ticker':ticker,'precio_actual':precio,'retorno_1h':ret_24h/24,'retorno_24h':ret_24h,
                'retorno_7d':ret_7d,'volatilidad':vol,'hurst':H,'regimen':reg,'rsi':rsi,
                'bb_posicion':bb,'volumen_ratio':vr,'señal_ml':0,'prob_ml':0.5,'whales_señal':ws,'whales_detalle':wd}

    def _hurst(self, s: np.ndarray) -> float:
        N = len(s)
        if N < 20: return 0.5
        try:
            from scipy import stats
            esc = [max(10,N//8),max(20,N//4),max(30,N//2)]
            rv,ev = [],[]
            for n in esc:
                if n>=N: continue
                rs = []
                for i in range(N//n):
                    b=s[i*n:(i+1)*n]; std=np.std(b,ddof=1)
                    if std>1e-10:
                        d=np.cumsum(b-b.mean()); rs.append((d.max()-d.min())/std)
                if rs: rv.append(np.mean(rs)); ev.append(n)
            if len(ev)>=2:
                H,*_ = stats.linregress(np.log(ev),np.log(rv)); return float(np.clip(H,0,1))
        except: pass
        return 0.5


# Singleton
_instance: Optional[DataLayer] = None
def get_data_layer() -> DataLayer:
    global _instance
    if _instance is None:
        _instance = DataLayer(); _instance.iniciar()
    return _instance


if __name__ == '__main__':
    print('\n'+'═'*50); print('  TEST DATA LAYER'); print('═'*50)
    dl = get_data_layer()
    for t,h,p in [('BTCUSDT','corto','3mo'),('ETH-USD','mediano','1y'),('AAPL','largo','5y')]:
        df = dl.get_ohlcv(t,h,p); precio = dl.get_precio_actual(t)
        print(f"  {t:<12} {h:<10} {p:<6} → {len(df):>4} filas  ${precio:,.2f}")
    m = dl.get_macro_crypto(); fg = m['fear_greed']
    print(f"\n  Fear&Greed: {fg['valor']} ({fg['nombre']}) → {fg['señal']}")
    print(f"  BTC Dom: {m['dominancia_btc']}%  Sentimiento: {m['sentimiento']}\n")
