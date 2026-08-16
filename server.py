"""
╔══════════════════════════════════════════════════════════════════╗
║   SERVIDOR WEB — FastAPI                                        ║
║   v2.0 — Usa DataLayer (Binance + CoinGecko + yfinance)         ║
╚══════════════════════════════════════════════════════════════════╝

CORRER:
  python server.py
  → http://localhost:8000
  → http://localhost:8000/docs  (API interactiva)

INSTALACIÓN:
  pip install fastapi uvicorn yfinance requests numpy scipy pandas scikit-learn
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn, json, numpy as np, pandas as pd
from datetime import datetime
from pathlib import Path
import warnings; warnings.filterwarnings('ignore')

from data_layer import get_data_layer, normalizar_ticker, TOP_10_BINANCE, TOP_10_YF

# ── App ───────────────────────────────────────────────────────
app = FastAPI(title="Trading Dashboard — Econofísica + ML", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = Path("static"); static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Modelos de request ────────────────────────────────────────
class AnalisisRequest(BaseModel):
    ticker: str
    periodo: str = "1y"

class PortafolioRequest(BaseModel):
    tickers: list[str]
    periodo: str = "1y"

# ── Cache ─────────────────────────────────────────────────────
_cache: dict = {}
CACHE_MIN = 15

def cache_ok(key: str) -> bool:
    if key not in _cache: return False
    diff = (datetime.now() - datetime.fromisoformat(_cache[key].get('timestamp','2000-01-01'))).seconds/60
    return diff < CACHE_MIN

# ── ML helpers (inline, sin imports externos complejos) ───────
def calcular_features_inline(df: pd.DataFrame) -> pd.DataFrame:
    c = df['close'].squeeze(); r = np.log(c/c.shift(1)); f = pd.DataFrame(index=df.index)
    def hv(s):
        if len(s)<20 or np.std(s)<1e-10: return 0.5
        d=np.cumsum(s-s.mean()); return float(np.clip(np.log((d.max()-d.min())/np.std(s))/np.log(len(s)),0,1))
    f['hurst']      = r.rolling(63).apply(hv, raw=True); f['fractal_dim'] = 2-f['hurst']
    v21=r.rolling(21).std()*np.sqrt(252); v63=r.rolling(63).std()*np.sqrt(252)
    f['vol_ratio']  = v21/(v63+1e-10); f['regimen_vol']=(v21>v63.rolling(63).median()).astype(int)
    f['tsallis_q']  = (1+2/(r.rolling(63).kurt().abs()+3)).clip(1.0,2.5)
    d2=c.diff(); g=d2.clip(lower=0).rolling(14).mean(); p=(-d2.clip(upper=0)).rolling(14).mean()
    f['rsi']        = 100-(100/(1+g/(p+1e-10)))
    m20=c.rolling(20).mean(); s20=c.rolling(20).std()
    f['bb_pos']     = (c-(m20-2*s20))/(4*s20+1e-10)-0.5
    for n in [5,10,21]: f[f'mom_{n}']=r.rolling(n).sum()
    f['vol_5']=r.rolling(5).std(); f['ret_lag1']=r.shift(1); f['ret_lag2']=r.shift(2)
    return f

def entrenar_modelo_inline(df: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV

    precios  = df['close'].squeeze()
    features = calcular_features_inline(df)

    labels = pd.Series(0, index=precios.index)
    p = precios.values; n = len(p); umbral = 0.025; horizonte = 5
    for i in range(n-horizonte):
        p0=p[i]
        for j in range(1,horizonte+1):
            if i+j>=n: break
            if p[i+j]>=p0*(1+umbral): labels.iloc[i]=1; break
            if p[i+j]<=p0*(1-umbral): labels.iloc[i]=-1; break

    idx = features.index.intersection(labels.index)
    X=features.loc[idx].dropna(); y=labels.loc[X.index]
    mask=~y.isna(); X,y=X[mask],y[mask]
    if len(X)<100: return None,None,None

    cols=list(X.columns); scaler=StandardScaler()
    clf=RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=20,class_weight='balanced',random_state=42,n_jobs=-1)
    modelo=CalibratedClassifierCV(clf,cv=3,method='sigmoid')
    modelo.fit(scaler.fit_transform(X),y)
    return modelo, scaler, cols

_modelos: dict = {}


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse)
async def root():
    p = static_dir/"index.html"
    return FileResponse(p) if p.exists() else HTMLResponse("<h1>Copiá index.html en static/</h1>")

@app.get("/api/health")
async def health():
    dl = get_data_layer()
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "fuentes": {
            "yfinance":   dl.yf.disponible,
            "binance":    dl.binance.disponible,
            "coingecko":  dl.coingecko.disponible,
        }
    }

@app.post("/api/analizar")
async def analizar_activo(req: AnalisisRequest):
    """Análisis completo de un activo. Funciona con BTCUSDT, BTC-USD, AAPL, etc."""
    ticker = req.ticker.upper().strip()
    key    = f"{ticker}_{req.periodo}"

    if cache_ok(key): return _cache[key]

    dl   = get_data_layer()
    info = normalizar_ticker(ticker)

    horizonte = 'corto' if req.periodo in ['1mo','3mo'] else 'mediano'
    df = dl.get_ohlcv(ticker, horizonte, req.periodo)
    if df.empty:
        raise HTTPException(404, f"Sin datos para {ticker}")

    c = df['close'].squeeze(); r = np.log(c/c.shift(1)).dropna()

    # Estadísticas
    vol_anual = float(r.std()*np.sqrt(252))
    ret_anual = float((1+r.mean())**252-1)
    sharpe    = float(r.mean()/r.std()*np.sqrt(252))
    peak      = c.cummax(); max_dd = float(((c-peak)/peak).min())
    var_95    = float(np.percentile(r,5))

    # Hurst
    H = dl._hurst(r.values)
    hurst_señal = 'PERSISTENTE' if H>0.55 else 'REVERSIVO' if H<0.45 else 'ALEATORIO'
    hurst_color = 'green' if H>0.55 else 'orange' if H<0.45 else 'gray'
    hurst_desc  = ('Tendencias duraderas — trend following' if H>0.55 else
                   'Mean-reverting — comprar caídas' if H<0.45 else 'Sin edge claro')

    # Régimen
    vol_r   = r.rolling(21).std()*np.sqrt(252)
    vol_act = float(vol_r.iloc[-1]); vol_med = float(vol_r.median())
    regimen = 'ESTRÉS' if vol_act>vol_med else 'CALMA'

    # Señal ML
    señal_ml = {'accion':'CALCULANDO','prob':0.0,'confianza':''}
    if ticker not in _modelos:
        m,s,cols = entrenar_modelo_inline(df)
        if m: _modelos[ticker] = {'modelo':m,'scaler':s,'cols':cols}

    if ticker in _modelos:
        md = _modelos[ticker]
        feat = calcular_features_inline(df)[md['cols']].iloc[-1:].fillna(0)
        pred = int(md['modelo'].predict(md['scaler'].transform(feat))[0])
        prob = md['modelo'].predict_proba(md['scaler'].transform(feat))[0]
        clases = list(md['modelo'].classes_)
        pp = float(prob[clases.index(pred)]) if pred in clases else 0.5
        señal_ml = {
            'accion': {1:'COMPRAR',-1:'VENDER',0:'ESPERAR'}[pred],
            'prob':   round(pp,3),
            'confianza': 'ALTA' if pp>0.65 else 'MEDIA' if pp>0.55 else 'BAJA',
            'color':  'green' if pred==1 else 'red' if pred==-1 else 'gray',
        }

    # Precio actual
    precio_actual = dl.get_precio_actual(ticker)

    # Extra crypto: orderbook + ballenas + Fear&Greed
    extra = {}
    if info['es_crypto']:
        if info['binance'] and dl.binance.disponible:
            ob = dl.binance.get_orderbook_ratio(info['binance'])
            extra['orderbook_ratio']  = ob
            extra['presion_orderbook']= 'COMPRADORA' if ob>1.2 else 'VENDEDORA' if ob<0.8 else 'NEUTRAL'
            bw = dl.binance.get_trades_grandes(info['binance'])
            extra['ballenas'] = bw
        if dl.coingecko.disponible:
            fg = dl.coingecko.fear_and_greed()
            extra['fear_greed'] = fg

    # Series para gráfico (últimos 180 días)
    tail = min(180, len(c))
    ph   = c.tail(tail); vh = vol_r.tail(tail); hh = r.rolling(63).apply(lambda s: dl._hurst(s.values) if len(s)>20 else 0.5, raw=False).tail(tail)
    reg_h = (vh>vol_med).astype(int)

    resultado = {
        'ticker': ticker, 'timestamp': datetime.now().isoformat(), 'periodo': req.periodo,
        'fuente': 'binance' if (info['es_crypto'] and dl.binance.disponible) else 'yfinance',
        'precio_actual': precio_actual,
        'stats': {
            'retorno_anual': round(ret_anual*100,2), 'vol_anual': round(vol_anual*100,2),
            'sharpe': round(sharpe,3), 'max_drawdown': round(max_dd*100,2),
            'var_95_diario': round(var_95*100,2), 'n_dias': len(r),
        },
        'hurst': {'valor':round(H,4),'señal':hurst_señal,'color':hurst_color,'desc':hurst_desc},
        'regimen': {'estado':regimen,'color':'red' if regimen=='ESTRÉS' else 'green',
                    'vol_actual':round(vol_act*100,2),'vol_mediana':round(vol_med*100,2)},
        'señal_ml': señal_ml,
        'extra_crypto': extra,
        'series': {
            'fechas':  [str(d.date()) for d in ph.index],
            'precios': [round(float(p),4) for p in ph.values],
            'vol':     [round(float(v)*100,2) if not np.isnan(v) else 0 for v in vh.values],
            'regimen': [int(rv) for rv in reg_h.values],
            'hurst':   [round(float(hv),3) if not np.isnan(hv) else 0.5 for hv in hh.values],
        }
    }
    _cache[key] = resultado
    return resultado


@app.get("/api/macro")
async def get_macro():
    """Datos macro del mercado crypto: Fear&Greed, dominancia BTC, top 10."""
    dl = get_data_layer()
    return dl.get_macro_crypto()


@app.get("/api/señales")
async def get_señales(tickers: str = "BTCUSDT,ETHUSDT,SOLUSDT,AAPL,NVDA,SPY"):
    """Señales rápidas para múltiples activos."""
    dl   = get_data_layer()
    lista = [t.strip().upper() for t in tickers.split(",")][:8]
    res  = []

    for ticker in lista:
        try:
            info = normalizar_ticker(ticker)
            df   = dl.get_ohlcv(ticker, 'corto', '3mo')
            if df.empty: continue

            c = df['close'].squeeze(); r = np.log(c/c.shift(1)).dropna()
            precio = dl.get_precio_actual(ticker) or float(c.iloc[-1])
            H      = dl._hurst(r.values)
            vol    = float(r.std()*np.sqrt(252)*100)
            ret_1m = float(r.tail(21).sum()*100)
            ret_1s = float(r.tail(5).sum()*100)
            v21    = r.rolling(21).std()*np.sqrt(252)
            reg    = 'ESTRÉS' if float(v21.iloc[-1])>float(v21.median()) else 'CALMA'
            señal  = ('↑' if H>0.55 and reg=='CALMA' else '↓' if reg=='ESTRÉS' else '→')

            extra = {}
            if info['es_crypto'] and info['binance'] and dl.binance.disponible:
                t24  = dl.binance.get_ticker_24h(info['binance'])
                if t24.get('cambio_24h'): ret_1m = t24['cambio_24h']
                ob   = dl.binance.get_orderbook_ratio(info['binance'])
                extra['ob_ratio'] = ob
                extra['ob_presion'] = 'COMPRADORA' if ob>1.2 else 'VENDEDORA' if ob<0.8 else 'NEUTRAL'

            res.append({'ticker':ticker,'precio':round(precio,4),'ret_1s':round(ret_1s,2),
                        'ret_1m':round(ret_1m,2),'vol_anual':round(vol,1),'hurst':round(H,3),
                        'regimen':reg,'señal':señal,'extra':extra})
        except Exception: continue

    return {'señales':res,'timestamp':datetime.now().isoformat()}


@app.post("/api/portfolio")
async def analizar_portfolio(req: PortafolioRequest):
    """Optimización de portafolio con entropía máxima."""
    from scipy.optimize import minimize as sci_min

    dl = get_data_layer()
    tickers = [t.upper().strip() for t in req.tickers]
    rets = {}

    for t in tickers:
        r = dl.get_retornos(t, 'mediano', req.periodo)
        if not r.empty: rets[t] = r

    if len(rets) < 2:
        raise HTTPException(400, "Se necesitan al menos 2 activos válidos")

    df = pd.DataFrame(rets).dropna(); n = len(df.columns)
    cov = df.cov().values; mu = df.mean().values*252

    def obj(w):
        ent = -np.sum(w*np.log(w+1e-10)); rsk = w@cov@w*252
        return -ent + 2.0*rsk

    res = sci_min(obj, np.ones(n)/n, method='SLSQP',
                   bounds=[(0.02,0.60)]*n, constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}],
                   options={'maxiter':1000})
    pesos = res.x

    corr  = df.corr()
    ret_p = float(np.sum(mu*pesos)); vol_p = float(np.sqrt(pesos@cov@pesos*252))

    return {
        'tickers': tickers, 'timestamp': datetime.now().isoformat(),
        'pesos':   {t:round(float(p),4) for t,p in zip(tickers,pesos)},
        'metricas':{'retorno_anual_est':round(ret_p*100,2),'volatilidad_anual':round(vol_p*100,2),
                    'sharpe_ratio':round(ret_p/vol_p if vol_p>0 else 0,3),
                    'entropia':round(float(-np.sum(pesos*np.log(pesos+1e-10))),3),
                    'entropia_maxima':round(float(np.log(n)),3)},
        'correlacion':{'tickers':list(corr.columns),'matrix':[[round(float(v),3) for v in row] for row in corr.values]}
    }


@app.get("/api/agentes")
async def agentes_señales(tickers: str = "BTCUSDT,ETHUSDT,AAPL,NVDA"):
    """Debate Bull/Bear/Risk para múltiples activos."""
    from sistema_agentes import SistemaAgentes
    lista = [t.strip().upper() for t in tickers.split(",")][:6]
    try:
        sistema = SistemaAgentes(lista, capital_inicial=100_000)
        decisiones = sistema.analizar_todos(verbose=False)
        return {
            'decisiones': [
                {'ticker':d.ticker,'accion':d.accion,'confianza':d.confianza,
                 'kelly':d.kelly_fraccion,'vetado':d.vetado,'timestamp':d.timestamp}
                for d in decisiones
            ],
            'macro': sistema.dl.get_macro_crypto() if any(normalizar_ticker(t)['es_crypto'] for t in lista) else {},
            'timestamp': datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/api/entrenar/{ticker}")
async def entrenar(ticker: str, background_tasks: BackgroundTasks):
    """Entrena el modelo ML en background."""
    ticker = ticker.upper()
    dl     = get_data_layer()
    def _train():
        df = dl.get_ohlcv(ticker, 'mediano', '2y')
        if not df.empty:
            m,s,cols = entrenar_modelo_inline(df)
            if m: _modelos[ticker] = {'modelo':m,'scaler':s,'cols':cols}
    background_tasks.add_task(_train)
    return {'status':'entrenando','ticker':ticker}


@app.get("/api/trades")
async def get_trades():
    p = Path("trades_log.json")
    if p.exists():
        with open(p) as f: return {"trades": json.load(f)[-50:]}
    return {"trades": []}


# ══════════════════════════════════════════════════════════════
# ARRANCAR
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*55)
    print("  TRADING DASHBOARD v2.0")
    print("  Binance REST + CoinGecko + yfinance")
    print("═"*55)
    print("  🌐 http://localhost:8000")
    print("  📡 http://localhost:8000/docs")
    print("  ⏹️  Ctrl+C para detener")
    print("═"*55 + "\n")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True, log_level="warning")
