"""
╔══════════════════════════════════════════════════════════════════╗
║   ECONOFÍSICA — ANÁLISIS CORTO PLAZO                            ║
║   v2.0 — Usa DataLayer (Binance + CoinGecko + yfinance)         ║
╚══════════════════════════════════════════════════════════════════╝
INSTALACIÓN: pip install yfinance numpy scipy pandas matplotlib seaborn requests
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings('ignore')
from scipy import stats
from scipy.optimize import minimize, curve_fit
from scipy.stats import norm
from data_layer import get_data_layer, normalizar_ticker, TOP_10_BINANCE, TOP_10_YF


class AnalisisHurst:
    def __init__(self, retornos):
        self.retornos = np.array(retornos); self.H = None
        self.escalas = None; self.rs_vals = None

    def calcular(self, n_escalas=20):
        serie = self.retornos; N = len(serie)
        min_e = max(10, int(N*0.02)); max_e = int(N/4)
        escalas = np.unique(np.logspace(np.log10(min_e), np.log10(max_e), n_escalas).astype(int))
        rs_p = []
        for n in escalas:
            nb = N//n; rs_b = []
            for i in range(nb):
                b = serie[i*n:(i+1)*n]; std = np.std(b, ddof=1)
                if std < 1e-10: continue
                d = np.cumsum(b - b.mean()); rs_b.append((d.max()-d.min())/std)
            if rs_b: rs_p.append(np.mean(rs_b))
        self.escalas = escalas[:len(rs_p)]; self.rs_vals = np.array(rs_p)
        self.H, *_ = stats.linregress(np.log(self.escalas), np.log(self.rs_vals))
        return self.H

    def interpretar(self):
        if self.H is None: return "Ejecuta calcular() primero"
        h = self.H
        if h < 0.4:   return f"H={h:.3f} │ ANTI-PERSISTENTE — Mean-reverting fuerte"
        elif h < 0.45: return f"H={h:.3f} │ Leve anti-persistencia"
        elif h < 0.55: return f"H={h:.3f} │ ALEATORIO — Sin edge claro"
        elif h < 0.65: return f"H={h:.3f} │ PERSISTENTE — Trend following viable"
        else:          return f"H={h:.3f} │ MUY PERSISTENTE — Momentum fuerte"


class DetectorRegimen:
    def __init__(self, retornos, ventana=21):
        self.retornos = retornos; self.ventana = ventana
        self.regimenes = None; self.vol_rolling = None

    def detectar(self):
        self.vol_rolling = self.retornos.rolling(self.ventana).std().dropna() * np.sqrt(252)
        umbral = self.vol_rolling.median()
        self.regimenes = (self.vol_rolling > umbral).astype(int)
        return self.regimenes

    def regimen_actual(self):
        if self.regimenes is None: self.detectar()
        reg = int(self.regimenes.iloc[-1]); vol = float(self.vol_rolling.iloc[-1])
        return {'regimen':reg,'nombre':'ESTRÉS' if reg==1 else 'CALMA',
                'volatilidad_actual':vol,'umbral_historico':float(self.vol_rolling.median()),
                'señal':'⚠️ Reducir exposición' if reg==1 else '✅ Condiciones normales'}

    def calcular_var(self, confianza=0.95):
        alpha = 1-confianza; r = self.retornos
        return {'VaR_historico':float(np.percentile(r, alpha*100)),
                'VaR_parametrico':float(norm.ppf(alpha, r.mean(), r.std())),
                'interpretacion':f"Pérdida máx diaria {confianza*100:.0f}%: {abs(np.percentile(r,alpha*100))*100:.2f}%"}


class AnalisisTsallis:
    def __init__(self, retornos):
        self.retornos = np.array(retornos); self.q = None; self.beta = None

    def _q_gaussiana(self, x, q, beta):
        arg = np.maximum(1-(1-q)*beta*x**2, 1e-300)
        return arg**(1/(1-q))

    def ajustar(self):
        r = self.retornos; counts, bins = np.histogram(r, bins=100, density=True)
        xc = 0.5*(bins[:-1]+bins[1:]); mask = counts>0
        try:
            popt, _ = curve_fit(self._q_gaussiana, xc[mask], counts[mask],
                                p0=[1.4, 1.0/np.var(r)], bounds=([1.0,1e-6],[3.0,1e6]), maxfev=10000)
            self.q, self.beta = popt
        except:
            self.q = float(np.clip(1.0+(r.kurt()/(r.kurt()+3)), 1.0, 2.5)); self.beta = 1.0/np.var(r)
        return {'q':float(self.q),'beta':float(self.beta),'tipo':self._tipo()}

    def _tipo(self):
        if self.q < 1.1:   return "Casi gaussiano"
        elif self.q < 1.3: return "Leve cola gruesa (stock maduro)"
        elif self.q < 1.5: return "Cola gruesa moderada"
        elif self.q < 1.7: return "Cola gruesa fuerte (crypto)"
        else:              return "Extremo — muy alta incertidumbre"


class OptimizadorPortafolio:
    def __init__(self, retornos_df):
        self.retornos = retornos_df.dropna()
        self.n = len(retornos_df.columns); self.tickers = list(retornos_df.columns)
        self.pesos_optimos = None

    def optimizar(self, lambda_riesgo=1.0):
        cov = self.retornos.cov().values; n = self.n
        def objetivo(w):
            return -(-np.sum(w*np.log(w+1e-10))) + lambda_riesgo*(w@cov@w)
        res = minimize(objetivo, np.ones(n)/n, method='SLSQP',
                       bounds=[(0.01,0.60)]*n, constraints=[{'type':'eq','fun':lambda w:np.sum(w)-1}],
                       options={'maxiter':1000})
        self.pesos_optimos = res.x
        rp = float(np.sum(self.retornos.mean().values*self.pesos_optimos*252))
        vp = float(np.sqrt(self.pesos_optimos@cov@self.pesos_optimos*252))
        return {'pesos':dict(zip(self.tickers,self.pesos_optimos)),'retorno_anual':rp,
                'volatilidad_anual':vp,'sharpe_ratio':rp/vp if vp>0 else 0}


class Dashboard:
    def __init__(self, ticker: str):
        self.ticker = ticker; self.dl = get_data_layer()
        self.hurst = None; self.regimen = None; self.tsallis = None
        self.precios = None; self.retornos = None

    def analizar(self, periodo='1y'):
        info = normalizar_ticker(self.ticker)
        horizonte = 'corto' if periodo in ['1mo','3mo'] else 'mediano'
        df = self.dl.get_ohlcv(self.ticker, horizonte, periodo)
        if df.empty: print(f"Sin datos para {self.ticker}"); return self
        self.precios  = df['close'].squeeze()
        self.retornos = np.log(self.precios/self.precios.shift(1)).dropna()
        self.hurst   = AnalisisHurst(self.retornos); self.hurst.calcular()
        self.regimen = DetectorRegimen(self.retornos); self.regimen.detectar()
        self.tsallis = AnalisisTsallis(self.retornos); self.tsallis.ajustar()
        fuente = 'Binance REST' if (info['es_crypto'] and self.dl.binance.disponible) else 'yfinance'
        print(f"✓ {self.ticker} analizado ({len(self.retornos)} días, fuente: {fuente})")
        return self

    def imprimir_reporte(self):
        if self.retornos is None: print("Ejecutá analizar() primero"); return
        r = self.retornos; sep = "─"*55
        print(f"\n{'═'*55}\n  REPORTE: {self.ticker}\n{'═'*55}")

        # Stats
        vol = float(r.std()*np.sqrt(252)); sharpe = float(r.mean()/r.std()*np.sqrt(252))
        peak = self.precios.cummax(); dd = float(((self.precios-peak)/peak).min())
        print(f"\n📊 ESTADÍSTICAS\n{sep}")
        print(f"  Volatilidad anual: {vol*100:.1f}%  │  Sharpe: {sharpe:.2f}  │  Max DD: {dd*100:.1f}%")
        print(f"  Asimetría: {r.skew():.3f}  │  Curtosis: {r.kurt():.3f}")

        # Hurst
        print(f"\n🌊 HURST\n{sep}")
        print(f"  {self.hurst.interpretar()}")

        # Régimen
        reg = self.regimen.regimen_actual(); var = self.regimen.calcular_var()
        print(f"\n🔄 RÉGIMEN\n{sep}")
        print(f"  {reg['nombre']} {reg['señal']}")
        print(f"  Vol: {reg['volatilidad_actual']*100:.1f}% (umbral: {reg['umbral_historico']*100:.1f}%)")
        print(f"  {var['interpretacion']}")

        # Tsallis
        print(f"\n⚛️  TSALLIS\n{sep}")
        print(f"  q={self.tsallis.q:.3f} │ {self.tsallis._tipo()}")

        # Señal integrada
        print(f"\n🎯 SEÑAL\n{sep}")
        pts = 0
        if self.hurst.H > 0.55: pts+=1; print("  ✅ Hurst persistente")
        elif self.hurst.H < 0.45: pts-=1; print("  ⚠️  Hurst reversivo")
        if reg['regimen']==0: pts+=1; print("  ✅ Régimen calmo")
        else: pts-=1; print("  ⚠️  Régimen estrés")
        if sharpe > 0.5: pts+=1; print("  ✅ Sharpe positivo")
        print(f"\n  ▶ {'ALCISTA' if pts>=2 else 'BAJISTA/CAUTELA' if pts<=-1 else 'NEUTRAL'} ({pts}/3)")

        # Extra crypto
        info = normalizar_ticker(self.ticker)
        if info['es_crypto'] and self.dl.binance.disponible:
            print(f"\n🔗 DATOS CRYPTO EN TIEMPO REAL\n{sep}")
            precio = self.dl.get_precio_actual(self.ticker)
            if precio > 0: print(f"  Precio actual: ${precio:,.4f}")
            if info['binance']:
                ob = self.dl.binance.get_orderbook_ratio(info['binance'])
                presion = 'COMPRADORA' if ob>1.2 else 'VENDEDORA' if ob<0.8 else 'NEUTRAL'
                print(f"  Orderbook: {presion} (ratio={ob:.2f})")
            if self.dl.coingecko.disponible:
                fg = self.dl.coingecko.fear_and_greed()
                print(f"  Fear & Greed: {fg['valor']} ({fg['nombre']}) → {fg['señal']}")
        print(f"{'═'*55}\n")

    def graficar(self):
        if self.retornos is None: return
        r = self.retornos; p = self.precios
        fig = plt.figure(figsize=(16,10)); fig.patch.set_facecolor('#0d1117')
        gs  = gridspec.GridSpec(2,2,figure=fig,hspace=0.4,wspace=0.35)
        BG='#161b22'; AZ='#58a6ff'; RJ='#f78166'; GR='#8b949e'

        def estilo(ax,t):
            ax.set_facecolor(BG); ax.set_title(t,color='white',fontsize=11,pad=8)
            ax.tick_params(colors=GR)
            for sp in ax.spines.values(): sp.set_edgecolor('#30363d')

        ax1=fig.add_subplot(gs[0,0]); estilo(ax1,f'{self.ticker} — Precio & Régimen')
        vr = self.regimen.vol_rolling.reindex(p.index).fillna(0)
        med = float(vr.median())
        for i in range(len(p)-1):
            c = '#f7826622' if float(vr.iloc[i])>med else '#58a6ff11'
            ax1.axvspan(p.index[i],p.index[i+1],alpha=0.3,color=c,linewidth=0)
        ax1.plot(p.index,p.values,color=AZ,linewidth=1.2)

        ax2=fig.add_subplot(gs[0,1]); estilo(ax2,'R/S Analysis — Hurst')
        if self.hurst.escalas is not None:
            ax2.scatter(np.log(self.hurst.escalas),np.log(self.hurst.rs_vals),color=AZ,s=20,alpha=0.8)
            xl=np.linspace(np.log(self.hurst.escalas).min(),np.log(self.hurst.escalas).max(),100)
            ax2.plot(xl,self.hurst.H*xl+np.log(self.hurst.rs_vals[0]),'--',color=RJ,linewidth=1.5,label=f'H={self.hurst.H:.3f}')
            ax2.legend(facecolor=BG,labelcolor='white',fontsize=9)

        ax3=fig.add_subplot(gs[1,0]); estilo(ax3,'Distribución de Retornos')
        ra=np.array(r); ax3.hist(ra,bins=80,density=True,alpha=0.5,color=AZ,label='Retornos reales')
        xr=np.linspace(ra.min(),ra.max(),300)
        ax3.plot(xr,norm.pdf(xr,ra.mean(),ra.std()),'w--',linewidth=1.5,alpha=0.7,label='Normal')
        if self.tsallis.q is not None:
            yt=self.tsallis._q_gaussiana(xr-ra.mean(),self.tsallis.q,self.tsallis.beta)
            dx=xr[1]-xr[0]; yt=yt/(np.sum(yt)*dx+1e-10)
            ax3.plot(xr,yt,color=RJ,linewidth=1.8,label=f'Tsallis q={self.tsallis.q:.2f}')
        ax3.set_yscale('log'); ax3.legend(facecolor=BG,labelcolor='white',fontsize=9)

        ax4=fig.add_subplot(gs[1,1]); estilo(ax4,'Volatilidad Rolling')
        vol=self.regimen.vol_rolling*100; um=float(vol.median())
        ax4.plot(vol.index,vol.values,color=AZ,linewidth=1.2)
        ax4.axhline(um,color=RJ,linestyle='--',linewidth=1.2,label=f'Mediana {um:.1f}%')
        ax4.fill_between(vol.index,vol.values,um,where=(vol.values>um),alpha=0.2,color=RJ)
        ax4.legend(facecolor=BG,labelcolor='white',fontsize=9)

        fig.suptitle(f'ANÁLISIS ECONOFÍSICO — {self.ticker}',color='white',fontsize=14,fontweight='bold')
        plt.tight_layout(); plt.show()


def analizar_activo(ticker: str, periodo: str = '1y'):
    d = Dashboard(ticker); d.analizar(periodo); d.imprimir_reporte(); d.graficar()
    return d

def analizar_portafolio(tickers: list, periodo: str = '1y'):
    dl = get_data_layer(); rets = {}
    for t in tickers:
        r = dl.get_retornos(t, 'mediano', periodo)
        if not r.empty: rets[t] = r
    if len(rets) < 2: print("Se necesitan al menos 2 activos"); return
    df = pd.DataFrame(rets).dropna()
    opt = OptimizadorPortafolio(df); res = opt.optimizar(lambda_riesgo=2.0)
    print(f"\n{'═'*50}\n  PORTFOLIO ÓPTIMO\n{'═'*50}")
    for t,p in res['pesos'].items():
        print(f"  {t:<14} {p*100:5.1f}%  {'█'*int(p*30)}")
    print(f"\n  Retorno est.: {res['retorno_anual']*100:.1f}%  Sharpe: {res['sharpe_ratio']:.2f}")


if __name__ == '__main__':
    # Crypto con datos de Binance en tiempo real
    analizar_activo('BTCUSDT', '1y')
    analizar_activo('ETHUSDT', '6mo')
    # Stocks con yfinance
    analizar_activo('AAPL', '1y')
    # Portfolio mixto
    analizar_portafolio(['BTCUSDT','ETHUSDT','AAPL','NVDA','SPY'], '1y')
