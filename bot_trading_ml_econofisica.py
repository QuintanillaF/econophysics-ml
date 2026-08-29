"""
╔══════════════════════════════════════════════════════════════════╗
║   BOT TRADING ML + ECONOFÍSICA — BACKTEST                       ║
║   v2.0 — Usa DataLayer (Binance + CoinGecko + yfinance)         ║
╚══════════════════════════════════════════════════════════════════╝
INSTALACIÓN: pip install yfinance scikit-learn numpy scipy pandas matplotlib requests
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings; warnings.filterwarnings('ignore')
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from data_layer import get_data_layer, normalizar_ticker
from varengine import deflated_sharpe_ratio, probability_of_backtest_overfitting

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, ValueError):
    pass

try:
    import xgboost as xgb; XGBOOST = True
except: XGBOOST = False


class GeneradorFeatures:
    def __init__(self, ventana_hurst=63, ventana_vol=21, ventana_larga=63):
        self.vh=ventana_hurst; self.vv=ventana_vol; self.vl=ventana_larga

    def _hurst_v(self, s):
        if len(s)<20 or np.std(s)<1e-10: return 0.5
        d=np.cumsum(s-s.mean()); return float(np.clip(np.log((d.max()-d.min())/np.std(s))/np.log(len(s)),0,1))

    def _rsi(self, p, n=14):
        d=p.diff(); g=d.clip(lower=0).rolling(n).mean(); lo=(-d.clip(upper=0)).rolling(n).mean()
        return 100-(100/(1+g/(lo+1e-10)))

    def _bb_pos(self, p, n=20):
        m=p.rolling(n).mean(); s=p.rolling(n).std()
        return (p-(m-2*s))/(4*s+1e-10)-0.5

    def calcular(self, df: pd.DataFrame) -> pd.DataFrame:
        c=df['close'].squeeze(); r=np.log(c/c.shift(1)); f=pd.DataFrame(index=df.index)
        f['hurst']      = r.rolling(self.vh).apply(self._hurst_v, raw=True)
        f['fractal_dim']= 2-f['hurst']
        v21=r.rolling(self.vv).std()*np.sqrt(252); v63=r.rolling(self.vl).std()*np.sqrt(252)
        f['vol_ratio']  = v21/(v63+1e-10)
        f['regimen_vol']= (v21>v63.rolling(63).median()).astype(int)
        f['tsallis_q']  = (1+2/(r.rolling(self.vl).kurt().abs()+3)).clip(1.0,2.5)
        f['rsi_14']     = self._rsi(c,14); f['rsi_28'] = self._rsi(c,28)
        f['bb_pos']     = self._bb_pos(c)
        for n in [5,10,21,63]: f[f'mom_{n}']=r.rolling(n).sum()
        f['vol_5']=r.rolling(5).std(); f['vol_21']=r.rolling(21).std()
        f['autocorr_5']=r.rolling(21).apply(lambda x: pd.Series(x).autocorr(lag=5) if len(x)>5 else 0, raw=False)
        f['ret_lag1']=r.shift(1); f['ret_lag2']=r.shift(2); f['ret_lag3']=r.shift(3)
        return f


class GeneradorLabels:
    def __init__(self, metodo='triple_barrier'):
        self.metodo = metodo

    def generar(self, precios, retornos, horizonte=5, umbral=0.025):
        if self.metodo == 'triple_barrier': return self._triple_barrier(precios, horizonte, umbral)
        return self._fixed_horizon(retornos, horizonte, umbral)

    def _fixed_horizon(self, r, h, u):
        rf=r.shift(-h).rolling(h).sum(); labels=pd.Series(0,index=r.index)
        labels[rf>u]=1; labels[rf<-u]=-1; return labels

    def _triple_barrier(self, p, h, u):
        labels=pd.Series(0,index=p.index); pv=p.values; n=len(pv)
        for i in range(n-h):
            p0=pv[i]
            for j in range(1,h+1):
                if i+j>=n: break
                if pv[i+j]>=p0*(1+u): labels.iloc[i]=1; break
                if pv[i+j]<=p0*(1-u): labels.iloc[i]=-1; break
        return labels


class ModeloTrading:
    def __init__(self, tipo='random_forest'):
        self.tipo=tipo; self.modelo=None; self.scaler=StandardScaler(); self.feature_names=None
        self._construir()

    def _construir(self):
        if self.tipo=='random_forest':
            clf=RandomForestClassifier(n_estimators=200,max_depth=4,min_samples_leaf=20,max_features='sqrt',class_weight='balanced',random_state=42,n_jobs=-1)
        elif self.tipo=='xgboost' and XGBOOST:
            clf=xgb.XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,eval_metric='mlogloss',random_state=42)
        else:
            clf=GradientBoostingClassifier(n_estimators=150,max_depth=3,learning_rate=0.05,subsample=0.8,random_state=42)
        self.modelo=CalibratedClassifierCV(clf,cv=3,method='sigmoid')

    def walk_forward(self, X, y, n_train=252, step=21, purge=0):
        # `purge` drops the last `purge` training rows before each test block: with
        # triple-barrier labels a training label realised over the next N days
        # overlaps the test period and leaks. Set purge = label horizon.
        self.feature_names=list(X.columns); n=len(X)
        preds=pd.Series(0,index=X.index,dtype=int)
        probs=pd.DataFrame(0.0,index=X.index,columns=[-1,0,1])
        print(f"   Walk-forward: {(n-n_train)//step} períodos (purge={purge})...")
        for t in range(n_train,n,step):
            cut=max(1, t-purge)
            Xtr=X.iloc[:cut]; ytr=y.iloc[:cut]; te=min(t+step,n); Xte=X.iloc[t:te]
            mask=~(Xtr.isna().any(axis=1)|ytr.isna()); Xtr,ytr=Xtr[mask],ytr[mask]
            if len(ytr.unique())<2 or len(Xtr)<50: continue
            try:
                Xts=self.scaler.fit_transform(Xtr); Xtes=self.scaler.transform(Xte.fillna(0))
                self.modelo.fit(Xts,ytr); ps=self.modelo.predict(Xtes)
                pb=self.modelo.predict_proba(Xtes); cl=list(self.modelo.classes_)
                preds.iloc[t:te]=ps
                for i,c in enumerate(cl):
                    if c in probs.columns: probs[c].iloc[t:te]=pb[:,i]
            except: continue
        return pd.DataFrame({'prediccion':preds,'prob_compra':probs[1],'prob_neutral':probs[0],'prob_venta':probs[-1]})

    def importancia_features(self):
        try:
            clf=self.modelo.calibrated_classifiers_[0].estimator
            return pd.Series(clf.feature_importances_,index=self.feature_names).sort_values(ascending=False)
        except: return pd.Series(dtype=float)


class BacktestEngine:
    def __init__(self, capital_inicial=10_000, comision=0.001, slippage=0.0005, umbral_confianza=0.55, fraccion_kelly=0.25, stop_loss_global=0.20):
        self.capital_0=capital_inicial; self.comision=comision; self.slippage=slippage
        self.umbral=umbral_confianza; self.kelly=fraccion_kelly; self.stop=stop_loss_global

    def correr(self, retornos, señales):
        idx=retornos.index.intersection(señales.index); r=retornos.loc[idx]; s=señales.loc[idx]
        capital=self.capital_0; posicion=0; detenido=False; historial=[]
        for i in range(len(r)):
            ret=float(r.iloc[i]); pred=int(s['prediccion'].iloc[i])
            pc=float(s['prob_compra'].iloc[i]); pv=float(s['prob_venta'].iloc[i])
            if capital<=self.capital_0*(1-self.stop): detenido=True; posicion=0
            sf=0
            if not detenido:
                if pred==1 and pc>=self.umbral: sf=1
                elif pred==-1 and pv>=self.umbral: sf=-1
            if sf!=posicion: capital*=(1-(self.comision+self.slippage)); posicion=sf
            pnl=posicion*self.kelly*ret; capital*=np.exp(pnl)
            historial.append({'fecha':r.index[i],'capital':capital,'posicion':posicion,'retorno':pnl,'señal':sf})
        df=pd.DataFrame(historial).set_index('fecha')
        df['retorno_acum']=(df['capital']/self.capital_0-1)*100
        df['buy_hold']=(np.exp(r.loc[idx].cumsum())-1)*100*self.kelly
        return df

    def calcular_metricas(self, h, retornos):
        cap=h['capital']; rd=h['retorno']; años=len(rd)/252
        rt=float(cap.iloc[-1]/self.capital_0-1); cagr=float((1+rt)**(1/años)-1) if años>0 else 0
        vol=float(rd.std()*np.sqrt(252)); sharpe=float(rd.mean()/(rd.std()+1e-10)*np.sqrt(252))
        rn=rd[rd<0]; sortino=float(rd.mean()/(rn.std()+1e-10)*np.sqrt(252))
        peak=cap.cummax(); max_dd=float(((cap-peak)/peak).min())
        bh=float(np.exp(retornos.sum())-1); alpha=rt-bh
        cambios=h['posicion'].diff().abs(); n_trades=int((cambios>0).sum())
        return {'capital_final':round(float(cap.iloc[-1]),2),'retorno_total':round(rt*100,2),
                'cagr':round(cagr*100,2),'vol_anual':round(vol*100,2),'sharpe_ratio':round(sharpe,3),
                'sortino_ratio':round(sortino,3),'max_drawdown':round(max_dd*100,2),
                'calmar_ratio':round(cagr/abs(max_dd) if max_dd!=0 else 0,3),
                'n_operaciones':n_trades,'buy_hold_total':round(bh*100,2),'alpha':round(alpha*100,2)}


class SistemaTrading:
    def __init__(self, ticker, periodo='2y', modelo_tipo='random_forest', label_metodo='triple_barrier',
                 horizonte_label=5, umbral_label=0.025, umbral_confianza=0.55, capital_inicial=10_000):
        self.ticker=ticker; self.periodo=periodo; self.capital_0=capital_inicial
        self.dl=get_data_layer(); self.info=normalizar_ticker(ticker)
        self.gen_f=GeneradorFeatures(); self.gen_l=GeneradorLabels(label_metodo)
        self.modelo=ModeloTrading(modelo_tipo)
        self.backtest=BacktestEngine(capital_inicial=capital_inicial,umbral_confianza=umbral_confianza)
        self.horizonte=horizonte_label; self.umbral=umbral_label
        self.historial=None; self.metricas=None

    def correr(self, verbose=True):
        ticker_d=self.info['yfinance'] or self.ticker
        fuente='Binance REST' if (self.info['es_crypto'] and self.dl.binance.disponible) else 'yfinance'
        print(f"\n{'═'*55}\n  BACKTEST ML: {ticker_d}\n  Fuente: {fuente}  │  Período: {self.periodo}\n{'═'*55}\n")

        print("[1/5] Descargando datos...")
        horizonte='mediano' if self.periodo in ['1y','2y'] else 'largo'
        df=self.dl.get_ohlcv(self.ticker,horizonte,self.periodo)
        if df.empty: print("Sin datos"); return {}
        precios=df['close'].squeeze(); retornos=np.log(precios/precios.shift(1)).dropna()
        print(f"  ✓ {len(retornos)} días  │  fuente: {fuente}")

        print("\n[2/5] Calculando features...")
        features=self.gen_f.calcular(df); print(f"  ✓ {features.shape[1]} features")

        print("\n[3/5] Generando labels...")
        labels=self.gen_l.generar(precios,retornos,self.horizonte,self.umbral)
        dist=labels.value_counts()
        print(f"  ✓ Compra={dist.get(1,0)} Neutral={dist.get(0,0)} Venta={dist.get(-1,0)}")

        print("\n[4/5] Walk-forward validation...")
        idx=features.index.intersection(labels.index); X=features.loc[idx]; y=labels.loc[idx]
        señales=self.modelo.walk_forward(X,y,purge=self.horizonte)

        print("\n[5/5] Backtest...")
        self.historial=self.backtest.correr(retornos,señales)
        self.metricas=self.backtest.calcular_metricas(self.historial,retornos)
        self._deflacionar_sharpe()
        self._imprimir(ticker_d); return self.metricas

    def _deflacionar_sharpe(self, n_trials=20):
        """Deflated Sharpe Ratio de la curva de estrategia.

        El Sharpe de un backtest está inflado por todo lo que se probó y descartó.
        El DSR compara el Sharpe observado contra el que se esperaría por azar tras
        `n_trials` configuraciones, ajustando por asimetría y curtosis. DSR < 0.5 =
        más probable que la habilidad sea ruido de selección.
        """
        r = self.historial['retorno'].dropna()
        try:
            self.dsr = deflated_sharpe_ratio(r, n_trials=n_trials)
        except ValueError:
            self.dsr = None

    def _imprimir(self, ticker):
        m=self.metricas; sep="─"*50
        print(f"\n{'═'*55}\n  RESULTADO — {ticker}\n{'═'*55}")
        print(f"  Capital inicial:  ${self.capital_0:>10,.2f}")
        print(f"  Capital final:    ${m['capital_final']:>10,.2f}")
        print(sep)
        print(f"  Retorno total:    {m['retorno_total']:>+8.1f}%")
        print(f"  CAGR:             {m['cagr']:>+8.1f}%")
        print(f"  Buy & Hold:       {m['buy_hold_total']:>+8.1f}%")
        print(f"  Alpha:            {m['alpha']:>+8.1f}%")
        print(sep)
        print(f"  Sharpe:           {m['sharpe_ratio']:>8.3f}")
        print(f"  Max Drawdown:     {m['max_drawdown']:>8.1f}%")
        print(f"  N Operaciones:    {m['n_operaciones']:>8}")
        if getattr(self, 'dsr', None):
            d = self.dsr
            print(sep)
            print(f"  Deflated Sharpe:  {d['dsr']:>8.2f}   (PSR {d['psr']:.2f}, "
                  f"umbral por azar {d['sr_threshold']:.2f} con {d['n_trials']} pruebas)")
            print(f"  Veredicto:        {d['verdict']:>8}   "
                  f"(skew {d['skew']:+.2f}, kurtosis {d['kurtosis']:.2f})")
        print(f"{'═'*55}\n")

    def evaluar_robustez(self, grid=None, verbose=True):
        """Corre una grilla de configuraciones y calcula la Probability of
        Backtest Overfitting (PBO) por CSCV.

        Si al variar umbral de confianza / horizonte / Kelly el "mejor" en la
        primera mitad de la muestra deja de ser bueno en la segunda, la búsqueda
        de parámetros está sobreajustando. PBO ≈ 0.5 = la selección no aporta
        información; PBO alto = el backtest miente.

        Es lento (una corrida walk-forward por combinación). No se ejecuta por
        defecto.
        """
        if grid is None:
            grid = [
                (u, h, k)
                for u in (0.52, 0.55, 0.60)
                for h in (3, 5, 10)
                for k in (0.15, 0.25)
            ]
        pnl_cols, etiquetas = [], []
        base_idx = None
        for (u, h, k) in grid:
            s = SistemaTrading(self.ticker, periodo=self.periodo,
                               modelo_tipo=self.modelo.tipo, horizonte_label=h,
                               umbral_confianza=u, capital_inicial=self.capital_0)
            s.backtest.kelly = k
            try:
                s.correr(verbose=False)
            except Exception:
                continue
            r = s.historial['retorno'].dropna()
            if base_idx is None:
                base_idx = r.index
            pnl_cols.append(r.reindex(base_idx).fillna(0.0).to_numpy())
            etiquetas.append(f"u{u}_h{h}_k{k}")

        if len(pnl_cols) < 4:
            print("No hay suficientes configuraciones válidas para PBO.")
            return None

        M = np.column_stack(pnl_cols)
        n_splits = 10 if M.shape[0] >= 20 else 4
        pbo = probability_of_backtest_overfitting(M, n_splits=n_splits)
        self.pbo = pbo
        if verbose:
            print(f"\n{'═'*55}\n  ROBUSTEZ — {len(etiquetas)} configuraciones\n{'═'*55}")
            print(f"  PBO (probability of backtest overfitting): {pbo['pbo']:.1%}")
            print(f"  logit mediano OOS del mejor IS: {pbo['median_logit']:+.2f}")
            print(f"  {pbo['n_combinations']} particiones IS/OOS evaluadas")
            lectura = ("la selección de parámetros NO sobreajusta" if pbo['pbo'] < 0.3
                       else "sobreajuste moderado" if pbo['pbo'] < 0.5
                       else "el backtest está sobreajustado — el 'mejor' no generaliza")
            print(f"  → {lectura}\n{'═'*55}\n")
        return pbo

    def graficar(self):
        if self.historial is None: return
        h=self.historial; m=self.metricas; BG='#0d1117'; BGX='#161b22'; AZ='#58a6ff'; VD='#3fb950'; RJ='#f78166'; GR='#8b949e'
        fig=plt.figure(figsize=(18,10)); fig.patch.set_facecolor(BG)
        gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.45,wspace=0.35)
        def st(ax,t): ax.set_facecolor(BGX); ax.set_title(t,color='white',fontsize=10,pad=7); ax.tick_params(colors=GR)
        for sp in plt.gca().spines.values(): sp.set_edgecolor('#30363d')

        ax1=fig.add_subplot(gs[0,:]); st(ax1,f'Equity Curve — {self.ticker}  Alpha: {m["alpha"]:+.1f}%')
        ax1.plot(h.index,h['capital'],color=AZ,linewidth=1.5,label=f'Estrategia ({m["retorno_total"]:+.1f}%)')
        bh=self.capital_0*(1+h['buy_hold']/100)
        ax1.plot(h.index,bh,'--',color=GR,linewidth=1.0,alpha=0.7,label=f'Buy&Hold ({m["buy_hold_total"]:+.1f}%)')
        ax1.axhline(self.capital_0,color=GR,linewidth=0.6,linestyle=':',alpha=0.5)
        ax1.fill_between(h.index,h['capital'],self.capital_0,where=h['capital']>=self.capital_0,alpha=0.1,color=VD)
        ax1.fill_between(h.index,h['capital'],self.capital_0,where=h['capital']<self.capital_0,alpha=0.15,color=RJ)
        ax1.legend(facecolor=BGX,labelcolor='white',fontsize=9)

        ax2=fig.add_subplot(gs[1,:2]); st(ax2,f'Drawdown (Máx: {m["max_drawdown"]:.1f}%)')
        peak=h['capital'].cummax(); dd=(h['capital']-peak)/peak*100
        ax2.fill_between(h.index,dd.values,0,color=RJ,alpha=0.4)
        ax2.plot(h.index,dd.values,color=RJ,linewidth=0.8)

        ax3=fig.add_subplot(gs[1,2]); st(ax3,'Importancia Features')
        imp=self.modelo.importancia_features()
        if not imp.empty:
            top=imp.head(10)
            cols=[('#e3b341' if any(x in f for x in ['hurst','fractal','tsallis','regimen','autocorr']) else AZ) for f in top.index]
            ax3.barh(range(len(top)),top.values,color=cols,alpha=0.8)
            ax3.set_yticks(range(len(top))); ax3.set_yticklabels(top.index,fontsize=8); ax3.invert_yaxis()

        fig.suptitle(f'BACKTEST ML — {self.ticker}',color='white',fontsize=13,fontweight='bold')
        plt.tight_layout(); plt.show()


def analizar_activo(ticker, periodo='2y'):
    s=SistemaTrading(ticker,periodo=periodo,capital_inicial=10_000); s.correr(); s.graficar(); return s

if __name__ == '__main__':
    analizar_activo('BTCUSDT','2y')
    analizar_activo('AAPL','2y')
