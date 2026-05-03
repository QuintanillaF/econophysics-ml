"""
╔══════════════════════════════════════════════════════════════════╗
║   BOT DE TRADING — ECONOFÍSICA + ML + BACKTESTING               ║
║   Sin broker │ Sin dinero real │ 100% local                     ║
╚══════════════════════════════════════════════════════════════════╝

ARQUITECTURA:
  ┌─────────────────┐     ┌──────────────────┐     ┌────────────┐
  │  FEATURES       │────▶│  MODELO ML       │────▶│ BACKTEST   │
  │  (Econofísica)  │     │  (RandomForest / │     │ ENGINE     │
  │                 │     │   XGBoost)       │     │            │
  │ • Hurst rolling │     │                  │     │ • PnL      │
  │ • Régimen vol   │     │  Predice:        │     │ • Sharpe   │
  │ • Tsallis q     │     │  +1 (subir)      │     │ • Drawdown │
  │ • Momentum      │     │  -1 (bajar)      │     │ • Win rate │
  │ • RSI, BB       │     │   0 (neutral)    │     │            │
  └─────────────────┘     └──────────────────┘     └────────────┘

FLUJO COMPLETO:
  1. Descarga datos históricos
  2. Calcula features econofísicas + técnicas
  3. Genera etiquetas (labels) de trading
  4. Entrena modelo con walk-forward validation
  5. Corre backtest realista (con comisiones + slippage)
  6. Reporta métricas y gráficos

INSTALACIÓN:
  pip install yfinance numpy scipy pandas matplotlib seaborn
  pip install scikit-learn xgboost ta

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize

try:
    import yfinance as yf
except ImportError:
    raise ImportError("pip install yfinance")

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
except ImportError:
    raise ImportError("pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_DISPONIBLE = True
except ImportError:
    XGBOOST_DISPONIBLE = False
    print("XGBoost no disponible — se usará GradientBoosting de sklearn")


# ══════════════════════════════════════════════════════════════
# MÓDULO 1: GENERADOR DE FEATURES ECONOFÍSICAS
# ══════════════════════════════════════════════════════════════
class GeneradorFeatures:
    """
    Convierte precios crudos en features para el modelo ML.

    FEATURES ECONOFÍSICAS:
        hurst_rolling    → ¿hay tendencia en la ventana reciente?
        regimen_vol      → 0=calma / 1=estrés
        tsallis_q_est    → estimación de colas gruesas (curtosis local)
        fractal_dim      → dimensión fractal (2 - H)

    FEATURES TÉCNICAS CLÁSICAS (complemento):
        rsi              → momentum (0-100)
        bb_pos           → posición dentro de Bollinger Bands
        momentum_n       → retorno acumulado últimos n días
        vol_ratio        → volatilidad corta / volatilidad larga

    
    """

    def __init__(self, ventana_hurst: int = 63,
                 ventana_vol: int = 21,
                 ventana_larga: int = 63):
        """
        Args:
            ventana_hurst: Días para calcular Hurst rolling (63 = 1 trimestre)
            ventana_vol:   Días para volatilidad corta (21 = 1 mes)
            ventana_larga: Días para volatilidad larga (63 = 1 trimestre)
        """
        self.v_hurst = ventana_hurst
        self.v_vol   = ventana_vol
        self.v_larga = ventana_larga

    # ── Hurst rolling (R/S simplificado para velocidad) ──────
    def _hurst_ventana(self, serie: np.ndarray) -> float:
        """R/S en una ventana — versión rápida para rolling."""
        n = len(serie)
        if n < 20:
            return 0.5

        media = serie.mean()
        std   = serie.std()
        if std < 1e-10:
            return 0.5

        desv = np.cumsum(serie - media)
        R    = desv.max() - desv.min()
        return np.log(R / std) / np.log(n)

    def _hurst_rolling(self, retornos: pd.Series) -> pd.Series:
        """Aplica Hurst sobre ventana deslizante."""
        return retornos.rolling(self.v_hurst).apply(
            self._hurst_ventana, raw=True
        )

    # ── RSI ───────────────────────────────────────────────────
    def _rsi(self, precios: pd.Series, n: int = 14) -> pd.Series:
        delta  = precios.diff()
        ganancia = delta.clip(lower=0).rolling(n).mean()
        perdida  = (-delta.clip(upper=0)).rolling(n).mean()
        rs  = ganancia / (perdida + 1e-10)
        return 100 - (100 / (1 + rs))

    # ── Bollinger Bands position ──────────────────────────────
    def _bb_position(self, precios: pd.Series, n: int = 20) -> pd.Series:
        """Posición dentro de las bandas: -1 (inferior) a +1 (superior)."""
        media = precios.rolling(n).mean()
        std   = precios.rolling(n).std()
        upper = media + 2 * std
        lower = media - 2 * std
        return (precios - lower) / (upper - lower + 1e-10) * 2 - 1

    # ── Estimación de q de Tsallis (curtosis local) ───────────
    def _tsallis_q_rolling(self, retornos: pd.Series) -> pd.Series:
        """
        Aproximación rápida de q usando curtosis local:
        q ≈ 1 + 2/(curtosis + 3)  para distribución q-Gaussiana
        """
        kurt = retornos.rolling(self.v_larga).kurt()
        q = 1 + 2 / (kurt.abs() + 3)
        return q.clip(1.0, 2.5)

    # ── Función principal ─────────────────────────────────────
    def calcular(self, datos: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todas las features.

        Args:
            datos: DataFrame con columnas OHLCV
        Returns:
            DataFrame con features (una fila por día)
        """
        precios  = datos['Close'].squeeze()
        retornos = np.log(precios / precios.shift(1))

        features = pd.DataFrame(index=datos.index)

        # ── Econofísicas ─────────────────────────────────────
        features['hurst']        = self._hurst_rolling(retornos)
        features['fractal_dim']  = 2 - features['hurst']   # dim fractal

        # Régimen de volatilidad
        vol_corta = retornos.rolling(self.v_vol).std() * np.sqrt(252)
        vol_larga = retornos.rolling(self.v_larga).std() * np.sqrt(252)
        features['regimen_vol']  = (vol_corta > vol_larga.rolling(63).median()
                                    ).astype(int)
        features['vol_ratio']    = vol_corta / (vol_larga + 1e-10)
        features['tsallis_q']    = self._tsallis_q_rolling(retornos)

        # ── Técnicas ─────────────────────────────────────────
        features['rsi_14']      = self._rsi(precios, 14)
        features['rsi_28']      = self._rsi(precios, 28)
        features['bb_pos']      = self._bb_position(precios)

        # Momentum a distintos horizontes
        for n in [5, 10, 21, 63]:
            features[f'mom_{n}'] = retornos.rolling(n).sum()

        # Volatilidad realizada
        features['vol_5']   = retornos.rolling(5).std()
        features['vol_21']  = retornos.rolling(21).std()

        # Autocorrelación (señal de memoria)
        features['autocorr_5'] = retornos.rolling(21).apply(
            lambda x: pd.Series(x).autocorr(lag=5) if len(x) > 5 else 0,
            raw=False
        )

        # Retorno del día anterior (lag feature)
        features['ret_lag1'] = retornos.shift(1)
        features['ret_lag2'] = retornos.shift(2)
        features['ret_lag3'] = retornos.shift(3)

        return features


# ══════════════════════════════════════════════════════════════
# MÓDULO 2: GENERADOR DE LABELS 
# ══════════════════════════════════════════════════════════════
class GeneradorLabels:
    """
    Genera las etiquetas (y) que el modelo debe aprender a predecir.

    MÉTODOS:
        triple_barrier  → Método de López de Prado (más realista)
        fixed_horizon   → Simple: sube/baja en N días (más fácil de entender)

    RECOMENDACIÓN:
        Usar triple_barrier para resultados más realistas.
        fixed_horizon para entender el sistema primero.

    CLASES:
         1 → BUY  (señal de compra)
         0 → HOLD (no hacer nada)
        -1 → SELL (señal de venta / short)
    """

    def __init__(self, método: str = 'triple_barrier'):
        """
        Args:
            método: 'triple_barrier' o 'fixed_horizon'
        """
        self.método = método

    def generar(self, precios: pd.Series, retornos: pd.Series,
                horizonte: int = 5, umbral: float = 0.02) -> pd.Series:
        """
        Args:
            precios:   Serie de precios de cierre
            retornos:  Serie de retornos logarítmicos
            horizonte: Días hacia adelante para clasificar
            umbral:    % de movimiento para considerar señal (ej: 0.02 = 2%)
        """
        if self.método == 'triple_barrier':
            return self._triple_barrier(precios, retornos, horizonte, umbral)
        else:
            return self._fixed_horizon(retornos, horizonte, umbral)

    def _fixed_horizon(self, retornos: pd.Series,
                       horizonte: int, umbral: float) -> pd.Series:
        """
        Retorno acumulado en los próximos N días.
        Si > umbral  → 1 (comprar)
        Si < -umbral → -1 (vender)
        Si en medio  → 0 (neutral)
        """
        ret_futuro = retornos.shift(-horizonte).rolling(horizonte).sum()
        labels = pd.Series(0, index=retornos.index)
        labels[ret_futuro >  umbral] =  1
        labels[ret_futuro < -umbral] = -1
        return labels

    def _triple_barrier(self, precios: pd.Series, retornos: pd.Series,
                        horizonte: int, umbral: float) -> pd.Series:
        """
        Método Triple Barrier de López de Prado:
        - Barrera superior: +umbral% (take profit → label 1)
        - Barrera inferior: -umbral% (stop loss  → label -1)
        - Barrera temporal: horizonte días (si no toca ninguna → 0)

        
        """
        labels = pd.Series(0, index=precios.index)
        p = precios.values
        n = len(p)

        for i in range(n - horizonte):
            p0 = p[i]
            tp = p0 * (1 + umbral)   # take profit
            sl = p0 * (1 - umbral)   # stop loss

            label = 0  # por defecto: neutral (llegó al tiempo límite)
            for j in range(1, horizonte + 1):
                if i + j >= n:
                    break
                if p[i + j] >= tp:
                    label = 1
                    break
                elif p[i + j] <= sl:
                    label = -1
                    break

            labels.iloc[i] = label

        return labels


# ══════════════════════════════════════════════════════════════
# MÓDULO 3: MODELO ML
# ══════════════════════════════════════════════════════════════
class ModeloTrading:
    """
    Modelo de clasificación: predice la dirección del mercado.

    MODELOS DISPONIBLES:
        'random_forest'  → Robusto, resistente a overfitting,
                           bueno para empezar, fácil de interpretar
        'xgboost'        → Más preciso, requiere más tuning
        'gradient_boost' → Alternativa si XGBoost no está instalado

    
    """

    def __init__(self, tipo: str = 'random_forest'):
        self.tipo    = tipo
        self.modelo  = None
        self.scaler  = StandardScaler()
        self.feature_names = None
        self._construir()

    def _construir(self):
        if self.tipo == 'random_forest':
            clf = RandomForestClassifier(
                n_estimators=200,
                max_depth=4,          # poco profundo → menos overfitting
                min_samples_leaf=20,  # mínimo 20 muestras por hoja
                max_features='sqrt',
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
            )
        elif self.tipo == 'xgboost' and XGBOOST_DISPONIBLE:
            clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=42,
            )
        else:
            clf = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )

        # Calibrar probabilidades (importante para sizing de posición)
        self.modelo = CalibratedClassifierCV(clf, cv=3, method='sigmoid')

    def walk_forward(self, X: pd.DataFrame, y: pd.Series,
                     n_train_min: int = 252,
                     step: int = 21) -> pd.DataFrame:
        """
        Validación walk-forward.

        Args:
            X:           Features
            y:           Labels
            n_train_min: Mínimo de días para primer entrenamiento (252 = 1 año)
            step:        Días entre reentrenamientos (21 = mensual)

        Returns:
            DataFrame con predicciones y probabilidades out-of-sample
        """
        self.feature_names = list(X.columns)
        n = len(X)

        predicciones = pd.Series(0, index=X.index, dtype=int)
        probabilidades = pd.DataFrame(0.0, index=X.index,
                                       columns=[-1, 0, 1])

        print(f"   Walk-forward validation "
              f"({(n - n_train_min) // step} períodos)...")

        periodos_completados = 0
        for t in range(n_train_min, n, step):
            # Datos de entrenamiento: todo lo anterior
            X_train = X.iloc[:t]
            y_train = y.iloc[:t]

            # Datos de test: próximos 'step' días
            t_end   = min(t + step, n)
            X_test  = X.iloc[t:t_end]

            # Eliminar NaNs del train
            mask = ~(X_train.isna().any(axis=1) | y_train.isna())
            X_tr = X_train[mask]
            y_tr = y_train[mask]

            if len(y_tr.unique()) < 2 or len(X_tr) < 50:
                continue  # poco data o solo una clase

            # Entrenar
            X_tr_sc = self.scaler.fit_transform(X_tr)
            X_te_sc = self.scaler.transform(X_test.fillna(0))

            try:
                self.modelo.fit(X_tr_sc, y_tr)
                preds = self.modelo.predict(X_te_sc)
                probas = self.modelo.predict_proba(X_te_sc)
                clases = self.modelo.classes_

                predicciones.iloc[t:t_end] = preds

                for i, c in enumerate(clases):
                    if c in probabilidades.columns:
                        probabilidades[c].iloc[t:t_end] = probas[:, i]

            except Exception as e:
                print(f"   Error en período {t}: {e}")
                continue

            periodos_completados += 1

        print(f"   ✓ {periodos_completados} períodos completados")

        return pd.DataFrame({
            'prediccion':    predicciones,
            'prob_compra':   probabilidades[1],
            'prob_neutral':  probabilidades[0],
            'prob_venta':    probabilidades[-1],
        })

    def importancia_features(self) -> pd.Series:
        """Importancia de cada feature (solo para RF y XGBoost)."""
        try:
            clf = self.modelo.calibrated_classifiers_[0].estimator
            imp = clf.feature_importances_
            return pd.Series(imp, index=self.feature_names).sort_values(
                ascending=False
            )
        except Exception:
            return pd.Series(dtype=float)


# ══════════════════════════════════════════════════════════════
# MÓDULO 4: MOTOR DE BACKTESTING
# ══════════════════════════════════════════════════════════════
class BacktestEngine:
   

    def __init__(self,
                 capital_inicial: float = 10_000,
                 comision: float = 0.001,      # 0.1% por operación
                 slippage: float = 0.0005,     # 0.05% slippage
                 umbral_confianza: float = 0.55,  # probabilidad mínima para operar
                 fraccion_kelly: float = 0.25,    # Kelly fraccionario (conservador)
                 stop_loss_global: float = 0.20): # stop si portfolio cae 20%
        self.capital_0         = capital_inicial
        self.comision          = comision
        self.slippage          = slippage
        self.umbral            = umbral_confianza
        self.fraccion_kelly    = fraccion_kelly
        self.stop_loss_global  = stop_loss_global

    def correr(self, retornos: pd.Series,
               señales: pd.DataFrame) -> pd.DataFrame:
        """
        Simula el trading día a día.

        Args:
            retornos: Retornos logarítmicos del activo
            señales:  DataFrame con 'prediccion', 'prob_compra', 'prob_venta'

        Returns:
            DataFrame con historial completo del portafolio
        """
        # Alinear índices
        idx = retornos.index.intersection(señales.index)
        r   = retornos.loc[idx]
        s   = señales.loc[idx]

        n            = len(r)
        capital      = self.capital_0
        posicion     = 0          # -1, 0, +1
        entrada_px   = 0.0
        n_trades     = 0
        trades_ganados = 0

        historial = []
        detenido  = False  # stop loss global activado

        for i in range(n):
            ret  = float(r.iloc[i])
            pred = int(s['prediccion'].iloc[i])
            prob_c = float(s['prob_compra'].iloc[i])
            prob_v = float(s['prob_venta'].iloc[i])

            # Stop loss global
            if capital <= self.capital_0 * (1 - self.stop_loss_global):
                if not detenido:
                    print(f"   Stop loss global activado en día {i}")
                detenido  = True
                posicion  = 0

            # Determinar señal filtrada por umbral de confianza
            señal_filtrada = 0
            if not detenido:
                if pred == 1  and prob_c >= self.umbral:
                    señal_filtrada =  1
                elif pred == -1 and prob_v >= self.umbral:
                    señal_filtrada = -1

            # Cambio de posición → aplicar costos
            if señal_filtrada != posicion:
                costo = abs(self.comision + self.slippage)
                capital *= (1 - costo)
                if posicion != 0:
                    n_trades += 1
                posicion = señal_filtrada

            # Kelly fraccionario para sizing
            if posicion != 0:
                # Estimación de win rate local (últimas 20 señales)
                tamaño = self.fraccion_kelly
            else:
                tamaño = 0.0

            # P&L del día
            pnl_dia = posicion * tamaño * ret
            capital *= np.exp(pnl_dia)

            historial.append({
                'fecha':    r.index[i],
                'capital':  capital,
                'posicion': posicion,
                'retorno':  pnl_dia,
                'señal':    señal_filtrada,
            })

        df = pd.DataFrame(historial).set_index('fecha')
        df['retorno_acum'] = (df['capital'] / self.capital_0 - 1) * 100

        # Benchmark: buy & hold
        r_bh = r.loc[idx]
        df['buy_hold'] = (np.exp(r_bh.cumsum()) - 1) * 100 * self.fraccion_kelly

        return df

    def calcular_metricas(self, historial: pd.DataFrame,
                          retornos: pd.Series) -> dict:
        """Calcula métricas estándar del backtest."""
        cap     = historial['capital']
        ret_d   = historial['retorno']
        n_dias  = len(ret_d)
        años    = n_dias / 252

        # Retorno total y anualizado
        ret_total  = float(cap.iloc[-1] / self.capital_0 - 1)
        cagr       = float((1 + ret_total) ** (1/años) - 1) if años > 0 else 0

        # Volatilidad anualizada
        vol = float(ret_d.std() * np.sqrt(252))

        # Sharpe (asumiendo tasa libre de riesgo = 0 para simplificar)
        sharpe = float(ret_d.mean() / (ret_d.std() + 1e-10) * np.sqrt(252))

        # Sortino (penaliza solo retornos negativos)
        ret_neg = ret_d[ret_d < 0]
        sortino = float(ret_d.mean() / (ret_neg.std() + 1e-10) * np.sqrt(252))

        # Máximo drawdown
        peak   = cap.cummax()
        dd     = (cap - peak) / peak
        max_dd = float(dd.min())

        # Calmar ratio
        calmar = cagr / abs(max_dd) if max_dd != 0 else 0

        # Operaciones
        cambios = historial['posicion'].diff().abs()
        n_trades = int((cambios > 0).sum())

        # Win rate
        ret_por_trade = []
        en_trade = False
        ret_acc = 0
        for i, row in historial.iterrows():
            if row['posicion'] != 0:
                ret_acc += row['retorno']
                en_trade = True
            elif en_trade:
                ret_por_trade.append(ret_acc)
                ret_acc = 0
                en_trade = False

        win_rate = (sum(1 for r in ret_por_trade if r > 0) /
                    len(ret_por_trade)) if ret_por_trade else 0

        # Comparar con buy & hold
        bh_total = float(np.exp(retornos.sum()) - 1)

        return {
            'capital_final':     round(cap.iloc[-1], 2),
            'retorno_total':     round(ret_total * 100, 2),
            'cagr':              round(cagr * 100, 2),
            'volatilidad_anual': round(vol * 100, 2),
            'sharpe_ratio':      round(sharpe, 3),
            'sortino_ratio':     round(sortino, 3),
            'max_drawdown':      round(max_dd * 100, 2),
            'calmar_ratio':      round(calmar, 3),
            'n_operaciones':     n_trades,
            'win_rate':          round(win_rate * 100, 1),
            'buy_hold_total':    round(bh_total * 100, 2),
            'alpha':             round((ret_total - bh_total) * 100, 2),
        }


# ══════════════════════════════════════════════════════════════
# MÓDULO 5: SISTEMA COMPLETO INTEGRADO
# ══════════════════════════════════════════════════════════════
class SistemaTrading:
    

    def __init__(self, ticker: str,
                 periodo: str = '3y',
                 modelo_tipo: str = 'random_forest',
                 label_método: str = 'triple_barrier',
                 horizonte_label: int = 5,
                 umbral_label: float = 0.025,
                 umbral_confianza: float = 0.55,
                 capital_inicial: float = 10_000):
        """
        Args:
            ticker:            Símbolo (ej: 'AAPL', 'BTC-USD')
            periodo:           Histórico ('2y', '3y', '5y')
            modelo_tipo:       'random_forest' o 'xgboost'
            label_método:      'triple_barrier' o 'fixed_horizon'
            horizonte_label:   Días adelante para generar labels
            umbral_label:      % movimiento para clasificar (0.025 = 2.5%)
            umbral_confianza:  Prob. mínima para ejecutar trade
            capital_inicial:   Capital simulado en USD
        """
        self.ticker         = ticker
        self.periodo        = periodo
        self.capital_0      = capital_inicial

        self.gen_features   = GeneradorFeatures()
        self.gen_labels     = GeneradorLabels(label_método)
        self.modelo         = ModeloTrading(modelo_tipo)
        self.backtest       = BacktestEngine(
            capital_inicial=capital_inicial,
            umbral_confianza=umbral_confianza,
        )

        self.horizonte      = horizonte_label
        self.umbral_label   = umbral_label

        # Resultados
        self.datos          = None
        self.features       = None
        self.labels         = None
        self.señales        = None
        self.historial      = None
        self.metricas       = None

    def correr(self, verbose: bool = True) -> dict:
        """Ejecuta el pipeline completo de principio a fin."""
        sep = "─" * 60
        print(f"\n{'═'*60}")
        print(f"  SISTEMA DE TRADING ML + ECONOFÍSICA")
        print(f"  Activo: {self.ticker}  │  Período: {self.periodo}")
        print(f"{'═'*60}\n")

        # ── 1. Datos ──────────────────────────────────────────
        print(f"[1/5] Descargando datos...")
        datos = yf.download(self.ticker, period=self.periodo,
                            auto_adjust=True, progress=False)
        if datos.empty:
            raise ValueError(f"Sin datos para {self.ticker}")
        self.datos = datos
        precios  = datos['Close'].squeeze()
        retornos = np.log(precios / precios.shift(1)).dropna()
        print(f"      ✓ {len(retornos)} días descargados")

        # ── 2. Features ───────────────────────────────────────
        print(f"\n[2/5] Calculando features econofísicas...")
        features_raw = self.gen_features.calcular(datos)
        self.features = features_raw

        n_features = features_raw.shape[1]
        print(f"      ✓ {n_features} features generadas")
        print(f"      Features: {list(features_raw.columns)}")

        # ── 3. Labels ─────────────────────────────────────────
        print(f"\n[3/5] Generando labels ({self.gen_labels.método})...")
        labels = self.gen_labels.generar(
            precios, retornos,
            horizonte=self.horizonte,
            umbral=self.umbral_label
        )
        self.labels = labels

        dist = labels.value_counts()
        print(f"      ✓ Distribución: "
              f"Compra={dist.get(1,0)} | "
              f"Neutral={dist.get(0,0)} | "
              f"Venta={dist.get(-1,0)}")

        # ── 4. Walk-forward ───────────────────────────────────
        print(f"\n[4/5] Entrenando modelo (walk-forward)...")

        # Alinear features y labels
        idx_común = features_raw.index.intersection(labels.index)
        X = features_raw.loc[idx_común]
        y = labels.loc[idx_común]

        señales = self.modelo.walk_forward(
            X, y,
            n_train_min=252,
            step=21,
        )
        self.señales = señales

        # Reporte de clasificación (período out-of-sample)
        if verbose:
            mask_oos = señales['prediccion'] != 0
            y_true = y.loc[señales.index]
            y_pred = señales['prediccion']
            valid  = ~(y_true.isna() | (y_pred == 0))
            if valid.sum() > 10:
                print(f"\n      Reporte clasificación (out-of-sample):")
                try:
                    rep = classification_report(
                        y_true[valid], y_pred[valid],
                        labels=[-1, 0, 1],
                        target_names=['VENTA','NEUTRAL','COMPRA'],
                        zero_division=0
                    )
                    for linea in rep.split('\n'):
                        print(f"      {linea}")
                except Exception:
                    pass

        # ── 5. Backtest ───────────────────────────────────────
        print(f"\n[5/5] Corriendo backtest...")
        self.historial = self.backtest.correr(retornos, señales)
        self.metricas  = self.backtest.calcular_metricas(
            self.historial, retornos
        )

        self._imprimir_metricas()
        return self.metricas

    def _imprimir_metricas(self):
        m   = self.metricas
        sep = "─" * 60
        print(f"\n{'═'*60}")
        print(f"  RESULTADOS DEL BACKTEST — {self.ticker}")
        print(f"{'═'*60}")
        print(f"\n  Capital inicial:    ${self.capital_0:>10,.2f}")
        print(f"  Capital final:      ${m['capital_final']:>10,.2f}")
        print(sep)
        print(f"  Retorno total:         {m['retorno_total']:>+8.1f}%")
        print(f"  CAGR (anualizado):     {m['cagr']:>+8.1f}%")
        print(f"  Buy & Hold:            {m['buy_hold_total']:>+8.1f}%")
        print(f"  Alpha vs B&H:          {m['alpha']:>+8.1f}%")
        print(sep)
        print(f"  Sharpe ratio:          {m['sharpe_ratio']:>8.3f}")
        print(f"  Sortino ratio:         {m['sortino_ratio']:>8.3f}")
        print(f"  Calmar ratio:          {m['calmar_ratio']:>8.3f}")
        print(f"  Volatilidad anual:     {m['volatilidad_anual']:>8.1f}%")
        print(f"  Máx. Drawdown:         {m['max_drawdown']:>8.1f}%")
        print(sep)
        print(f"  Nº operaciones:        {m['n_operaciones']:>8}")
        print(f"  Win rate:              {m['win_rate']:>8.1f}%")
        print(f"{'═'*60}\n")

        # Advertencia si los resultados parecen demasiado buenos
        if m['sharpe_ratio'] > 3.0:
            print("  Sharpe > 3 — posible overfitting. "
                  "Validar con datos nuevos.\n")

    def graficar(self):
        """Dashboard visual del backtest."""
        if self.historial is None:
            print("Ejecuta correr() primero.")
            return

        h   = self.historial
        m   = self.metricas
        BG  = '#0d1117'
        BGX = '#161b22'
        AZUL  = '#58a6ff'
        VERDE = '#3fb950'
        ROJO  = '#f78166'
        GRIS  = '#8b949e'
        AMA   = '#e3b341'

        fig = plt.figure(figsize=(18, 12))
        fig.patch.set_facecolor(BG)
        gs  = gridspec.GridSpec(3, 3, figure=fig,
                                hspace=0.5, wspace=0.35)

        def estilo(ax, titulo):
            ax.set_facecolor(BGX)
            ax.set_title(titulo, color='white', fontsize=10, pad=7)
            ax.tick_params(colors=GRIS, labelsize=8)
            for sp in ax.spines.values():
                sp.set_edgecolor('#30363d')

        # ── 1. Equity curve (principal) ───────────────────────
        ax1 = fig.add_subplot(gs[0, :])
        estilo(ax1, f'Equity Curve — {self.ticker}  '
                    f'│  Capital inicial: ${self.capital_0:,.0f}  '
                    f'│  Alpha: {m["alpha"]:+.1f}%')

        ax1.plot(h.index, h['capital'], color=AZUL,
                 linewidth=1.5, label=f'Estrategia ML ({m["retorno_total"]:+.1f}%)')
        bh = self.capital_0 * (1 + h['buy_hold'] / 100)
        ax1.plot(h.index, bh, '--', color=GRIS,
                 linewidth=1.0, alpha=0.7,
                 label=f'Buy & Hold ({m["buy_hold_total"]:+.1f}%)')
        ax1.axhline(self.capital_0, color=GRIS, linewidth=0.6,
                    linestyle=':', alpha=0.5)
        ax1.fill_between(h.index, h['capital'], self.capital_0,
                         where=h['capital'] >= self.capital_0,
                         alpha=0.1, color=VERDE)
        ax1.fill_between(h.index, h['capital'], self.capital_0,
                         where=h['capital'] < self.capital_0,
                         alpha=0.15, color=ROJO)
        ax1.legend(facecolor=BGX, labelcolor='white', fontsize=9)
        ax1.set_ylabel('Capital (USD)', color=GRIS, fontsize=9)

        # ── 2. Drawdown ───────────────────────────────────────
        ax2 = fig.add_subplot(gs[1, :2])
        estilo(ax2, f'Drawdown  (Máx: {m["max_drawdown"]:.1f}%)')
        peak = h['capital'].cummax()
        dd   = (h['capital'] - peak) / peak * 100
        ax2.fill_between(h.index, dd.values, 0, color=ROJO, alpha=0.4)
        ax2.plot(h.index, dd.values, color=ROJO, linewidth=0.8)
        ax2.set_ylabel('Drawdown (%)', color=GRIS, fontsize=9)

        # ── 3. Posiciones en el tiempo ────────────────────────
        ax3 = fig.add_subplot(gs[1, 2])
        estilo(ax3, 'Distribución de Posiciones')
        pos_counts = h['posicion'].value_counts()
        labels_pos = {-1: 'VENTA', 0: 'NEUTRAL', 1: 'COMPRA'}
        colores_pos = {-1: ROJO, 0: GRIS, 1: VERDE}
        vals   = [pos_counts.get(k, 0) for k in [-1, 0, 1]]
        etiq   = [labels_pos[k] for k in [-1, 0, 1]]
        colores = [colores_pos[k] for k in [-1, 0, 1]]
        bars = ax3.bar(etiq, vals, color=colores, alpha=0.8)
        for bar, v in zip(bars, vals):
            ax3.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, str(v),
                     ha='center', va='bottom', color='white', fontsize=8)
        ax3.set_ylabel('Días', color=GRIS, fontsize=9)

        # ── 4. Importancia de features ────────────────────────
        ax4 = fig.add_subplot(gs[2, :2])
        estilo(ax4, 'Importancia de Features (Econofísica + Técnicas)')
        imp = self.modelo.importancia_features()
        if not imp.empty:
            top = imp.head(12)
            colores_feat = []
            for feat in top.index:
                if any(x in feat for x in
                       ['hurst','fractal','tsallis','regimen','autocorr']):
                    colores_feat.append(AMA)   # econofísica → amarillo
                else:
                    colores_feat.append(AZUL)  # técnica → azul
            ax4.barh(range(len(top)), top.values,
                     color=colores_feat, alpha=0.8)
            ax4.set_yticks(range(len(top)))
            ax4.set_yticklabels(top.index, fontsize=8)
            ax4.invert_yaxis()
            ax4.set_xlabel('Importancia', color=GRIS, fontsize=8)
            # Leyenda manual
            from matplotlib.patches import Patch
            leyenda = [Patch(color=AMA, label='Econofísica'),
                       Patch(color=AZUL, label='Técnica')]
            ax4.legend(handles=leyenda, facecolor=BGX,
                       labelcolor='white', fontsize=8)

        # ── 5. Métricas resumen ───────────────────────────────
        ax5 = fig.add_subplot(gs[2, 2])
        ax5.set_facecolor(BGX)
        ax5.axis('off')
        metricas_texto = [
            ('MÉTRICAS CLAVE', '', 'white', 11),
            ('', '', 'white', 9),
            ('CAGR',         f"{m['cagr']:+.1f}%",     VERDE if m['cagr']>0 else ROJO, 10),
            ('Sharpe',       f"{m['sharpe_ratio']:.2f}", AZUL, 10),
            ('Sortino',      f"{m['sortino_ratio']:.2f}", AZUL, 10),
            ('Max Drawdown', f"{m['max_drawdown']:.1f}%", ROJO, 10),
            ('Win Rate',     f"{m['win_rate']:.1f}%",    AMA,  10),
            ('Alpha',        f"{m['alpha']:+.1f}%",
             VERDE if m['alpha']>0 else ROJO, 10),
            ('N Operaciones',f"{m['n_operaciones']}",    GRIS, 10),
        ]
        y_pos = 0.95
        for label, valor, color, size in metricas_texto:
            if valor:
                ax5.text(0.05, y_pos, label, color=GRIS,
                         fontsize=size, transform=ax5.transAxes)
                ax5.text(0.75, y_pos, valor, color=color,
                         fontsize=size, fontweight='bold',
                         transform=ax5.transAxes, ha='right')
            else:
                ax5.text(0.05, y_pos, label, color=color,
                         fontsize=size, fontweight='bold',
                         transform=ax5.transAxes)
            y_pos -= 0.10

        fig.suptitle(
            f'BACKTEST ML + ECONOFÍSICA  │  {self.ticker}  │  {self.periodo}',
            color='white', fontsize=13, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        plt.show()


# ══════════════════════════════════════════════════════════════
# EJECUCIÓN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Ejemplo 1: Stock ─────────────────────────────────────
    sistema_stock = SistemaTrading(
        ticker='NVDA',
        periodo='3y',
        modelo_tipo='random_forest',
        label_método='triple_barrier',
        horizonte_label=5,
        umbral_label=0.025,
        umbral_confianza=0.55,
        capital_inicial=10_000,
    )
    sistema_stock.correr()
    sistema_stock.graficar()

    # ── Ejemplo 2: Crypto ────────────────────────────────────
    sistema_btc = SistemaTrading(
        ticker='BTC-USD',
        periodo='3y',
        modelo_tipo='random_forest',
        label_método='triple_barrier',
        horizonte_label=3,      # crypto se mueve más rápido
        umbral_label=0.04,      # umbral más alto por mayor volatilidad
        umbral_confianza=0.58,
        capital_inicial=10_000,
    )
    sistema_btc.correr()
    sistema_btc.graficar()
