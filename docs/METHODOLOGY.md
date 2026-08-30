# Metodología

Documento de referencia del framework de riesgo de mercado y análisis
cuantitativo de `econophysics-ml`. Para el panorama general y cómo correr cada
módulo, ver el [README](../README.md).

> Proyecto educativo y de investigación. Las fórmulas y convenciones acá siguen
> las de un desk de riesgo de mercado; las implementaciones son propias y
> verificadas con tests de propiedades, no con datos de una corrida anterior.

---

## Índice

1. [Alcance y filosofía](#1-alcance-y-filosofía)
2. [Datos y convenciones](#2-datos-y-convenciones)
3. [Estimadores de VaR](#3-estimadores-de-var)
4. [Volatilidad condicional y filtered historical simulation](#4-volatilidad-condicional-y-filtered-historical-simulation)
5. [Expected Shortfall](#5-expected-shortfall)
6. [Extreme Value Theory](#6-extreme-value-theory)
7. [Backtesting supervisor](#7-backtesting-supervisor)
8. [Semáforo de Basilea](#8-semáforo-de-basilea)
9. [Stress testing](#9-stress-testing)
10. [Modelo de factores](#10-modelo-de-factores)
11. [Rigor de backtest: PSR, DSR, PBO, purged CV](#11-rigor-de-backtest)
12. [Microestructura de cripto](#12-microestructura-de-cripto)
13. [Econofísica](#13-econofísica)
14. [Limitaciones conocidas](#14-limitaciones-conocidas)
15. [Referencias](#15-referencias)

---

## 1. Alcance y filosofía

**Los mercados no son gaussianos ni eficientes.** Tienen memoria (persistencia y
reversión), colas gruesas (eventos extremos mucho más frecuentes de lo que
predice la normal) y volatilidad que se agrupa en rachas. Todo lo que sigue parte
de ahí.

El flujo de trabajo de riesgo es cuatro pasos:

1. **Medir** — VaR, Expected Shortfall, por varios métodos que discrepan a
   propósito.
2. **Validar** — backtest walk-forward + tests de hipótesis (Kupiec,
   Christoffersen, Acerbi-Székely). Un número de riesgo que no está calibrado es
   peor que ninguno.
3. **Estresar** — escenarios concretos (2008, COVID, shock de tasas, colapsos
   cripto), porque el VaR no dice nada sobre un evento específico.
4. **Contextualizar** — descomponer el riesgo en factores conocidos y ubicar la
   cartera en el régimen macro.

El alcance es **riesgo de mercado sobre instrumentos lineales** (acciones,
cripto, ETFs). Quedan fuera: derivados con convexidad (Greeks), riesgo de
contraparte (CVA/PFE), riesgo de crédito (PD/LGD/EAD), IRRBB y liquidez de
fondeo.

---

## 2. Datos y convenciones

| Convención | Elección | Por qué |
|---|---|---|
| Retornos | logarítmicos `ln(P_t / P_{t-1})` | aditivos en el tiempo → escalado de horizonte coherente. No aditivos entre activos: la cartera de log-returns es una aproximación, aceptable para datos diarios |
| Pesos de cartera | fijos (rebalanceo diario implícito) | estándar de reporte de riesgo de mercado; un libro buy-and-hold derivaría |
| Signo del VaR | **número positivo = pérdida** | "VaR 99% = 1.9%" significa que 1 día de cada 100 se pierde más de 1.9% |
| Días hábiles / año | 252 | convención de anualización |

**Fuentes.** `data_layer.py` unifica Binance REST (spot y perpetuos), CoinGecko
(sentimiento cripto) y yfinance (acciones, ETFs, cripto de largo plazo, índices
macro). Selección: cripto de corto/mediano plazo → Binance (sin delay); cripto de
largo plazo y todo lo demás → yfinance. Los factores de Fama-French vienen de la
Ken French Data Library.

**Datos sintéticos.** `varengine.simulate_market` genera precios de un proceso
GARCH(1,1) con innovaciones Student-t (ν = 4.5) correlacionadas. Reproduce
agrupamiento de volatilidad y colas gruesas con parámetros conocidos, para que
los tests corran offline y de forma determinista, y para que el "hallazgo" del
README sea reproducible.

---

## 3. Estimadores de VaR

El VaR al nivel `c` y horizonte `h` es la pérdida que no se supera con
probabilidad `c`. Cinco métodos:

### 3.1. Histórico

`VaR = −Q_{1−c}(r)`, el cuantil empírico de los retornos realizados. Sin supuesto
distribucional: colas y asimetría reales vienen incluidas. **Limitación:** no
puede producir una pérdida que nunca ocurrió, y pesa igual una crisis de hace dos
años que ayer. A 99% con 250 observaciones, solo ~2.5 caen más allá del umbral →
el estimador depende de 2-3 puntos.

### 3.2. Paramétrico — normal

`VaR = −(μ + z_α · σ)` con `z_α = Φ⁻¹(1−c)`. Rápido, suave, analíticamente
descomponible. **Es el que usaban los bancos antes de 2008 y el que falló
entonces:** subestima el riesgo de cola de forma sistemática porque los retornos
reales tienen mucha más masa en los extremos que una gaussiana.

ES en forma cerrada: `ES = −(μ − σ · φ(z_α) / α)`.

### 3.3. Paramétrico — Student-t

Misma forma, pero el cuantil sale de una t con ν grados de libertad ajustados por
máxima verosimilitud, y la escala se corrige para que la varianza de la t iguale
la muestral (`σ_t = √(ν/(ν−2))`). Con los ν típicos del equity diario (3-6), el
cuantil 99% queda materialmente más afuera que el gaussiano. **Limitación:** ν es
inestable en ventanas cortas — puede dar valores < 2 (donde la varianza no
existe); el código lo acota en 2.05 y lo registra en la nota del resultado.

### 3.4. Monte Carlo

Estima el vector de medias y la matriz de covarianza de los activos, simula
`n_sim` trayectorias conjuntas sobre el horizonte, agrega a nivel cartera y lee
el cuantil empírico. Bajo normalidad multivariada y horizonte de 1 día converge
al paramétrico — un chequeo gratis de que la plomería está bien. El camino
Student-t usa una **cópula gaussiana con marginales t**, así que la dependencia
cross-asset queda lineal y cada margen carga colas gruesas. Guarda un ajuste
nearest-PSD sobre la covarianza (Cholesky rompería con una matriz ligeramente
indefinida por error numérico).

### 3.5. Filtered Historical Simulation

Ver [sección 4](#4-volatilidad-condicional-y-filtered-historical-simulation).

### Escalado raíz-del-tiempo

`VaR_h = VaR_1 · √h`. Asume retornos i.i.d. — **falso**: la volatilidad se
agrupa, así que subestima el riesgo en períodos turbulentos y lo sobreestima en
los calmos. Basilea lo permite, que es la única razón por la que es estándar. El
GARCH da un pronóstico multi-día que revierte a la media y es algo mejor.

---

## 4. Volatilidad condicional y filtered historical simulation

Los métodos 3.1–3.4 asumen que la distribución de retornos es estable sobre la
ventana. No lo es. Un σ estático de 250 días queda **alto después de un tramo
calmo y bajo justo cuando llega el shock**. Esto se ve en un backtest como
excepciones agrupadas (Christoffersen) y, seguido, como exceso de excepciones
(Kupiec).

### 4.1. EWMA (RiskMetrics)

`σ²_t = λ · σ²_{t−1} + (1−λ) · r²_{t−1}`, con `λ = 0.94` por convención diaria
(memoria efectiva ~33 días). Un solo parámetro, sin ajuste. Es un GARCH con
`ω = 0` y `α + β = 1` (IGARCH), por eso no revierte a la media.

### 4.2. GARCH(1,1)

`σ²_t = ω + α · ε²_{t−1} + β · σ²_{t−1}`, tres parámetros por MLE (innovación
normal o t). La media se trata como constante y se remueve primero. Se impone
`α + β < 1` (estacionariedad). Revierte a la varianza de largo plazo
`σ²_∞ = ω / (1 − α − β)`, así que el pronóstico a k días es
`E[σ²_{t+k}] = σ²_∞ + (α+β)^{k−1} · (σ²_{t+1} − σ²_∞)`.

### 4.3. FHS

1. Estimar el σ condicional (EWMA o GARCH).
2. Estandarizar: `z_t = r_t / σ_t`. Si el modelo de vol es bueno, los `z` quedan
   casi i.i.d. — el agrupamiento se dividió.
3. `VaR = −Q_{1−c}(z) · σ_{T+1}`, con `σ_{T+1}` el pronóstico de un paso.

La **forma** de la cola viene de toda la historia (sin supuesto distribucional);
su **escala** es la volatilidad de hoy. En el hallazgo del README, la FHS-EWMA
clava la tasa de excepciones en 1.10% y pasa todos los tests, donde el
paramétrico normal caía en zona amarilla.

El backtest rodante usa **EWMA-FHS** (sin ajuste, rápido). El GARCH-FHS queda
como estimación puntual: refit por ventana sería lento.

---

## 5. Expected Shortfall

`ES_c = E[−r | −r > VaR_c]` — la pérdida promedio *una vez cruzado* el umbral.

Basilea III / FRTB **reemplazó el VaR por el ES al 97.5%** como medida
regulatoria de riesgo de mercado:

- El ES es **subaditivo** (coherente): nunca dice que una cartera diversificada
  es más riesgosa que la suma de sus partes. El VaR sí puede.
- El ES mira *dentro* de la cola; el VaR es ciego a qué tan mala es la pérdida
  una vez superado el umbral.
- El nivel 97.5% está calibrado para ser comparable en severidad al VaR 99% bajo
  una normal, y estrictamente más conservador con colas gruesas.

---

## 6. Extreme Value Theory

A 99.5% o 99.9% de confianza la simulación histórica se queda sin datos. El
teorema de **Pickands-Balkema-de Haan** dice que los excesos sobre un umbral alto
`u` convergen a una **distribución de Pareto generalizada (GPD)** con parámetros
de forma `ξ` y escala `β`, sea cual sea la distribución madre.

**Peaks-over-threshold:** `u` = cuantil 95% de las pérdidas; se ajusta la GPD por
MLE a los excesos `(pérdida − u | pérdida > u)`. Luego, en forma cerrada:

```
VaR_c = u + (β/ξ) · [ (n/N_u · (1−c))^{−ξ} − 1 ]
ES_c  = VaR_c/(1−ξ) + (β − ξ·u)/(1−ξ)          (válido para ξ < 1)
```

El parámetro `ξ` es toda la historia:

- `ξ > 0` — cola pesada (ley de potencia). Índice de cola `1/ξ`; los momentos de
  orden ≥ `1/ξ` no existen. Equity diario: `ξ ≈ 0.2–0.3`.
- `ξ = 0` — cola exponencial (la normal cae acá).
- `ξ < 0` — cola acotada (hay una pérdida máxima dura).

---

## 7. Backtesting supervisor

Producir un VaR es fácil. Demostrar que está **calibrado** es lo que exige el
regulador. Todo se hace con **validación walk-forward**: en cada fecha el VaR se
estima con las 250 observaciones previas y se compara con el retorno que
efectivamente siguió. Estimar sobre toda la muestra y testear sobre la misma
filtra el futuro.

### 7.1. Kupiec — proportion of failures (cobertura incondicional)

Bajo H0 (modelo bien calibrado), las excepciones son Bernoulli con `p = 1 − c`.
Razón de verosimilitud:

```
LR_pof = −2 · ln[ (1−p)^{T−N} · p^N  /  (1−N/T)^{T−N} · (N/T)^N ]  ~  χ²(1)
```

Con `T` días y `N` excepciones. Rechaza si `LR_pof > χ²_{0.95}(1) = 3.84`.
Detecta si el **número** de excepciones es el correcto — no si están agrupadas.

### 7.2. Christoffersen — independencia

Ajusta una cadena de Markov de primer orden a la secuencia de excepciones y
testea si `P(excepción | ayer hubo) = P(excepción | ayer no hubo)`:

```
LR_ind = −2 · ln[ L(π) / L(π_01, π_11) ]  ~  χ²(1)
```

Rechazo ⇒ las excepciones **se agrupan** — la firma de un modelo con volatilidad
estática contra retornos que se agrupan. Un modelo puede tener el conteo exacto
esperado y aun así ser inservible si todas caen en la misma semana. El test suite
tiene un caso construido: dos secuencias con idéntico conteo, Kupiec no las
distingue, Christoffersen rechaza la agrupada.

### 7.3. Cobertura condicional

`LR_cc = LR_pof + LR_ind ~ χ²(2)`. El test conjunto: un modelo debe pasar los dos.

### 7.4. Acerbi-Székely — backtest del Expected Shortfall

Kupiec y Christoffersen solo miran *si* el VaR fue superado; son ciegos a
*cuánto*. Como FRTB hizo del ES la medida regulatoria, hay que validar el ES.
Estadístico Z2 (Test 2 de Acerbi-Székely 2014):

```
Z2 = 1 + (1/(T·α)) · Σ_t [ r_t · 1{r_t < −VaR_t} / ES_t ]
```

`r_t` es negativo en pérdida y `ES_t` es una pérdida positiva, así que bajo un ES
bien calibrado `E[Z2] = 0`. **`Z2 < 0` ⇒ las pérdidas más allá del VaR fueron
peores de lo que el ES decía** — el modelo subestima la severidad de la cola. La
significancia sale de un bootstrap de la distribución muestral de Z2 (remuestreo
de días completos): se rechaza si esa distribución queda confiablemente por
debajo de 0.

En el hallazgo del README, **la simulación histórica pasa Kupiec pero falla acá**
— aprobada por el semáforo, mal calibrada en lo que importa.

---

## 8. Semáforo de Basilea

Regla supervisora que convierte el conteo de excepciones sobre 250 días hábiles
en una penalización de capital. El incremento se suma al factor base de 3:

| Zona | Excepciones / 250 días | Multiplicador | Lectura |
|---|---|---|---|
| Verde | 0–4 | 3.00 | modelo aceptado, sin penalización |
| Amarilla | 5 → 9 | 3.40 → 3.85 | bajo revisión, más capital regulatorio |
| Roja | ≥ 10 | 4.00 | modelo rechazado, remediación obligatoria |

Ventanas distintas de 250 días se escalan proporcionalmente. **Esto es lo que un
equipo de riesgos mira antes que cualquier p-valor:** convierte un supuesto
estadístico en un costo concreto.

---

## 9. Stress testing

El VaR y el ES responden "¿qué tan malo es un día malo *normal*?". No dicen nada
de un evento concreto porque ese vive donde no hay datos para ajustar. El stress
pregunta: *dado este set exacto de movimientos, ¿qué le pasa a la cartera?*

**Mapeo a factores.** La cartera se regresa (OLS, ~500 días recientes) sobre un
set chico de factores tradables: `SPY` (equity), `IEF` (tasas), `HYG` (crédito),
`DX-Y.NYB` (dólar), `GLD` (oro), `CL=F` (petróleo), `BTC-USD` (cripto). Los betas
son las exposiciones.

### 9.1. Replay histórico

Ventanas de crisis: GFC 2008, China 2015, Volmageddon 2018, Q4-2018, COVID
marzo-2020, shock de tasas 2022, **Terra/UST mayo-2022, 3AC junio-2022, FTX
noviembre-2022**. Si todos los activos de la cartera tienen historia en la
ventana, se aplican los **retornos reales de los activos** (el supuesto de
linealidad desaparece); si no, `P&L = betas · (retorno acumulado de los factores
en la ventana)`.

### 9.2. Shocks hipotéticos

Sets de movimientos de factores definidos a mano: `equity_crash` (equity −30%,
crédito −15%, oil −20%, oro +8%), `rates_shock`, `risk_off`, `stagflation`,
`crypto_winter`, `stablecoin_depeg` (cripto −45%), `exchange_insolvency`
(cripto −30%). Única forma de estresar algo que nunca pasó. `P&L = betas · shock`.

### 9.3. Reverse stress

Se fija la pérdida primero ("¿qué nos costaría el 10% del libro?") y se resuelve
el escalado del shock en una dirección dada que la produce. Devuelve los
movimientos de factores implícitos. Suele revelar el escenario que no estabas
mirando.

---

## 10. Modelo de factores

Un banco nunca reporta riesgo de equity sin descomponerlo en factores. La mayor
parte del retorno de una cartera no es habilidad — es **exposición**: al mercado,
a small caps, a acciones baratas, a ganadores recientes, a calidad, a baja
volatilidad. Esas exposiciones están priceadas y spanned por productos indexados
baratos.

**Fama-French 5 + momentum** (Ken French Data Library, diario, desde 1963):
`Mkt-RF, SMB` (size), `HML` (value), `RMW` (profitability/quality),
`CMA` (investment), `Mom`. Alternativa práctica: factores de estilo por ETF
(`VLUE, MTUM, QUAL, USMV, IWM` menos `SPY`).

**Regresión:** `r_cartera − RF = α + Σ_k β_k · f_k + ε`.

- `β_k` — exposición a cada factor, con t-stat.
- `α` anualizado + t-stat — **lo que sobrevive a las exposiciones conocidas.** Si
  `|t| < 2`, el "alpha" no es distinguible de ruido.
- `R²` — fracción de varianza explicada por los factores.
- **Descomposición de riesgo:** `Var(r) ≈ β' Σ_f β + Var(ε)`. La contribución del
  factor `k` es `β_k · (Σ_f β)_k` (suma a la varianza sistémica; puede ser
  negativa si el factor cubre). El resto es **% idiosincrático**.
- **Atribución de retorno:** `β_k · E[f_k] · 252`.

Ejemplo real (AAPL, 5 años): market beta ≈ 1.2 (t ≈ 36), `HML` ≈ −0.34 (growth),
`RMW` ≈ +0.46 (long calidad), alpha ≈ 2.6%/año con t ≈ 0.3 → **no significativo**;
41% idiosincrático.

---

## 11. Rigor de backtest

*"Tu backtest miente. ¿Cómo lo sabés?"* Cada parámetro que probaste, cada fecha de
inicio que moviste, cada feature que agregaste y dejaste porque "ayudaba" es
selección, y la selección convierte ruido en un Sharpe. Fórmulas de Bailey &
López de Prado.

### 11.1. Probabilistic Sharpe Ratio (PSR)

`PSR(SR*) = P(SR_real > SR*)` dado el estimador, el largo de muestra y los momentos:

```
PSR(SR*) = Φ( (ŜR − SR*) · √(n−1) / √(1 − γ₃·ŜR + (γ₄−1)/4·ŜR²) )
```

con `γ₃` asimetría y `γ₄` curtosis (no en exceso). Un Sharpe de 2.0 sobre 40
observaciones con skew negativo no es la misma evidencia que 2.0 sobre 2000
casi-normales, y el PSR lo dice.

### 11.2. Deflated Sharpe Ratio (DSR)

PSR contra un benchmark que **no es cero** sino el Sharpe que se esperaría *por
azar* tras `N` pruebas independientes:

```
SR* = √Var(SR_trials) · [ (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]
```

con `γ` la constante de Euler-Mascheroni. Una estrategia de puro ruido probada
1000 veces da DSR ≈ 0. Veredicto: `DSR ≥ 0.95` creíble; `≥ 0.50` inconcluso;
`< 0.50` probablemente sobreajustado.

### 11.3. Probability of Backtest Overfitting (PBO)

Combinatorial symmetric cross-validation. Dada una matriz `T × N` de P&L por
configuración: se corta el timeline en `S` bloques; para cada forma de elegir la
mitad como in-sample:

1. rankear los trials por Sharpe IS, tomar el mejor,
2. su rango out-of-sample → `ω ∈ (0,1)`,
3. logit `λ = ln(ω/(1−ω))`.

`PBO = P(λ < 0)` — la fracción de particiones donde el ganador IS quedó en la
*peor* mitad OOS. ~0.5 = la selección no aporta información; alto = el backtest
está sobreajustado.

### 11.4. Purged K-Fold

Con etiquetas triple-barrera que se resuelven en N días, una observación de train
pegada al test comparte información. El **purge** saca del train las
observaciones cuyo horizonte de etiqueta pisa el test; el **embargo** saca una
franja después. El pipeline ML aplica un purge = horizonte de la barrera en su
walk-forward.

---

## 12. Microestructura de cripto

De Binance Futures (perpetuos), sin autenticación:

| Métrica | Qué es | Lectura |
|---|---|---|
| **Funding rate** | lo que los longs pagan a los shorts cada 8h | positivo y alto (anualizado > ~10%) = demanda de apalancamiento long saturada → contrarian bajista |
| **Basis** | `(mark − index) / index` del perpetuo | contango (positivo) = apalancamiento long; backwardation (negativo) = presión short / miedo |
| **Open interest** | contratos abiertos | OI ↑ con precio ↑ = dinero nuevo apalancado; OI ↓ fuerte = liquidaciones / deleveraging |
| **Long/short ratio** | cuentas long vs short en Binance | > 1.5 sesgo largo (crowded); < 0.8 sesgo corto |

La lectura combinada busca extremos: funding alto + OI subiendo = riesgo de long
squeeze; funding negativo = posible short squeeze; OI cayendo fuerte =
deleveraging en curso.

---

## 13. Econofísica

Métodos de mecánica estadística aplicados a series financieras. Se usan como
features del modelo ML y como señales de los agentes.

- **Exponente de Hurst (R/S, DFA).** Memoria de largo plazo. `H > 0.5` persistente
  (trend following), `H < 0.5` anti-persistente (reversión a la media), `H ≈ 0.5`
  ruido. DFA quita la tendencia local de cada ventana → más estable que R/S.
- **Dimensión fractal.** `D = 2 − H`: rugosidad de la trayectoria del precio.
- **Entropía de Tsallis / estadística-q.** La q-gaussiana interpola entre la
  normal (`q → 1`) y colas de ley de potencia (`q > 1`). `q ≈ 1.5–1.7` = cola
  gruesa fuerte (cripto).
- **LPPLS (Sornette).** Burbuja = crecimiento super-exponencial con oscilaciones
  log-periódicas que se aceleran hacia un tiempo crítico `t_c`.
- **RMT (Teoría de Matrices Aleatorias).** Los autovalores de la matriz de
  correlaciones que caen dentro de la banda de Marchenko-Pastur son ruido y se
  filtran; los que la superan son estructura real.
- **Cointegración (Engle-Granger).** Dos activos cointegrados tienen un spread
  estacionario con vida media de reversión finita → base para pairs trading.

---

## 14. Limitaciones conocidas

Un modelo de riesgo cuyas limitaciones no están documentadas es peor que ninguno.

- **Escalado raíz-del-tiempo** asume retornos i.i.d. Subestima el riesgo en
  períodos turbulentos. Basilea lo permite.
- **La simulación histórica** no puede producir una pérdida que nunca vio y pesa
  igual una crisis vieja que ayer. EWMA-FHS resuelve lo segundo, EVT lo primero.
- **GARCH-FHS rodante es lento** (un MLE por ventana). El backtest rodante usa
  EWMA-FHS.
- **Solo instrumentos lineales.** Opciones y payoffs convexos necesitan
  revaluación completa o delta-gamma. La maquinaria Monte Carlo y de stress es la
  base correcta; la capa de instrumentos no está.
- **El stress por betas de factores** asume exposición lineal y estable — bien
  para un primer orden sobre un libro de contado, no para uno con opcionalidad.
- **El spread de crédito del módulo macro es un proxy de ETF** (HYG vs IEF), no un
  OAS real (eso necesita FRED).
- **ν de la Student-t es inestable** en ventanas cortas — el código lo acota en
  2.05 y lo avisa.
- **La covarianza muestral es ruidosa** con muchos activos; se ofrece shrinkage de
  Ledoit-Wolf pero no es el default.
- **El modelo de factores** usa OLS clásico; errores estándar HAC/Newey-West
  serían más honestos con la autocorrelación residual.
- **Riesgo de contraparte, crédito, IRRBB y liquidez de fondeo** están fuera de
  alcance.

---

## 15. Referencias

**Riesgo de mercado y backtesting**
- Kupiec (1995), *Techniques for verifying the accuracy of risk measurement models*
- Christoffersen (1998), *Evaluating interval forecasts*
- Acerbi & Székely (2014), *Back-testing Expected Shortfall*
- Barone-Adesi, Giannopoulos & Vosper (1999), *VaR without correlations…* (FHS)
- McNeil & Frey (2000), *Estimation of tail-related risk measures… an extreme value approach*
- J.P. Morgan/Reuters (1996), *RiskMetrics Technical Document* (EWMA)
- Bollerslev (1986), *Generalized Autoregressive Conditional Heteroskedasticity* (GARCH)
- Ledoit & Wolf (2004), *Honey, I shrunk the sample covariance matrix*
- Comité de Basilea (1996), *Supervisory framework for the use of backtesting*
- Comité de Basilea (2019), *Minimum capital requirements for market risk* (FRTB)

**Factores y rigor de backtest**
- Fama & French (1993, 2015), *Common risk factors…* / *A five-factor asset pricing model*
- Carhart (1997), *On persistence in mutual fund performance* (momentum)
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*
- Bailey, Borwein, López de Prado & Zhu (2017), *The Probability of Backtest Overfitting*
- López de Prado (2018), *Advances in Financial Machine Learning*

**Econofísica**
- Mantegna & Stanley (2000), *An Introduction to Econophysics*
- Sornette (2003), *Why Stock Markets Crash* (LPPLS)
- Tsallis (2009), *Introduction to Nonextensive Statistical Mechanics*
