"""
╔══════════════════════════════════════════════════════════════════╗
║   ANÁLISIS MACRO / RÉGIMEN DE MERCADO                           ║
║   Curva de tasas · VIX term structure · crédito · dólar        ║
╚══════════════════════════════════════════════════════════════════╝

Los módulos de econofísica analizan activos sueltos. Una mesa mira el SISTEMA:
si la curva de tasas se invierte, si el VIX está en backwardation, si los spreads
de crédito se abren, si el dólar sube. Este módulo arma un "régimen de mercado"
cross-asset a partir de datos públicos (yfinance, sin API keys).

INSTALACIÓN: pip install yfinance numpy pandas matplotlib

NOTA: el spread de crédito es un PROXY de ETFs (HYG vs IEF), no un OAS real.
El OAS de verdad sale de FRED (BAMLH0A0HYM2) y requiere una API key.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:  # Windows consoles are cp1252 and choke on the report glyphs / emoji.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

# ── Cestas de tickers ─────────────────────────────────────────
TASAS = {"^IRX": "3m", "^FVX": "5y", "^TNX": "10y", "^TYX": "30y"}
VOL = {"^VIX9D": "VIX9D", "^VIX": "VIX", "^VIX3M": "VIX3M", "^MOVE": "MOVE"}
OTROS = {
    "DX-Y.NYB": "DXY", "HYG": "HYG", "IEF": "IEF", "LQD": "LQD",
    "SPY": "SPY", "GLD": "GLD", "CL=F": "OIL",
}
TODOS = {**TASAS, **VOL, **OTROS}


@dataclass
class RegimenMacro:
    estado: str            # RISK-ON / NEUTRAL / RISK-OFF
    score: float           # -1 (estrés) .. +1 (apetito de riesgo)
    señales: list = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class AnalisisMacro:
    """Descarga la cesta macro y calcula curva, vol term structure, crédito y régimen."""

    def __init__(self, lookback_dias: int = 400):
        self.lookback = lookback_dias
        self.px: pd.DataFrame | None = None
        self._ok = False

    # ---------------------------------------------------------------- datos
    def cargar(self) -> AnalisisMacro:
        try:
            import yfinance as yf
        except ImportError:
            print("Necesitás yfinance: pip install yfinance")
            return self

        raw = yf.download(
            list(TODOS), period="2y", auto_adjust=True, progress=False
        )
        px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        px = pd.DataFrame(px).rename(columns=TODOS)
        # Drop tickers that came back empty (yfinance occasionally delists an
        # index alias), then forward-fill the rest.
        px = px.dropna(axis=1, how="all").ffill().dropna(how="all")
        if px.empty:
            print("yfinance no devolvió datos.")
            return self
        self.px = px.tail(self.lookback)
        self._ok = True
        return self

    def _ultimo(self, col: str) -> float | None:
        if self.px is None or col not in self.px:
            return None
        s = self.px[col].dropna()
        return float(s.iloc[-1]) if len(s) else None

    # ---------------------------------------------------------------- curva de tasas
    def curva_tasas(self) -> dict:
        y3m, y5, y10, y30 = (self._ultimo(k) for k in ("3m", "5y", "10y", "30y"))
        if None in (y3m, y10):
            return {}
        pendiente = y10 - y3m
        pendiente_2_10 = y10 - y5 if y5 else None
        curvatura = (2 * y5 - y3m - y30) if (y5 and y30) else None
        invertida = pendiente < 0
        return {
            "nivel_10y": round(y10, 2),
            "pendiente_10y_3m": round(pendiente, 2),
            "pendiente_10y_5y": round(pendiente_2_10, 2) if pendiente_2_10 is not None else None,
            "curvatura": round(curvatura, 2) if curvatura is not None else None,
            "invertida": invertida,
            "lectura": (
                "curva INVERTIDA — señal recesiva clásica" if invertida
                else "curva plana — mercado dividido" if pendiente < 0.5
                else "curva con pendiente positiva — expansión"
            ),
        }

    # ---------------------------------------------------------------- VIX term structure
    def term_structure_vix(self) -> dict:
        vix, vix3m, vix9d, move = (self._ultimo(k) for k in ("VIX", "VIX3M", "VIX9D", "MOVE"))
        if None in (vix, vix3m):
            return {}
        ratio = vix / vix3m
        # ratio < 1: contango (calma) — el mercado espera menos vol a futuro.
        # ratio > 1: backwardation (estrés) — pánico ahora, se espera que baje.
        estado = "BACKWARDATION (estrés)" if ratio > 1.0 else "CONTANGO (calma)"
        return {
            "vix": round(vix, 2),
            "vix3m": round(vix3m, 2),
            "vix9d": round(vix9d, 2) if vix9d else None,
            "ratio_vix_vix3m": round(ratio, 3),
            "estado": estado,
            "move": round(move, 1) if move else None,
            "lectura": (
                f"{estado} — "
                + ("cobertura cara, típico de un techo de estrés" if ratio > 1.0
                   else "condiciones normales, sin pánico a corto plazo")
            ),
        }

    # ---------------------------------------------------------------- crédito (proxy)
    def spread_credito(self) -> dict:
        if self.px is None or "HYG" not in self.px or "IEF" not in self.px:
            return {}
        rel = np.log(self.px["HYG"] / self.px["IEF"]).dropna()
        # Retorno relativo HY vs duración: cae cuando el crédito se estresa.
        rel_ret = rel.diff()
        r20 = float(rel_ret.tail(20).sum())
        r60 = float(rel_ret.tail(60).sum())
        # percentil del cambio a 20d en la ventana disponible
        roll20 = rel_ret.rolling(20).sum().dropna()
        pctl = float((roll20 < r20).mean()) if len(roll20) else np.nan
        return {
            "hy_vs_ief_20d_pct": round(r20 * 100, 2),
            "hy_vs_ief_60d_pct": round(r60 * 100, 2),
            "percentil_historico": round(pctl, 2),
            "lectura": (
                "crédito HY bajo estrés (percentil bajo) — apetito de riesgo cayendo"
                if pctl < 0.20 else
                "crédito HY fuerte — apetito de riesgo saludable"
                if pctl > 0.80 else
                "crédito HY en rango normal"
            ),
            "nota": "proxy de ETF (HYG vs IEF), no un OAS real",
        }

    # ---------------------------------------------------------------- dólar
    def dolar(self) -> dict:
        if self.px is None or "DXY" not in self.px:
            return {}
        dxy = self.px["DXY"].dropna()
        nivel = float(dxy.iloc[-1])
        ma50 = float(dxy.tail(50).mean())
        tendencia = "SUBIENDO (risk-off)" if nivel > ma50 * 1.005 else \
                    "BAJANDO (risk-on)" if nivel < ma50 * 0.995 else "LATERAL"
        return {
            "nivel": round(nivel, 2),
            "vs_ma50_pct": round((nivel / ma50 - 1) * 100, 2),
            "tendencia": tendencia,
        }

    # ---------------------------------------------------------------- correlaciones
    def correlaciones_cross_asset(self, ventana: int = 60) -> dict:
        if self.px is None:
            return {}
        rets = np.log(self.px / self.px.shift(1))
        out = {}
        for a, b, etiqueta in [("SPY", "IEF", "acciones-bonos"),
                               ("SPY", "GLD", "acciones-oro"),
                               ("SPY", "DXY", "acciones-dólar")]:
            if a in rets and b in rets:
                pair = rets[[a, b]].dropna().tail(ventana)
                if len(pair) >= 20:
                    out[etiqueta] = round(float(pair[a].corr(pair[b])), 2)
        return out

    # ---------------------------------------------------------------- régimen compuesto
    def regimen_macro(self) -> RegimenMacro:
        score, señales = 0.0, []

        vts = self.term_structure_vix()
        if vts:
            if vts["ratio_vix_vix3m"] > 1.0:
                score -= 2; señales.append(f"[-] VIX en backwardation ({vts['ratio_vix_vix3m']:.2f})")
            elif vts["vix"] > 25:
                score -= 1; señales.append(f"[-] VIX elevado ({vts['vix']:.0f})")
            else:
                score += 1; señales.append(f"[+] VIX en contango ({vts['vix']:.0f})")
            if vts.get("move") and vts["move"] > 120:
                score -= 1; señales.append(f"[-] MOVE alto ({vts['move']:.0f}): estrés en bonos")

        cr = self.spread_credito()
        if cr and not np.isnan(cr["percentil_historico"]):
            if cr["percentil_historico"] < 0.20:
                score -= 2; señales.append("[-] Crédito HY estresado")
            elif cr["percentil_historico"] > 0.70:
                score += 1; señales.append("[+] Crédito HY fuerte")

        cv = self.curva_tasas()
        if cv:
            if cv["invertida"]:
                score -= 1; señales.append("[-] Curva de tasas invertida")
            else:
                score += 0.5; señales.append("[+] Curva con pendiente positiva")

        dl = self.dolar()
        if dl:
            if dl["tendencia"].startswith("SUBIENDO"):
                score -= 1; señales.append("[-] Dólar subiendo (risk-off)")
            elif dl["tendencia"].startswith("BAJANDO"):
                score += 0.5; señales.append("[+] Dólar débil (risk-on)")

        norm = float(np.clip(score / 5.0, -1, 1))
        estado = "RISK-OFF" if norm < -0.3 else "RISK-ON" if norm > 0.3 else "NEUTRAL"
        return RegimenMacro(estado=estado, score=round(norm, 2), señales=señales)

    # ---------------------------------------------------------------- reporte
    def to_dict(self) -> dict:
        reg = self.regimen_macro()
        return {
            "timestamp": datetime.now().isoformat(),
            "disponible": self._ok,
            "curva_tasas": self.curva_tasas(),
            "vix_term_structure": self.term_structure_vix(),
            "credito": self.spread_credito(),
            "dolar": self.dolar(),
            "correlaciones": self.correlaciones_cross_asset(),
            "regimen": {"estado": reg.estado, "score": reg.score, "señales": reg.señales},
        }

    def imprimir_reporte(self) -> None:
        if not self._ok:
            print("Sin datos — ejecutá cargar() con conexión a internet.")
            return
        sep = "─" * 60
        print(f"\n{'═' * 60}\n  ANÁLISIS MACRO — {datetime.now():%Y-%m-%d %H:%M}\n{'═' * 60}")

        cv = self.curva_tasas()
        if cv:
            print(f"\n CURVA DE TASAS\n{sep}")
            print(f"  10y: {cv['nivel_10y']}%   pendiente 10y-3m: {cv['pendiente_10y_3m']:+.2f}pp"
                  f"   curvatura: {cv['curvatura']}")
            print(f"  → {cv['lectura']}")

        vts = self.term_structure_vix()
        if vts:
            print(f"\n VOLATILIDAD (term structure)\n{sep}")
            print(f"  VIX {vts['vix']}  │  VIX3M {vts['vix3m']}  │  ratio {vts['ratio_vix_vix3m']:.3f}"
                  f"  │  MOVE {vts['move']}")
            print(f"  → {vts['lectura']}")

        cr = self.spread_credito()
        if cr:
            print(f"\n CRÉDITO (proxy HYG vs IEF)\n{sep}")
            print(f"  20d: {cr['hy_vs_ief_20d_pct']:+.2f}%   60d: {cr['hy_vs_ief_60d_pct']:+.2f}%"
                  f"   percentil: {cr['percentil_historico']:.0%}")
            print(f"  → {cr['lectura']}")

        dl = self.dolar()
        if dl:
            print(f"\n DÓLAR (DXY)\n{sep}")
            print(f"  {dl['nivel']}  ({dl['vs_ma50_pct']:+.2f}% vs MA50)  → {dl['tendencia']}")

        corr = self.correlaciones_cross_asset()
        if corr:
            print(f"\n CORRELACIONES 60d\n{sep}")
            for k, v in corr.items():
                print(f"  {k:<18} {v:+.2f}")

        reg = self.regimen_macro()
        print(f"\n RÉGIMEN DE MERCADO\n{sep}")
        for s in reg.señales:
            print(f"  {s}")
        print(f"\n  ▶ {reg.estado}  (score {reg.score:+.2f})")
        print(f"{'═' * 60}\n")

    def graficar(self) -> None:
        if not self._ok:
            return
        import matplotlib.pyplot as plt

        BG, BGX, AZ, RJ, GR = "#0d1117", "#161b22", "#58a6ff", "#f78166", "#8b949e"
        fig, axes = plt.subplots(2, 2, figsize=(15, 9))
        fig.patch.set_facecolor(BG)
        for ax in axes.flat:
            ax.set_facecolor(BGX); ax.tick_params(colors=GR)
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363d")

        rets = np.log(self.px / self.px.shift(1))

        # Curva de tasas hoy
        ax = axes[0, 0]
        pts = [(0.25, "3m"), (5, "5y"), (10, "10y"), (30, "30y")]
        xs = [p[0] for p in pts if p[1] in self.px]
        ys = [self._ultimo(p[1]) for p in pts if p[1] in self.px]
        ax.plot(xs, ys, "o-", color=AZ, lw=1.6)
        ax.set_title("Curva de tasas (hoy)", color="white", fontsize=10)
        ax.set_xlabel("plazo (años)", color=GR); ax.set_ylabel("%", color=GR)

        # VIX vs VIX3M
        ax = axes[0, 1]
        if "VIX" in self.px and "VIX3M" in self.px:
            ax.plot(self.px.index, self.px["VIX"], color=RJ, lw=1.0, label="VIX")
            ax.plot(self.px.index, self.px["VIX3M"], color=AZ, lw=1.0, label="VIX3M")
            ax.legend(facecolor=BGX, labelcolor="white", fontsize=8)
        ax.set_title("VIX term structure", color="white", fontsize=10)

        # Crédito proxy
        ax = axes[1, 0]
        if "HYG" in self.px and "IEF" in self.px:
            rel = np.log(self.px["HYG"] / self.px["IEF"])
            ax.plot(rel.index, (rel - rel.iloc[0]) * 100, color=AZ, lw=1.2)
            ax.axhline(0, color=GR, lw=0.6)
        ax.set_title("Crédito HY vs duración (proxy, acum. %)", color="white", fontsize=10)

        # Correlación acciones-bonos rolling
        ax = axes[1, 1]
        if "SPY" in rets and "IEF" in rets:
            c = rets["SPY"].rolling(60).corr(rets["IEF"])
            ax.plot(c.index, c, color=AZ, lw=1.2)
            ax.axhline(0, color=GR, lw=0.6)
        ax.set_title("Correlación acciones-bonos (60d)", color="white", fontsize=10)

        fig.suptitle("ANÁLISIS MACRO / RÉGIMEN DE MERCADO", color="white",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def analizar(self) -> AnalisisMacro:
        self.cargar()
        self.imprimir_reporte()
        if self._ok:
            self.graficar()
        return self


if __name__ == "__main__":
    AnalisisMacro().analizar()
