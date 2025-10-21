# ОПТИМИЗАЦИЯ VOLATILITY MEDIAN (БАЗА)
"""
Volatility Median – гибкая оптимизация (с тайм-фреймом в сетке)
"""

import os, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt

# ---------- путёвка к данным ----------
# DATA_DIR = '/Users/evgenvg/Downloads/'
DATA_DIR = r"G:\My Drive\data_fut"
TICKER   = "RTSM"

# ---------- дефолтные значения ----------
DEFAULTS = dict(
    KATR     = 2.0,
    PerATR   = 14,
    SMA      = 1,
    MinRA    = 0.5,
    FlATR    = 0,
    FlHL     = 0,
    tf       = 15,
    offset   = 0.0, # минут
    mode     = "deviation",   # новый параметр: режим тестирования
    slope_thr= 0.01           # новый параметр: порог наклона
)

# ---------- ЗАДАЙТЕ СЕТКУ ----------------
GRID = dict(
    KATR   = [2, 3, 4, 5, 7, 9, 11],
    PerATR = [5, 20, 80],
    SMA    = None, #[3,10,25],
    MinRA  = [3, 5, 7, 8, 9, 10],
    FlATR  = 0,
    FlHL   = 0,
    tf     = [5],
    offset = [2.0],
    mode   = ["slope"],   # оба режима в сетке
    slope_thr = [0.01, 0.5, 1, 2],    # значения для наклона
)

# --------------------------------------------------------------------------- #
#                               ЗАГРУЗКА ДАННЫХ                               #
# --------------------------------------------------------------------------- #

def load_minute_file(path):
    cols = ["TICKER","PER","DATE","TIME","OPEN","HIGH","LOW","CLOSE","VOL"]
    df   = pd.read_csv(path, names=cols, sep=",", header=0)
    df.columns = [c.strip("<>") for c in df.columns]
    dt = pd.to_datetime(df["DATE"].astype(str) + df["TIME"].astype(str).str.zfill(6),
                        format="%Y%m%d%H%M%S")
    df.index = dt
    return df[["OPEN","HIGH","LOW","CLOSE","VOL"]].astype(float).sort_index()


_df_cache = {}
def get_df_tf(tf):
    if tf not in _df_cache:
        path = os.path.join(DATA_DIR, f"{TICKER}.txt")
        df_min = load_minute_file(path)
        if tf == 1:
            _df_cache[tf] = df_min
        else:
            rule = f"{tf}min"  # <--- вот тут замени T на min!
            _df_cache[tf] = df_min.resample(rule, label="right", closed="right").agg({
                "OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last", "VOL": "sum"
            }).dropna()
    date_from = '2024-12-25'
    date_to   = '2025-08-29'
    df = _df_cache[tf]
    return df.loc[date_from:date_to]


# --------------------------------------------------------------------------- #
#                            VOLATILITY  MEDIAN                               #
# --------------------------------------------------------------------------- #

def volatility_median(df, KATR, PerATR, SMA, MinRA, FlATR, FlHL):
    h,l,c = df["HIGH"].values, df["LOW"].values, df["CLOSE"].values
    n = len(df)
    tr = np.empty(n); tr[:] = np.nan
    tr[1:] = np.maximum.reduce([h[1:]-l[1:],
                                np.abs(h[1:]-c[:-1]),
                                np.abs(l[1:]-c[:-1])])
    atr = (pd.Series(tr).rolling(PerATR,PerATR).mean() if FlATR==0
           else pd.Series(tr).rolling(PerATR,PerATR).max()).to_numpy()
    price  = (h+l)/2 if FlHL==0 else (h+l+c)/3
    offset = np.maximum(KATR*atr, MinRA)
    vr = np.empty(n); vr[:] = np.nan
    for i in range(n):
        if np.isnan(offset[i]): continue
        if np.isnan(vr[i-1] if i else np.nan):
            vr[i] = price[i]-offset[i]
        else:
            prev = vr[i-1]
            vr[i] = max(prev, price[i]-offset[i]) if price[i]>prev else min(prev, price[i]+offset[i])
    return pd.Series(vr, index=df.index).rolling(SMA,SMA).mean()

# --------------------------------------------------------------------------- #
#                             BACK-TEST  и  МЕТРИКИ                           #
# --------------------------------------------------------------------------- #



def backtest(df, vm, offset=0.0, mode="deviation", slope_thr=0.01):
    price = df["CLOSE"]
    if mode == "deviation":
        raw_pos = np.where(price > vm + offset, 1,
                   np.where(price < vm - offset, -1, np.nan))
    elif mode == "slope":
        slope = vm.diff()
        raw_pos = np.where(slope > slope_thr, 1,
                   np.where(slope < -slope_thr, -1, np.nan))
    else:
        raise ValueError(f"Unknown mode: {mode}")
    pos = pd.Series(raw_pos, index=df.index).ffill().shift(1).fillna(0)
    # --- тут считаем перевороты ---
    n_flip = (pos.diff().abs() == 2).sum()
    ret    = price.pct_change().fillna(0)
    strat  = pos * ret
    equity = (1 + strat).cumprod()
    total  = equity.iat[-1] - 1
    dd     = 1 - equity / equity.cummax()
    max_dd = dd.max()
    rec    = total / max_dd if max_dd else np.nan
    return dict(equity=equity, total=total, max_dd=max_dd, recovery=rec, n_flip=int(n_flip))

# --------------------------------------------------------------------------- #
#                           ГЕНЕРАЦИЯ  ПАРАМ-СЕТА                             #
# --------------------------------------------------------------------------- #

def expand_grid(grid_dict):
    """Преобразуем GRID → список опций для itertools.product"""
    keys, values = [], []
    for k, v in grid_dict.items():
        if v is None:                       # отключено, вставляем значение по умолчанию
            v = [DEFAULTS[k]]
        elif not isinstance(v,(list,tuple,set)):
            v = [v]                         # фиксированное значение
        keys.append(k); values.append(list(v))
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))

# --------------------------------------------------------------------------- #
#                               ОПТИМИЗАЦИЯ                                   #
# --------------------------------------------------------------------------- #



def run_optim(grid, top_n=10):
    results = []
    iter_count = 0   # <- счётчик оптимизаций
    for p in expand_grid(grid):
        iter_count += 1
        df = get_df_tf(p["tf"])
        vm = volatility_median(df, p["KATR"], p["PerATR"], p["SMA"],
                               p["MinRA"], p["FlATR"], p["FlHL"])
        stats = backtest(
            df, vm,
            offset=p.get("offset", 0.0),
            mode=p.get("mode", "deviation"),
            slope_thr=p.get("slope_thr", 0.01)
        )
        results.append({**p,
                        "Recovery": stats["recovery"],
                        "Total":    stats["total"],
                        "MaxDD":    stats["max_dd"],
                        "N_Flip":   stats["n_flip"],
                        "Equity":   stats["equity"]})
    print(f"\nКоличество перебранных вариантов: {iter_count}")
    res = pd.DataFrame(results).sort_values("Recovery", ascending=False).head(top_n)
    return res

# ---------------------расчет KPI ----------------

# --------------------------------------------------------------------------- #
#                                   MAIN                                      #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # print(len.grid)
    top = run_optim(GRID, top_n=20)
    # MIN_FLIP = 25

    # def kpi(row):
    #     if row["N_Flip"] < MIN_FLIP or row["MaxDD"] == 0:
    #         return -np.inf  # (или np.nan, или фильтруй позже)
    #     return row["Total"] / (row["N_Flip"] * row["MaxDD"])
    
    # top["KPI"] = top.apply(kpi, axis=1)
    # top = top[top["KPI"] != -np.inf]           # убрать те, что отсекли
    # top = top.sort_values("KPI", ascending=False).head(10)   # ← итоговый топ по новому критерию
    print("\nTOP-5 по Recovery Factor:\n")
    print(top[["KATR", "PerATR", "SMA", "MinRA", "offset", "tf", "mode", "slope_thr", "N_Flip", "Recovery", "Total", "MaxDD"]])

    # print(top[["KATR","PerATR","SMA","MinRA","offset","FlATR","FlHL","tf",
    #        "Recovery","Total","MaxDD"]])
    # print(top[["KATR","PerATR","SMA","MinRA","FlATR","FlHL","tf",
    #            "Recovery","Total","MaxDD"]])
    

    best = top.iloc[0]
    best["Equity"].plot(title=f"{TICKER}  best VM-set", figsize=(9,4))
    plt.ylabel("Кумулятивная доходность"); plt.grid(); plt.tight_layout()
    plt.show()

