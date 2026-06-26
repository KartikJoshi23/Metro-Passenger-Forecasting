"""
Dubai Metro Passenger Forecasting — core pipeline (next-hour, station-level inflow).
Trained on REAL Dubai RTA AFC tap data (Dubai Pulse `rta_metro_ridership-open`).

This script is the executable twin of dubai_metro_forecasting.ipynb. Running it end-to-end
produces: data/processed/*, outputs/figures/*.png, outputs/metrics.json,
outputs/forecast_sample.json, outputs/eda.json  (the dashboard reads these JSONs).
"""
import os, glob, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
np.random.seed(42)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
OUT  = os.path.join(ROOT, "outputs")
FIG  = os.path.join(OUT, "figures")
for d in (PROC, OUT, FIG): os.makedirs(d, exist_ok=True)

# Dubai Metro brand palette for figures
RED, GREEN, INK, GRID = "#E4002B", "#00A651", "#0B1B3A", "#dfe6f3"
plt.rcParams.update({"figure.facecolor":"white","axes.grid":True,"grid.color":GRID,
                     "axes.edgecolor":"#94a3b8","font.size":11})

LOOKBACK = 4          # hours of history fed to the model
OP_HOURS = list(range(5, 24))   # Dubai Metro operating window ~05:00–24:00

# ----------------------------------------------------------------------------------
# 1. LOAD REAL DUBAI AFC TAP DATA
# ----------------------------------------------------------------------------------
def load_taps():
    files = sorted(glob.glob(os.path.join(RAW, "metro_ridership_*.csv")))
    if not files:
        raise FileNotFoundError("No metro_ridership_*.csv in data/raw/")
    df = pd.read_csv(files[0], usecols=["txn_type","txn_date","txn_time",
                                        "start_location","line_name"])
    df["txn_type"] = df["txn_type"].str.strip()
    df = df[df["txn_type"] == "Check in"].copy()          # entries = inflow
    df["date"] = pd.to_datetime(df["txn_date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["hour"] = df["txn_time"].str.slice(0,2).astype(int)
    df["station"] = (df["start_location"].str.replace(" Metro Station","",regex=False)
                                          .str.strip())
    df["line"] = df["line_name"].str.strip()
    return df

# ----------------------------------------------------------------------------------
# 2. AGGREGATE TO HOURLY STATION-LEVEL INFLOW (real)
# ----------------------------------------------------------------------------------
def to_hourly(df):
    g = (df.groupby([df["date"].dt.normalize().rename("day"), "station", "line", "hour"])
            .size().reset_index(name="inflow"))
    # Reindex each (day, station) to the full operating-hour grid, filling gaps with 0
    keys = g[["day","station","line"]].drop_duplicates()
    full = (keys.assign(k=1).merge(pd.DataFrame({"hour":OP_HOURS,"k":1}), on="k")
                .drop(columns="k"))
    h = full.merge(g, on=["day","station","line","hour"], how="left")
    h["inflow"] = h["inflow"].fillna(0.0)
    h["dow"]    = h["day"].dt.dayofweek                 # 0=Mon..6=Sun
    h["is_weekend"] = h["dow"].isin([5,6]).astype(int)  # UAE weekend = Sat/Sun
    return h.sort_values(["station","day","hour"]).reset_index(drop=True)

# ----------------------------------------------------------------------------------
# 3. SUPERVISED WINDOWS (within-day; never cross the day boundary)
# ----------------------------------------------------------------------------------
def make_windows(h, lookback=LOOKBACK):
    # per-station scaling factor (mean inflow) so big & small stations train together
    st_mean = h.groupby("station")["inflow"].mean().clip(lower=1.0)
    X_seq, X_ctx, y, meta = [], [], [], []
    for (day, station), grp in h.groupby(["day","station"]):
        grp = grp.sort_values("hour")
        vals = grp["inflow"].values.astype("float32")
        hours = grp["hour"].values
        sc = st_mean[station]
        v = vals / sc                                   # scale-normalised inflow
        for i in range(lookback, len(v)):
            X_seq.append(v[i-lookback:i])
            hr = hours[i]
            dow = grp["dow"].iloc[0]; wk = grp["is_weekend"].iloc[0]
            X_ctx.append([np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24),
                          np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7), wk])
            y.append(v[i])
            meta.append((pd.Timestamp(day), station, int(hr), float(sc), float(vals[i])))
    X_seq = np.array(X_seq)[..., None]                  # (N, lookback, 1)
    X_ctx = np.array(X_ctx, dtype="float32")            # (N, 5)
    y = np.array(y, dtype="float32")
    meta = pd.DataFrame(meta, columns=["day","station","hour","scale","actual_real"])
    return X_seq, X_ctx, y, meta

# ----------------------------------------------------------------------------------
# 4. MODELS (Keras).
#    SOLUTION = "LSTM+RNN" hybrid: an LSTM encoder feeding a SimpleRNN — combining LSTM's
#    long-range memory with the RNN's compact recent-momentum read. The remaining models
#    (standalone LSTM, standalone RNN, GRU, CNN-LSTM) exist ONLY for the comparison table.
# ----------------------------------------------------------------------------------
SOLUTION = "LSTM+RNN"

def build_models(lookback, n_ctx):
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    tf.random.set_seed(42)
    def head(seq_out):
        c = Input(shape=(n_ctx,), name="ctx")
        x = layers.Concatenate()([seq_out.output, c])
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(1)(x)
        return Model([seq_out.input, c], x)
    def seq_input():
        return Input(shape=(lookback,1), name="seq")
    models = {}
    # ---- THE SOLUTION: LSTM + RNN hybrid (LSTM encoder -> SimpleRNN) ----
    s = seq_input()
    r = layers.LSTM(48, return_sequences=True)(s)   # LSTM captures longer-range structure
    r = layers.SimpleRNN(32)(r)                     # RNN distils recent momentum
    models[SOLUTION] = head(Model(s, r))
    # ---- comparison-only architectures ----
    s = seq_input(); r = layers.LSTM(48, return_sequences=True)(s); r = layers.LSTM(24)(r)
    models["LSTM (only)"] = head(Model(s, r))
    s = seq_input(); r = layers.SimpleRNN(32)(s); models["RNN (only)"] = head(Model(s, r))
    s = seq_input(); r = layers.GRU(40)(s); models["GRU"] = head(Model(s, r))
    s = seq_input(); r = layers.Conv1D(32,2,activation="relu",padding="causal")(s)
    r = layers.LSTM(32)(r); models["CNN-LSTM"] = head(Model(s, r))
    for m in models.values():
        m.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return models

# ----------------------------------------------------------------------------------
# 5. METRICS
# ----------------------------------------------------------------------------------
def metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true > 5                                   # MAPE only where demand is meaningful
    mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100) if mask.any() else None
    r2   = r2_score(y_true, y_pred)
    return {"MAE":round(float(mae),2),"RMSE":round(float(rmse),2),
            "MAPE":round(mape,2) if mape is not None else None,"R2":round(float(r2),4)}

# ----------------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------------
def main():
    print(">> Loading real Dubai AFC taps ...")
    df = load_taps()
    print(f"   check-in taps: {len(df):,} | days: {df['date'].nunique()} | stations: {df['station'].nunique()}")

    print(">> Aggregating to hourly station inflow ...")
    h = to_hourly(df)
    h.to_parquet(os.path.join(PROC, "hourly_station_inflow.parquet"))
    print(f"   hourly rows: {len(h):,}")

    # ---- EDA artefacts ----
    prof = df.groupby("hour").size() / df["date"].nunique()
    prof = prof.reindex(OP_HOURS).fillna(0)
    wk = df.assign(w=df["date"].dt.dayofweek.isin([5,6]))
    prof_wd = (df[~df["date"].dt.dayofweek.isin([5,6])].groupby("hour").size()
               / max(1,(df["date"].dt.dayofweek.isin([5,6])==False).groupby(df["date"].dt.date).first().sum())).reindex(OP_HOURS).fillna(0)
    # simpler weekday/weekend avg profile
    days_wd = df[~df["date"].dt.dayofweek.isin([5,6])]["date"].dt.normalize().nunique()
    days_we = df[df["date"].dt.dayofweek.isin([5,6])]["date"].dt.normalize().nunique()
    pwd = (df[~df["date"].dt.dayofweek.isin([5,6])].groupby("hour").size()/max(1,days_wd)).reindex(OP_HOURS).fillna(0)
    pwe = (df[df["date"].dt.dayofweek.isin([5,6])].groupby("hour").size()/max(1,days_we)).reindex(OP_HOURS).fillna(0)
    top = (df.groupby("station").size().sort_values(ascending=False).head(12))
    line_split = df.groupby("line").size()

    # Figure 1: intraday profile
    plt.figure(figsize=(9,4.2))
    plt.plot(OP_HOURS, prof.values, color=RED, lw=2.5, marker="o", label="All days")
    plt.plot(OP_HOURS, pwd.values, color=INK, lw=1.8, ls="--", label="Weekday")
    plt.plot(OP_HOURS, pwe.values, color=GREEN, lw=1.8, ls=":", label="Weekend")
    plt.title("Dubai Metro — average hourly inflow profile (real AFC)"); plt.xlabel("Hour of day")
    plt.ylabel("Avg check-ins / station-hour"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"01_intraday_profile.png"), dpi=120); plt.close()

    # Figure 2: top stations
    plt.figure(figsize=(9,4.6))
    top.sort_values().plot(kind="barh", color=RED)
    plt.title("Busiest stations by check-ins (real sample)"); plt.xlabel("Check-ins")
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"02_top_stations.png"), dpi=120); plt.close()

    eda = {
        "hours": OP_HOURS,
        "profile_all":[round(float(x),1) for x in prof.values],
        "profile_weekday":[round(float(x),1) for x in pwd.values],
        "profile_weekend":[round(float(x),1) for x in pwe.values],
        "top_stations":[{"station":k,"checkins":int(v)} for k,v in top.items()],
        "line_split":{k:int(v) for k,v in line_split.items()},
        "n_taps":int(len(df)), "n_days":int(df['date'].nunique()),
        "n_stations":int(df['station'].nunique()),
    }
    json.dump(eda, open(os.path.join(OUT,"eda.json"),"w"), indent=2)

    # ---- Windows + chronological split by day ----
    print(">> Building supervised windows (next-hour) ...")
    Xs, Xc, y, meta = make_windows(h)
    print(f"   windows: {len(y):,}")
    days_sorted = np.sort(meta["day"].unique())
    n = len(days_sorted)
    tr_days = set(days_sorted[:int(n*0.7)])
    va_days = set(days_sorted[int(n*0.7):int(n*0.85)])
    te_days = set(days_sorted[int(n*0.85):])
    tr = meta["day"].isin(tr_days).values
    va = meta["day"].isin(va_days).values
    te = meta["day"].isin(te_days).values
    print(f"   train/val/test days: {len(tr_days)}/{len(va_days)}/{len(te_days)}")

    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    cbs = [EarlyStopping(patience=8, restore_best_weights=True),
           ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-4)]

    models = build_models(LOOKBACK, Xc.shape[1])
    results, histories = {}, {}
    # de-normalise helper: pred(scaled) * scale = real inflow
    sc_te = meta.loc[te,"scale"].values
    y_te_real = meta.loc[te,"actual_real"].values

    # seasonal-naive baseline: predict = last observed hour (persistence) in real units
    persist_real = (Xs[te,-1,0] * sc_te)
    results["Naive (persistence)"] = metrics(y_te_real, persist_real)

    for name, model in models.items():
        print(f">> Training {name} ...")
        hcb = model.fit([Xs[tr],Xc[tr]], y[tr],
                        validation_data=([Xs[va],Xc[va]], y[va]),
                        epochs=40, batch_size=128, verbose=0, callbacks=cbs)
        histories[name] = [round(float(x),4) for x in hcb.history["val_mae"]]
        pred_scaled = model.predict([Xs[te],Xc[te]], verbose=0).ravel()
        pred_real = np.clip(pred_scaled * sc_te, 0, None)
        results[name] = metrics(y_te_real, pred_real)
        print("   ", results[name])

    json.dump({"metrics":results,"val_mae_curves":histories,"solution_model":SOLUTION,
               "lookback":LOOKBACK,"test_windows":int(te.sum())},
              open(os.path.join(OUT,"metrics.json"),"w"), indent=2)

    # ---- Figure 3: model comparison (MAE/RMSE) — solution highlighted ----
    names = [k for k in results]
    mae = [results[k]["MAE"] for k in names]; rmse=[results[k]["RMSE"] for k in names]
    x = np.arange(len(names)); w=0.38
    bar_colors = [GREEN if k==SOLUTION else RED for k in names]
    plt.figure(figsize=(9.5,4.4))
    plt.bar(x-w/2, mae, w, label="MAE", color=bar_colors)
    plt.bar(x+w/2, rmse, w, label="RMSE", color=INK)
    plt.xticks(x, names, rotation=20, ha="right"); plt.ylabel("error (check-ins/hr)")
    plt.title("Approach comparison — next-hour inflow (LSTM+RNN = our solution)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"03_model_comparison.png"), dpi=120); plt.close()

    # ---- Forecast sample for dashboard: OUR SOLUTION (LSTM+RNN) on busy test station-days ----
    best = SOLUTION
    bm = models[best]
    # pick a test day + busy station with the most windows
    cand = meta[te].groupby(["day","station"]).size().sort_values(ascending=False)
    series_out = []
    for (day, station) in cand.index[:6]:
        idx = meta.index[(meta["day"]==day)&(meta["station"]==station)&te]
        if len(idx)<3: continue
        sc = meta.loc[idx,"scale"].values
        pr = np.clip(bm.predict([Xs[idx],Xc[idx]],verbose=0).ravel()*sc,0,None)
        series_out.append({
            "station":station,"day":str(pd.Timestamp(day).date()),
            "hours":[int(x) for x in meta.loc[idx,"hour"].values],
            "actual":[round(float(x),1) for x in meta.loc[idx,"actual_real"].values],
            "predicted":[round(float(x),1) for x in pr],
        })
    json.dump({"solution_model":SOLUTION,"best_model":best,"metrics":results[best],
               "series":series_out},
              open(os.path.join(OUT,"forecast_sample.json"),"w"), indent=2)

    # ---- Figure 4: actual vs predicted for the top sample ----
    if series_out:
        s0 = series_out[0]
        plt.figure(figsize=(9,4.2))
        plt.plot(s0["hours"], s0["actual"], color=INK, lw=2.4, marker="o", label="Actual")
        plt.plot(s0["hours"], s0["predicted"], color=RED, lw=2.4, ls="--", marker="s", label=f"{best} predicted")
        plt.title(f"Next-hour forecast — {s0['station']} ({s0['day']})")
        plt.xlabel("Hour"); plt.ylabel("Inflow (check-ins/hr)"); plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(FIG,"04_forecast_sample.png"), dpi=120); plt.close()

    print("\n=== DONE ===  solution model:", SOLUTION)
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
