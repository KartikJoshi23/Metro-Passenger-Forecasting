"""
Dubai Metro Passenger Forecasting — core pipeline (next-hour, ALL stations).
Trained on REAL Dubai RTA AFC tap data (Dubai Pulse `rta_metro_ridership-open`).

SOLUTION = LSTM + RNN hybrid (LSTM encoder -> SimpleRNN) with:
  * per-station learned EMBEDDING            (distinguishes 65 stations, not just mean-scaling)
  * station x hour x weekend CLIMATOLOGY      (strong real prior; computed on TRAIN days only)
  * log1p targets                             (handles the wide 0..hundreds range; proportional error)
Comparison-only models (same inputs, different sequence core): LSTM, RNN, GRU, CNN-LSTM,
plus two non-NN baselines (Naive persistence, Climatology).

Run end-to-end -> data/processed/*, outputs/figures/*.png, outputs/{metrics,eda,live_forecast}.json
(the dashboard reads the JSONs).
"""
import os, glob, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
np.random.seed(42)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
OUT  = os.path.join(ROOT, "outputs")
FIG  = os.path.join(OUT, "figures")
for d in (PROC, OUT, FIG): os.makedirs(d, exist_ok=True)

RED, GREEN, INK, GRID = "#E4002B", "#00A651", "#0B1B3A", "#dfe6f3"
plt.rcParams.update({"figure.facecolor":"white","axes.grid":True,"grid.color":GRID,
                     "axes.edgecolor":"#94a3b8","font.size":11})

LOOKBACK = 6
OP_HOURS = list(range(5, 24))          # Dubai Metro operating window ~05:00–24:00
SOLUTION = "LSTM+RNN"
CAP_PER_TRAIN = 643                    # real RTA figure: 5-car train, 643 passengers
MIN_YEAR = 2026                        # use the LATEST regime only — older sampled days
                                       # (2017–2020) are far lower-volume and would inject a
                                       # train/test distribution shift (and the brief wants latest data)

# ----------------------------------------------------------------------------------
# 1. LOAD + 2. HOURLY AGGREGATION
# ----------------------------------------------------------------------------------
def load_taps():
    files = sorted(glob.glob(os.path.join(RAW, "metro_ridership_*.csv")))
    if not files:
        raise FileNotFoundError("No metro_ridership_*.csv in data/raw/")
    df = pd.read_csv(files[0], usecols=["txn_type","txn_date","txn_time",
                                        "start_location","line_name"])
    df["txn_type"] = df["txn_type"].str.strip()
    df = df[df["txn_type"] == "Check in"].copy()
    df["date"] = pd.to_datetime(df["txn_date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"].dt.year >= MIN_YEAR].copy()        # latest regime only
    df["hour"] = df["txn_time"].str.slice(0,2).astype(int)
    df["station"] = df["start_location"].str.replace(" Metro Station","",regex=False).str.strip()
    df["line"] = df["line_name"].str.strip()
    return df

def to_hourly(df):
    g = (df.groupby([df["date"].dt.normalize().rename("day"),"station","line","hour"])
            .size().reset_index(name="inflow"))
    keys = g[["day","station","line"]].drop_duplicates()
    full = (keys.assign(k=1).merge(pd.DataFrame({"hour":OP_HOURS,"k":1}), on="k").drop(columns="k"))
    h = full.merge(g, on=["day","station","line","hour"], how="left")
    h["inflow"] = h["inflow"].fillna(0.0)
    h["dow"] = h["day"].dt.dayofweek
    h["is_weekend"] = h["dow"].isin([5,6]).astype(int)     # UAE weekend = Sat/Sun
    return h.sort_values(["station","day","hour"]).reset_index(drop=True)

# ----------------------------------------------------------------------------------
# 3. FEATURES: sequence (+climatology padding) + station idx + calendar + clim prior
# ----------------------------------------------------------------------------------
def build_dataset(h, train_days):
    """Per-station scaling: divide by each station's TRAIN-mean inflow (leakage-safe) so big and
    small stations train together. A learned station EMBEDDING and a station x hour x weekend
    CLIMATOLOGY feature give the model a strong real prior, which lets it anticipate the daily
    peak shape (reducing the one-step 'lag' look) instead of just echoing the last hour."""
    stations = sorted(h["station"].unique())
    st_idx = {s:i for i,s in enumerate(stations)}
    tr = h[h["day"].isin(train_days)]
    st_mean = tr.groupby("station")["inflow"].mean().clip(lower=1.0)
    glob_mean = float(tr["inflow"].mean())
    clim = tr.groupby(["station","hour","is_weekend"])["inflow"].mean()
    glob_clim = tr.groupby(["hour","is_weekend"])["inflow"].mean()
    def sc_of(station): return float(st_mean.get(station, glob_mean))
    def clim_at(station, hour, wk):
        hour = min(max(hour, OP_HOURS[0]), OP_HOURS[-1])
        v = clim.get((station, hour, wk))
        if v is None or np.isnan(v): v = glob_clim.get((hour, wk), 0.0)
        return float(v)

    Xseq, Xsid, Xctx, Xclim, y, wm, meta = [], [], [], [], [], [], []
    for (day, station, line), grp in h.groupby(["day","station","line"]):
        grp = grp.sort_values("hour")
        wk = int(grp["is_weekend"].iloc[0]); dow = int(grp["dow"].iloc[0])
        inflow = dict(zip(grp["hour"], grp["inflow"]))
        sc = sc_of(station)
        for hr in OP_HOURS:
            hist  = [inflow.get(L, clim_at(station, L, wk)) for L in range(hr-LOOKBACK, hr)]
            hclim = [clim_at(station, L, wk) for L in range(hr-LOOKBACK, hr)]
            # today's busyness LEVEL = recent actual / recent typical (past-only, leakage-safe).
            level = float(np.clip(np.sum(hist)/max(np.sum(hclim), 1e-6), 0.2, 5.0))
            Xseq.append(np.array(hist)/sc)
            Xsid.append(st_idx[station])
            Xctx.append([np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24),
                         np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7), wk])
            # clim feature carries the typical SHAPE; level scales it to today (both past-only)
            Xclim.append([clim_at(station, hr, wk)/sc, level])
            y.append(inflow.get(hr, 0.0)/sc)
            wm.append(sc)
            meta.append((pd.Timestamp(day), station, line, hr, float(inflow.get(hr,0.0)),
                         float(clim_at(station, hr, wk)), float(hist[-1])))
    ds = dict(
        seq=np.array(Xseq, dtype="float32")[..., None],
        sid=np.array(Xsid, dtype="int32"),
        ctx=np.array(Xctx, dtype="float32"),
        clim=np.array(Xclim, dtype="float32"),
        y=np.array(y, dtype="float32"),
        wm=np.array(wm, dtype="float32"),
        meta=pd.DataFrame(meta, columns=["day","station","line","hour","actual","clim","prev"]),
        stations=stations, st_idx=st_idx,
    )
    return ds

# ----------------------------------------------------------------------------------
# 4. MODELS — shared embedding/clim/ctx head, swappable sequence core (fair comparison)
# ----------------------------------------------------------------------------------
def build_models(n_stations):
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    tf.random.set_seed(42)

    def make(core_fn):
        seq = Input(shape=(LOOKBACK,1), name="seq")
        sid = Input(shape=(), dtype="int32", name="sid")
        ctx = Input(shape=(5,), name="ctx")
        clim = Input(shape=(2,), name="clim")   # [climatology/scale, today's level]
        s = core_fn(seq)
        emb = layers.Flatten()(layers.Embedding(n_stations, 8)(sid))
        x = layers.Concatenate()([s, emb, ctx, clim])
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(1)(x)
        m = Model([seq, sid, ctx, clim], x)
        m.compile(optimizer="adam", loss="huber", metrics=["mae"])
        return m

    def core_hybrid(s):   # THE SOLUTION: LSTM encoder -> SimpleRNN
        r = layers.LSTM(64, return_sequences=True)(s)
        return layers.SimpleRNN(32)(r)
    def core_lstm(s):
        r = layers.LSTM(64, return_sequences=True)(s); return layers.LSTM(32)(r)
    def core_rnn(s):
        return layers.SimpleRNN(48)(s)
    def core_gru(s):
        return layers.GRU(48)(s)
    def core_cnnlstm(s):
        r = layers.Conv1D(32,2,activation="relu",padding="causal")(s); return layers.LSTM(32)(r)

    return {SOLUTION: make(core_hybrid), "LSTM (only)": make(core_lstm),
            "RNN (only)": make(core_rnn), "GRU": make(core_gru),
            "CNN-LSTM": make(core_cnnlstm)}

def inputs_for(ds, mask):
    return [ds["seq"][mask], ds["sid"][mask], ds["ctx"][mask], ds["clim"][mask]]

# ----------------------------------------------------------------------------------
# 5. METRICS (computed on real scale via expm1)
# ----------------------------------------------------------------------------------
def metrics(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true > 5
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
    h = to_hourly(df)
    h.to_parquet(os.path.join(PROC, "hourly_station_inflow.parquet"))

    # chronological day split
    days = np.sort(h["day"].unique()); n = len(days)
    tr_days = set(days[:int(n*0.7)]); va_days = set(days[int(n*0.7):int(n*0.85)])
    te_days = set(days[int(n*0.85):])
    print(f">> days train/val/test = {len(tr_days)}/{len(va_days)}/{len(te_days)}")

    ds = build_dataset(h, tr_days)
    M = ds["meta"]
    tr = M["day"].isin(tr_days).values; va = M["day"].isin(va_days).values; te = M["day"].isin(te_days).values
    print(f">> samples: {len(M):,} (train {tr.sum()} / val {va.sum()} / test {te.sum()})")

    # ---- EDA ----
    prof = (df.groupby("hour").size()/df["date"].nunique()).reindex(OP_HOURS).fillna(0)
    isw = df["date"].dt.dayofweek.isin([5,6])
    dwd = df[~isw]["date"].dt.normalize().nunique(); dwe = df[isw]["date"].dt.normalize().nunique()
    pwd = (df[~isw].groupby("hour").size()/max(1,dwd)).reindex(OP_HOURS).fillna(0)
    pwe = (df[isw].groupby("hour").size()/max(1,dwe)).reindex(OP_HOURS).fillna(0)
    top = df.groupby("station").size().sort_values(ascending=False).head(12)
    line_split = df.groupby("line").size()
    plt.figure(figsize=(9,4.2))
    plt.plot(OP_HOURS, prof.values, color=RED, lw=2.5, marker="o", label="All days")
    plt.plot(OP_HOURS, pwd.values, color=INK, lw=1.8, ls="--", label="Weekday")
    plt.plot(OP_HOURS, pwe.values, color=GREEN, lw=1.8, ls=":", label="Weekend")
    plt.title("Dubai Metro — average hourly inflow profile (real AFC)"); plt.xlabel("Hour of day")
    plt.ylabel("Avg check-ins / station-hour"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"01_intraday_profile.png"), dpi=120); plt.close()
    json.dump({"hours":OP_HOURS,
               "profile_all":[round(float(x),1) for x in prof.values],
               "profile_weekday":[round(float(x),1) for x in pwd.values],
               "profile_weekend":[round(float(x),1) for x in pwe.values],
               "top_stations":[{"station":k,"checkins":int(v)} for k,v in top.items()],
               "line_split":{k:int(v) for k,v in line_split.items()},
               "n_taps":int(len(df)),"n_days":int(df['date'].nunique()),
               "n_stations":int(df['station'].nunique())},
              open(os.path.join(OUT,"eda.json"),"w"), indent=2)

    # ---- baselines ----
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    y_te = M.loc[te,"actual"].values
    wm_te = ds["wm"][te]
    results, histories = {}, {}
    results["Naive (persistence)"] = metrics(y_te, M.loc[te,"prev"].values)
    results["Climatology"]         = metrics(y_te, M.loc[te,"clim"].values)

    # ---- NN models ----
    models = build_models(len(ds["stations"]))
    cbs = [EarlyStopping(patience=8, restore_best_weights=True),
           ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-4)]
    preds_te = {}
    for name, model in models.items():
        print(f">> Training {name} ...")
        hcb = model.fit(inputs_for(ds,tr), ds["y"][tr],
                        validation_data=(inputs_for(ds,va), ds["y"][va]),
                        epochs=60, batch_size=256, verbose=0, callbacks=cbs)
        histories[name] = [round(float(x),4) for x in hcb.history["val_mae"]]
        p = np.clip(model.predict(inputs_for(ds,te), verbose=0).ravel() * wm_te, 0, None)
        preds_te[name] = p
        results[name] = metrics(y_te, p)
        print("   ", results[name])

    json.dump({"metrics":results,"val_mae_curves":histories,"solution_model":SOLUTION,
               "lookback":LOOKBACK,"test_windows":int(te.sum()),
               "n_stations":int(len(ds["stations"]))},
              open(os.path.join(OUT,"metrics.json"),"w"), indent=2)

    # ---- Fig 3: comparison ----
    names=list(results); x=np.arange(len(names)); w=.38
    cols=[GREEN if k==SOLUTION else RED for k in names]
    plt.figure(figsize=(10,4.4))
    plt.bar(x-w/2,[results[k]["MAE"] for k in names],w,label="MAE",color=cols)
    plt.bar(x+w/2,[results[k]["RMSE"] for k in names],w,label="RMSE",color=INK)
    plt.xticks(x,names,rotation=20,ha="right"); plt.ylabel("error (check-ins/hr)")
    plt.title("Approach comparison — LSTM+RNN is our solution (green)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"03_model_comparison.png"), dpi=120); plt.close()

    # ---- LIVE forecast export: pick the busiest COMPLETE test day; all stations + network ----
    sol = models[SOLUTION]
    te_meta = M[te].copy(); te_meta["pred"] = np.clip(preds_te[SOLUTION], 0, None)
    day_load = te_meta.groupby("day")["actual"].sum().sort_values(ascending=False)
    live_day = day_load.index[0]
    dsel = te_meta[te_meta["day"]==live_day]
    # network-level (sum across stations) per hour
    net = dsel.groupby("hour").agg(actual=("actual","sum"), predicted=("pred","sum")).reindex(OP_HOURS).fillna(0)
    # per-station series (ALL stations on that day)
    stations_out=[]
    for station, g in dsel.groupby("station"):
        g = g.set_index("hour").reindex(OP_HOURS)
        line = g["line"].dropna().iloc[0] if g["line"].notna().any() else ""
        stations_out.append({
            "station":station, "line":str(line),
            "total":int(np.nansum(g["actual"].values)),
            "actual":[round(float(x),1) for x in np.nan_to_num(g["actual"].values)],
            "predicted":[round(float(x),1) for x in np.nan_to_num(g["pred"].values)],
            "climatology":[round(float(x),1) for x in np.nan_to_num(g["clim"].values)],
        })
    stations_out.sort(key=lambda s:-s["total"])
    json.dump({"solution_model":SOLUTION,"day":str(pd.Timestamp(live_day).date()),
               "hours":OP_HOURS,"capacity_per_train":CAP_PER_TRAIN,
               "metrics":results[SOLUTION],
               "network":{"actual":[round(float(x),1) for x in net["actual"].values],
                          "predicted":[round(float(x),1) for x in net["predicted"].values]},
               "stations":stations_out},
              open(os.path.join(OUT,"live_forecast.json"),"w"), indent=2)

    # ---- Fig 4: network actual vs predicted on the live day ----
    plt.figure(figsize=(9.5,4.2))
    plt.plot(OP_HOURS, net["actual"].values, color=INK, lw=2.6, marker="o", label="Actual (network)")
    plt.plot(OP_HOURS, net["predicted"].values, color=RED, lw=2.6, ls="--", marker="s", label=f"{SOLUTION} predicted")
    plt.title(f"Network next-hour forecast — {pd.Timestamp(live_day).date()}")
    plt.xlabel("Hour"); plt.ylabel("Check-ins / hr (all stations)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"04_forecast_sample.png"), dpi=120); plt.close()

    # correlation sanity (real scale)
    from numpy import corrcoef
    r = corrcoef(y_te, preds_te[SOLUTION])[0,1]
    print(f"\n=== DONE ===  solution={SOLUTION}  test corr(actual,pred)={r:.3f}  live_day={pd.Timestamp(live_day).date()}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
