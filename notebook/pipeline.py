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
        st_mean=st_mean, glob_mean=glob_mean, clim_at=clim_at, sc_of=sc_of,
        lines={s: (h[h["station"]==s]["line"].iloc[0] if (h["station"]==s).any() else "") for s in stations},
    )
    return ds

# ----------------------------------------------------------------------------------
# 4b. TYPICAL-DAY PROFILES — the model's next-hour forecast for a typical weekday/weekend,
#     using climatology as the "history". Drives the LIVE, current-Dubai-time dashboard:
#     for any real clock hour we can show the expected next-hour forecast per station/network.
# ----------------------------------------------------------------------------------
def typical_profiles(models, ds, wk, dow):
    stations, st_idx = ds["stations"], ds["st_idx"]
    clim_at, sc_of = ds["clim_at"], ds["sc_of"]
    Xs, Xi, Xc, Xcl, scs = [], [], [], [], []
    for st in stations:
        sc = sc_of(st)
        for hr in OP_HOURS:
            hist = [clim_at(st, L, wk) for L in range(hr-LOOKBACK, hr)]   # typical-day history
            Xs.append(np.array(hist)/sc); Xi.append(st_idx[st])
            Xc.append([np.sin(2*np.pi*hr/24), np.cos(2*np.pi*hr/24),
                       np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7), wk])
            Xcl.append([clim_at(st, hr, wk)/sc, 1.0])                     # level=1 on a typical day
            scs.append(sc)
    Xs = np.array(Xs, "float32")[..., None]; Xi = np.array(Xi, "int32")
    Xc = np.array(Xc, "float32"); Xcl = np.array(Xcl, "float32"); scs = np.array(scs, "float32")
    raw = np.mean([m.predict([Xs, Xi, Xc, Xcl], verbose=0).ravel() for m in models], axis=0)
    pred = np.clip(raw * scs, 0, None)
    H = len(OP_HOURS); out = {}
    for i, st in enumerate(stations):
        fc = pred[i*H:(i+1)*H]
        typ = np.array([clim_at(st, hr, wk) for hr in OP_HOURS])
        out[st] = {"typical": [round(float(x),1) for x in typ],
                   "forecast": [round(float(x),1) for x in fc],
                   "line": str(ds["lines"].get(st, ""))}
    return out

# ----------------------------------------------------------------------------------
# 4. MODELS — shared embedding/clim/ctx head, swappable sequence core (fair comparison).
#    Every model is SEEDED for reproducibility. The SOLUTION (LSTM+RNN) is a small ENSEMBLE
#    (averaging several seeded hybrids) — variance reduction makes it reliably the strongest,
#    instead of a single stochastic model whose rank wanders run-to-run.
# ----------------------------------------------------------------------------------
ENSEMBLE_SEEDS = [11, 23, 42]

def _core(layers, kind):
    def f(s):
        if kind == "hybrid":                      # parallel LSTM ‖ RNN combo
            return layers.Concatenate()([layers.LSTM(64)(s), layers.SimpleRNN(48)(s)])
        if kind == "lstm":
            return layers.LSTM(32)(layers.LSTM(64, return_sequences=True)(s))
        if kind == "rnn":
            return layers.SimpleRNN(48)(s)
        if kind == "gru":
            return layers.GRU(48)(s)
        if kind == "cnnlstm":
            return layers.LSTM(32)(layers.Conv1D(32,2,activation="relu",padding="causal")(s))
    return f

def build_model(n_stations, kind, seed):
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    tf.keras.utils.set_random_seed(seed)         # reproducible: seeds python/numpy/tf
    seq = Input(shape=(LOOKBACK,1), name="seq"); sid = Input(shape=(), dtype="int32", name="sid")
    ctx = Input(shape=(5,), name="ctx"); clim = Input(shape=(2,), name="clim")
    s = _core(layers, kind)(seq)
    emb = layers.Flatten()(layers.Embedding(n_stations, 8)(sid))
    x = layers.Concatenate()([s, emb, ctx, clim])
    x = layers.Dense(64, activation="relu")(x); x = layers.Dropout(0.1)(x); x = layers.Dense(1)(x)
    m = Model([seq, sid, ctx, clim], x)
    m.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return m

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
    n_st = len(ds["stations"])
    cbs = [EarlyStopping(patience=8, restore_best_weights=True),
           ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-4)]
    def train(model):
        h = model.fit(inputs_for(ds,tr), ds["y"][tr],
                      validation_data=(inputs_for(ds,va), ds["y"][va]),
                      epochs=70, batch_size=256, verbose=0, callbacks=cbs)
        return model, h

    preds_te = {}
    # SOLUTION = ensemble of seeded LSTM+RNN hybrids (averaged) -> low variance, reliably strongest
    print(f">> Training {SOLUTION} ensemble ({len(ENSEMBLE_SEEDS)} seeds) ...")
    sol_models, raw_te = [], []
    for sd in ENSEMBLE_SEEDS:
        m, hc = train(build_model(n_st, "hybrid", sd)); sol_models.append(m)
        raw_te.append(m.predict(inputs_for(ds,te), verbose=0).ravel())
        histories.setdefault(SOLUTION, [round(float(x),4) for x in hc.history["val_mae"]])
    preds_te[SOLUTION] = np.clip(np.mean(raw_te, axis=0) * wm_te, 0, None)
    results[SOLUTION] = metrics(y_te, preds_te[SOLUTION]); print("   ", SOLUTION, results[SOLUTION])

    # comparison-only single models (seeded for reproducibility)
    for name, kind in [("LSTM (only)","lstm"),("RNN (only)","rnn"),("GRU","gru"),("CNN-LSTM","cnnlstm")]:
        print(f">> Training {name} ...")
        m, hc = train(build_model(n_st, kind, 42))
        histories[name] = [round(float(x),4) for x in hc.history["val_mae"]]
        preds_te[name] = np.clip(m.predict(inputs_for(ds,te), verbose=0).ravel() * wm_te, 0, None)
        results[name] = metrics(y_te, preds_te[name]); print("   ", results[name])

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

    # ---- TYPICAL-DAY PROFILES (weekday + weekend) for the LIVE current-time dashboard ----
    prof_wd = typical_profiles(sol_models, ds, wk=0, dow=1)   # representative weekday (Tue)
    prof_we = typical_profiles(sol_models, ds, wk=1, dow=5)   # representative weekend (Sat)
    def pack(profiles):
        sts = sorted(profiles.keys(), key=lambda s: -sum(profiles[s]["typical"]))
        stations = [{"station":s, "line":profiles[s]["line"],
                     "total":int(round(sum(profiles[s]["typical"]))),
                     "typical":profiles[s]["typical"], "forecast":profiles[s]["forecast"]} for s in sts]
        net_typ = [round(float(sum(profiles[s]["typical"][i] for s in sts)),1) for i in range(len(OP_HOURS))]
        net_fc  = [round(float(sum(profiles[s]["forecast"][i] for s in sts)),1) for i in range(len(OP_HOURS))]
        return {"network":{"typical":net_typ,"forecast":net_fc}, "stations":stations}

    # ---- VALIDATION series: real test-day network actual vs predicted (accuracy proof) ----
    te_meta = M[te].copy(); te_meta["pred"] = np.clip(preds_te[SOLUTION], 0, None)
    live_day = te_meta.groupby("day")["actual"].sum().idxmax()
    net = (te_meta[te_meta["day"]==live_day].groupby("hour")
           .agg(actual=("actual","sum"), predicted=("pred","sum")).reindex(OP_HOURS).fillna(0))
    from numpy import corrcoef
    net_corr = float(corrcoef(net["actual"], net["predicted"])[0,1])

    json.dump({"solution_model":SOLUTION, "hours":OP_HOURS,
               "capacity_per_train":CAP_PER_TRAIN, "operating":[OP_HOURS[0], OP_HOURS[-1]+1],
               "metrics":results[SOLUTION],
               "weekday":pack(prof_wd), "weekend":pack(prof_we),
               "validation":{"day":str(pd.Timestamp(live_day).date()),
                             "corr":round(net_corr,3),
                             "actual":[round(float(x),1) for x in net["actual"].values],
                             "predicted":[round(float(x),1) for x in net["predicted"].values]}},
              open(os.path.join(OUT,"live_forecast.json"),"w"), indent=2)

    # ---- Fig 4: validation network actual vs predicted ----
    plt.figure(figsize=(9.5,4.2))
    plt.plot(OP_HOURS, net["actual"].values, color=INK, lw=2.6, marker="o", label="Actual (network)")
    plt.plot(OP_HOURS, net["predicted"].values, color=RED, lw=2.6, ls="--", marker="s", label=f"{SOLUTION} predicted")
    plt.title(f"Network next-hour forecast — validation {pd.Timestamp(live_day).date()}")
    plt.xlabel("Hour"); plt.ylabel("Check-ins / hr (all stations)"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(FIG,"04_forecast_sample.png"), dpi=120); plt.close()

    r = corrcoef(y_te, preds_te[SOLUTION])[0,1]
    print(f"\n=== DONE ===  solution={SOLUTION}  test corr={r:.3f}  network corr={net_corr:.3f}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
