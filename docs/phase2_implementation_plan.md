# Phase 2 — Implementation Plan + Dataset Links (DUBAI METRO)

**Project:** Dubai Metro Passenger Forecasting (LSTM + RNN)
**Date:** 2026-06-26
**Data decision:** Hybrid — REAL Dubai Pulse (RTA) transaction data as the backbone, with a
Dubai-calibrated layer only to fill gaps. User will register on Dubai Pulse and download the
real CSVs into `data/raw/`.

---

## 0. Why Dubai (not Hangzhou)
Hangzhou patterns don't transfer to Dubai (climate, weekend = Sat/Sun since 2024, Ramadan
effects, event calendar, network shape). We therefore train on **real Dubai RTA open data**,
kept **current (through 2025)**.

---

## 1. The real Dubai data source

**Dubai Pulse — RTA Open Data → `rta_metro_ridership-open`**
- Page: https://www.dubaipulse.gov.ae/data/rta-rail/rta_metro_ridership-open
- **Granularity: transaction-level** (individual tap events), published as **one CSV per day**.
- **Verified columns:** `txn_type` (e.g., "Check in"), `txn_date` (YYYY-MM-DD), `txn_time`,
  `start_location`, `end_location`, `line_name`, `start_zone`, `end_zone`.
- Coverage: 2018 → present, refreshed regularly (portal noted last update May 2025).
- **Access:** requires FREE registration (portal redirects to `data.dubai` login). The user
  downloads; the notebook parses whatever daily files are present in `data/raw/`.

Companion datasets (optional context/joins):
- `rta_metro_lines-open` — https://www.dubaipulse.gov.ae/data/rta-rail/rta_metro_lines-open
- `rta_metro_stations` / capacity — via the Rail service list:
  https://www.dubaipulse.gov.ae/organisation/rta/service/rta-rail

**Reality check (what is NOT public):** Dubai does not release per-passenger smart-card IDs or
official 10-min platform-density feeds. We reconstruct intraday station flow by binning the real
tap timestamps — which is exactly what an operator's AFC pipeline does.

---

## 2. Hybrid data strategy
1. **REAL backbone:** parse the downloaded Dubai daily transaction CSVs → aggregate taps into
   **hourly (and 10-min) station-level inflow** per `start_location` / `line_name`.
   Also produce a **network-level daily series** (real totals) for the long-horizon model.
2. **Calibration anchors (real published figures):** 294.7M metro riders in 2025 (+7% YoY),
   ~2.2M/day, busiest stations BurJuman (17.8M) & Al Rigga (13.8M), Oct busiest month — used to
   sanity-check and, if the user only downloads a short window, to **extend/fill** the series so
   the demo is complete. Clearly labelled "modelled" wherever used.
3. **Result:** a fully Dubai, current dataset at two resolutions (daily network + intraday
   station), driven by real taps wherever available.

---

## 3. Scope (locked)
- **Targets:** station-level **inflow** (entries; `txn_type = Check in`) at intraday resolution,
  and **network-level daily ridership**.
- **Resolution:** hourly station-level (primary), daily network-level (secondary). 10-min optional.
- **Horizon:** next 1–6 hours (intraday) and next 1–7 days (daily).
- **Models:** stacked **LSTM** (primary), **SimpleRNN** baseline, **GRU** + **CNN-LSTM** comparisons,
  seasonal-naive baseline.
- **Metrics:** MAE, RMSE, MAPE, R².
- **Dubai features:** hour, day-of-week, **weekend = Fri/Sat→Sun (UAE 2024+ Sat/Sun)**, UAE public
  holidays, Ramadan flag, cyclical encodings, lags (recent / same-hour-yesterday / same-day-last-week).

---

## 4. Project structure
```
Metro-Passenger-Forecasting/
├── README.md
├── docs/{phase1_research.md, phase2_implementation_plan.md}
├── data/
│   ├── raw/          # <-- user drops Dubai Pulse metro_ridership_YYYY-MM-DD_*.csv here
│   └── processed/    # generated hourly/daily Dubai flow tables
├── notebook/dubai_metro_forecasting.ipynb
├── outputs/{metrics.json, forecast_sample.json, figures/}
└── frontend/         # React + TS Netlify dashboard
```

---

## 5. Notebook steps (Phase 3a)
1. Imports, seeds, versions.
2. **Load real Dubai CSVs** from `data/raw/` (auto-glob `metro_ridership_*.csv`); robust column
   mapping for the verified schema; if none found, build Dubai-calibrated series from anchors.
3. **Aggregate** taps → hourly station inflow + daily network totals; persist to `data/processed/`.
4. **EDA:** Dubai daily/weekly profile, peak hours, weekend effect, top stations (validate vs
   BurJuman/Al Rigga), monthly trend vs real 2025 figures.
5. **Feature engineering** (Dubai calendar incl. Ramadan/holidays, cyclical, lags).
6. **Windowing** + chronological train/val/test split.
7. **Models:** RNN, LSTM, GRU, CNN-LSTM (Keras).
8. **Train** (EarlyStopping, ReduceLROnPlateau).
9. **Evaluate** MAE/RMSE/MAPE/R² per model; comparison table.
10. **Visualize** actual vs predicted, residuals, peak zoom.
11. **Export** `outputs/metrics.json`, `outputs/forecast_sample.json` for the dashboard.

---

## 6. Frontend (Phase 3b — preview)
React + TypeScript (Vite) · Framer Motion · Recharts · Tailwind dark glassmorphism · Netlify.
Dubai Metro branding (Red/Green line colors). Reads exported JSON. Views: KPI hero, station
forecast chart, model comparison, station selector, metrics cards.

---

## 7. DATASET LINKS & DOWNLOAD INSTRUCTIONS (Dubai)

**PRIMARY (real, required) — Dubai Pulse RTA Metro Ridership**
1. Go to https://www.dubaipulse.gov.ae/data/rta-rail/rta_metro_ridership-open
2. Register a free account / sign in (portal now requires login).
3. Download the **daily CSV files** — for "latest data", grab as many **2025** daily files as
   practical (e.g., a few recent months; more days = better training). Files look like
   `metro_ridership_2025-05-01_00-00-00.csv`.
4. Place all downloaded CSVs in **`data/raw/`** (unzipped).

**OPTIONAL context joins**
- Metro lines: https://www.dubaipulse.gov.ae/data/rta-rail/rta_metro_lines-open
- Rail service (stations, capacity): https://www.dubaipulse.gov.ae/organisation/rta/service/rta-rail

**Fallback:** none needed — if `data/raw/` is empty or sparse, the notebook builds a
Dubai-calibrated series from the real published anchors so everything still runs end-to-end.

---

## 8. What I need from you to start Phase 3
1. **Approval of this revised (Dubai) plan.**
2. Drop your downloaded Dubai CSVs into `data/raw/` when ready (or tell me to proceed on the
   Dubai-calibrated series now and swap real files in later — identical pipeline).
3. **Helpful:** paste one CSV header row if your downloaded file's columns differ from the
   verified schema above, so I lock the loader exactly.
