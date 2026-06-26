import { motion } from "framer-motion";
import Background from "./components/Background";
import KpiCards from "./components/KpiCards";
import IntradayChart from "./components/IntradayChart";
import TopStations from "./components/TopStations";
import ModelComparison from "./components/ModelComparison";
import StationForecast from "./components/StationForecast";
import CrowdingGauge from "./components/CrowdingGauge";

import metrics from "./data/metrics.json";
import forecast from "./data/forecast_sample.json";
import eda from "./data/eda.json";
import type { MetricsFile, ForecastFile, EdaFile } from "./types";

const M = metrics as MetricsFile;
const F = forecast as ForecastFile;
const E = eda as EdaFile;

export default function App() {
  const solution = F.solution_model ?? F.best_model;
  const solR2 = (F.metrics.R2 * 100).toFixed(1);
  const naiveMae = M.metrics["Naive (persistence)"]?.MAE ?? 0;
  const solMae = F.metrics.MAE;
  const lift = naiveMae ? (((naiveMae - solMae) / naiveMae) * 100).toFixed(0) : "0";
  const redShare = (
    (E.line_split["Red Metro Line"] /
      (E.line_split["Red Metro Line"] + E.line_split["Green Metro Line"])) *
    100
  ).toFixed(0);

  const kpis = [
    { label: "Solution R² (next-hour)", value: `${solR2}%`, sub: `${solution} hybrid`, color: "#2ee08a",
      spark: M.val_mae_curves[solution]?.map((v) => -v) },
    { label: "Lift vs naive", value: `−${lift}%`, sub: "lower MAE than persistence", color: "#f4b740" },
    { label: "Real taps analysed", value: `${(E.n_taps / 1000).toFixed(0)}k`, sub: `${E.n_days} real days · ${E.n_stations} stations`, color: "#ff3b5c" },
    { label: "Red line share", value: `${redShare}%`, sub: "of check-ins (Red vs Green)", color: "#e8eefc" },
  ];

  return (
    <>
      <Background />
      <div className="app">
        <header className="hero">
          <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
            <span className="badge"><span className="dot" /> LIVE FORECAST · REAL RTA AFC DATA</span>
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, duration: 0.6 }}>
            Dubai Metro<br /><span className="grad">Next-Hour Passenger Forecasting</span>
          </motion.h1>
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3, duration: 0.6 }}>
            LSTM + RNN sequence models trained on real Dubai Pulse fare-collection taps —
            predicting station inflow an hour ahead so operators add trains <i>before</i> the surge.
          </motion.p>
          <motion.div className="pills" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }}>
            <span className="pill">Solution: LSTM + RNN hybrid</span>
            <span className="pill">TensorFlow / Keras</span>
            <span className="pill">MAE · RMSE · MAPE · R²</span>
            <span className="pill">Data through 2026</span>
          </motion.div>
        </header>

        <div className="section-title">Overview</div>
        <KpiCards items={kpis} />

        <div className="section-title">Demand signal</div>
        <div className="grid cols-2">
          <IntradayChart eda={E} />
          <TopStations eda={E} />
        </div>

        <div className="section-title">Forecasting</div>
        <div className="grid cols-2">
          <StationForecast data={F} />
          <ModelComparison data={M} solution={solution} />
        </div>

        <div className="section-title">Operations</div>
        <CrowdingGauge data={F} />

        <div className="foot">
          Built with React · TypeScript · Framer Motion · Recharts ·
          Models in TensorFlow/Keras<br />
          Data: <a href="https://www.dubaipulse.gov.ae/data/rta-rail/rta_metro_ridership-open" target="_blank" rel="noreferrer">
          Dubai Pulse — RTA Metro Ridership (Open Data)</a> · figures reflect a sampled open-data extract
        </div>
      </div>
    </>
  );
}
