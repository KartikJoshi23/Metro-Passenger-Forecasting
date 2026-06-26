import { motion } from "framer-motion";
import Background from "./components/Background";
import KpiCards from "./components/KpiCards";
import IntradayChart from "./components/IntradayChart";
import TopStations from "./components/TopStations";
import LiveForecast from "./components/LiveForecast";
import AllStationsCrowding from "./components/AllStationsCrowding";
import ModelComparison from "./components/ModelComparison";

import metrics from "./data/metrics.json";
import live from "./data/live_forecast.json";
import eda from "./data/eda.json";
import type { MetricsFile, LiveFile, EdaFile } from "./types";

const M = metrics as MetricsFile;
const L = live as LiveFile;
const E = eda as EdaFile;

export default function App() {
  const solution = M.solution_model ?? L.solution_model;
  const solR2 = (L.metrics.R2 * 100).toFixed(1);
  const naiveMae = M.metrics["Naive (persistence)"]?.MAE ?? 0;
  const solMae = L.metrics.MAE;
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
    { label: "Stations forecast", value: `${L.stations.length}`, sub: "entire Dubai network, live", color: "#ff3b5c" },
    { label: "Real taps analysed", value: `${(E.n_taps / 1000).toFixed(0)}k`, sub: `${E.n_days} real days · 2026 data`, color: "#9cc0ff" },
  ];

  return (
    <>
      <Background />
      <div className="app">
        <header className="hero">
          <motion.div initial={{ opacity: 0, y: -12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
            <span className="badge"><span className="dot" /> LIVE FORECAST · REAL RTA AFC DATA · 2026</span>
          </motion.div>
          <motion.h1 initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1, duration: 0.6 }}>
            Dubai Metro<br /><span className="grad">Next-Hour Passenger Forecasting</span>
          </motion.h1>
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3, duration: 0.6 }}>
            An <b>LSTM + RNN</b> hybrid trained on real Dubai Pulse fare-collection taps — forecasting
            inflow an hour ahead across <b>every station</b> so operators add trains <i>before</i> the surge.
          </motion.p>
          <motion.div className="pills" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.45 }}>
            <span className="pill">Solution: LSTM + RNN hybrid</span>
            <span className="pill">{L.stations.length} stations · network-wide</span>
            <span className="pill">TensorFlow / Keras</span>
            <span className="pill">Latest 2026 data</span>
          </motion.div>
        </header>

        <div className="section-title">Overview</div>
        <KpiCards items={kpis} />

        <div className="section-title">Live forecasting</div>
        <LiveForecast live={L} />

        <div className="section-title">Operations — crowding across Dubai</div>
        <AllStationsCrowding live={L} />

        <div className="section-title">Demand signal</div>
        <div className="grid cols-2">
          <IntradayChart eda={E} />
          <TopStations eda={E} />
        </div>

        <div className="section-title">Model comparison</div>
        <ModelComparison data={M} solution={solution} />

        <div className="foot">
          Built with <span style={{ color: "#ff3b5c" }}>❤</span> by <b>Kartik</b> • Prem • Gagandeep • Sam<br />
          <span style={{ opacity: 0.85 }}>
            Built with React · TypeScript · Framer Motion · Recharts · Models in TensorFlow/Keras
          </span><br />
          Data: <a href="https://www.dubaipulse.gov.ae/data/rta-rail/rta_metro_ridership-open" target="_blank" rel="noreferrer">
          Dubai Pulse — RTA Metro Ridership (Open Data)</a> · figures reflect a sampled open-data extract
        </div>
      </div>
    </>
  );
}
