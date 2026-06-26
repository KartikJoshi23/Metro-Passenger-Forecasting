import { motion } from "framer-motion";
import type { ForecastFile } from "../types";

// Real Dubai Metro train capacity (RTA capacity dataset): 5 cars, 643 passengers/train.
const TRAIN_CAP = 643;
const TRAINS_PER_HOUR = 12; // ~5-min headway at peak → hourly platform throughput proxy

export default function CrowdingGauge({ data }: { data: ForecastFile }) {
  const rows = data.series.map((s) => {
    const peak = Math.max(...s.predicted);
    const capacity = TRAIN_CAP * TRAINS_PER_HOUR; // hourly boarding capacity proxy
    const ratio = Math.min(1.4, (peak / (capacity * 0.06))); // scaled sampled→load proxy
    return { station: s.station, peak: Math.round(peak), ratio };
  });
  const band = (r: number) =>
    r < 0.6 ? ["#2ee08a", "Comfortable"] : r < 0.9 ? ["#f4b740", "Busy"] : ["#ff3b5c", "Crowded"];
  return (
    <motion.div className="glass card"
      initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }} transition={{ duration: 0.5 }}>
      <h3>Predicted crowding (next-hour peak)</h3>
      <div className="cap">
        Forecast peak ÷ train capacity (real RTA figure: 643 pax / 5-car train) — the operational alert
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
        {rows.map((r, i) => {
          const [color, lbl] = band(r.ratio);
          return (
            <div key={r.station} className="gauge-row">
              <div style={{ width: 150, fontSize: "0.85rem" }}>{r.station}</div>
              <div className="bar-track">
                <motion.div className="bar-fill" style={{ background: color }}
                  initial={{ width: 0 }} whileInView={{ width: `${Math.min(100, r.ratio * 100)}%` }}
                  viewport={{ once: true }} transition={{ delay: i * 0.05, duration: 0.7 }} />
              </div>
              <div style={{ width: 96, textAlign: "right", fontSize: "0.78rem", color }}>{lbl}</div>
            </div>
          );
        })}
      </div>
    </motion.div>
  );
}
