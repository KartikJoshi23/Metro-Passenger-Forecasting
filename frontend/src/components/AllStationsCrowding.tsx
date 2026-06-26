import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import type { LiveFile } from "../types";

// Real Dubai Metro train capacity (RTA capacity dataset): 5 cars, 643 passengers/train.
// Hourly boarding-capacity proxy at ~5-min peak headway; sampled extract scaled for a load index.
export default function AllStationsCrowding({ live }: { live: LiveFile }) {
  const [q, setQ] = useState("");
  const cap = live.capacity_per_train * 12 * 0.06;

  const rows = useMemo(() => {
    return live.stations
      .map((s) => {
        const peak = Math.max(...s.predicted);
        return { station: s.station, line: s.line, peak: Math.round(peak), ratio: Math.min(1.4, peak / cap) };
      })
      .filter((r) => r.station.toLowerCase().includes(q.toLowerCase()))
      .sort((a, b) => b.ratio - a.ratio);
  }, [live, q, cap]);

  const band = (r: number): [string, string] =>
    r < 0.55 ? ["#2ee08a", "Comfortable"] : r < 0.85 ? ["#f4b740", "Busy"] : ["#ff3b5c", "Crowded"];

  return (
    <motion.div className="glass card"
      initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }} transition={{ duration: 0.5 }}>
      <div className="live-head">
        <div>
          <h3>Predicted crowding — every station</h3>
          <div className="cap">
            Forecast peak ÷ train capacity (real RTA figure: {live.capacity_per_train} pax / 5-car train).
            All {live.stations.length} stations on the network — search any.
          </div>
        </div>
        <input className="sel" placeholder="Search station…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      <div className="crowd-scroll">
        {rows.map((r, i) => {
          const [color, lbl] = band(r.ratio);
          return (
            <div key={r.station} className="gauge-row">
              <div style={{ width: 168, fontSize: "0.84rem" }}>
                <span style={{ color: r.line.includes("Red") ? "#ff6b82" : "#2ee08a", marginRight: 6 }}>●</span>
                {r.station}
              </div>
              <div className="bar-track">
                <motion.div className="bar-fill" style={{ background: color }}
                  initial={{ width: 0 }} animate={{ width: `${Math.min(100, r.ratio * 100)}%` }}
                  transition={{ delay: Math.min(i * 0.02, 0.5), duration: 0.6 }} />
              </div>
              <div style={{ width: 96, textAlign: "right", fontSize: "0.76rem", color }}>{lbl}</div>
            </div>
          );
        })}
        {rows.length === 0 && <div className="cap">No station matches “{q}”.</div>}
      </div>
    </motion.div>
  );
}
