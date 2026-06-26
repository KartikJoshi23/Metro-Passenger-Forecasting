import { motion } from "framer-motion";
import type { EdaFile } from "../types";

export default function TopStations({ eda }: { eda: EdaFile }) {
  const top = eda.top_stations.slice(0, 8);
  const max = Math.max(...top.map((t) => t.checkins));
  return (
    <motion.div className="glass card"
      initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }} transition={{ duration: 0.5 }}>
      <h3>Busiest stations</h3>
      <div className="cap">Ranked by real check-ins — matches Dubai Metro's known hotspots</div>
      <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
        {top.map((t, i) => (
          <div key={t.station}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.86rem", marginBottom: 5 }}>
              <span>{i + 1}. {t.station}</span>
              <span style={{ color: "#8da2c8" }}>{t.checkins.toLocaleString()}</span>
            </div>
            <div className="bar-track">
              <motion.div className="bar-fill"
                style={{ background: "linear-gradient(90deg,#e4002b,#f4b740)" }}
                initial={{ width: 0 }} whileInView={{ width: `${(t.checkins / max) * 100}%` }}
                viewport={{ once: true }} transition={{ delay: i * 0.05, duration: 0.7 }} />
            </div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
