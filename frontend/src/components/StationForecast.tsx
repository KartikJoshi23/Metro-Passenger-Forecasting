import { useState } from "react";
import { motion } from "framer-motion";
import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from "recharts";
import type { ForecastFile } from "../types";

export default function StationForecast({ data }: { data: ForecastFile }) {
  const [sel, setSel] = useState(0);
  const s = data.series[sel];
  const chart = s.hours.map((h, i) => ({
    hour: `${String(h).padStart(2, "0")}:00`,
    Actual: s.actual[i],
    Predicted: s.predicted[i],
  }));
  return (
    <motion.div className="glass card"
      initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }} transition={{ duration: 0.5 }}>
      <h3>Next-hour forecast vs actual</h3>
      <div className="cap">
        Our solution <b style={{ color: "#2ee08a" }}>{data.solution_model ?? data.best_model}</b> · real test day · pick a station
      </div>
      <div className="chips">
        {data.series.map((ser, i) => (
          <button key={ser.station + i}
            className={`chip ${i === sel ? "active" : ""}`}
            onClick={() => setSel(i)}>
            {ser.station}
          </button>
        ))}
      </div>
      <div style={{ fontSize: "0.78rem", color: "#8da2c8", marginBottom: 8 }}>
        {s.station} · {s.day}
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chart} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
          <CartesianGrid stroke="rgba(125,155,220,0.10)" vertical={false} />
          <XAxis dataKey="hour" tick={{ fill: "#8da2c8", fontSize: 11 }} />
          <YAxis tick={{ fill: "#8da2c8", fontSize: 11 }} />
          <Tooltip content={<Box />} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line type="monotone" dataKey="Actual" stroke="#e8eefc" strokeWidth={2.6}
            dot={{ r: 3, fill: "#e8eefc" }} />
          <Line type="monotone" dataKey="Predicted" stroke="#e4002b" strokeWidth={2.6}
            strokeDasharray="6 4" dot={{ r: 3, fill: "#ff3b5c" }} />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

function Box({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip-box">
      <div style={{ marginBottom: 6 }}>{label}</div>
      {payload.map((p: any) => (
        <div className="t-row" key={p.name}>
          <span style={{ color: p.color }}>{p.name}</span>
          <span>{Math.round(p.value)}</span>
        </div>
      ))}
    </div>
  );
}
