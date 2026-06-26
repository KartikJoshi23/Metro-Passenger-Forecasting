import { motion } from "framer-motion";
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from "recharts";
import type { EdaFile } from "../types";

export default function IntradayChart({ eda }: { eda: EdaFile }) {
  const data = eda.hours.map((h, i) => ({
    hour: `${String(h).padStart(2, "0")}:00`,
    Weekday: eda.profile_weekday[i],
    Weekend: eda.profile_weekend[i],
  }));
  return (
    <motion.div className="glass card"
      initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }} transition={{ duration: 0.5 }}>
      <h3>Real Dubai demand — average hourly inflow</h3>
      <div className="cap">Check-ins per station-hour · morning & evening peaks from real AFC taps</div>
      <ResponsiveContainer width="100%" height={300}>
        <AreaChart data={data} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
          <defs>
            <linearGradient id="wd" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#ff3b5c" stopOpacity={0.55} />
              <stop offset="100%" stopColor="#ff3b5c" stopOpacity={0.02} />
            </linearGradient>
            <linearGradient id="we" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#2ee08a" stopOpacity={0.45} />
              <stop offset="100%" stopColor="#2ee08a" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(125,155,220,0.10)" vertical={false} />
          <XAxis dataKey="hour" tick={{ fill: "#8da2c8", fontSize: 11 }} interval={2} />
          <YAxis tick={{ fill: "#8da2c8", fontSize: 11 }} />
          <Tooltip content={<Box />} />
          <Legend wrapperStyle={{ fontSize: 12, color: "#8da2c8" }} />
          <Area type="monotone" dataKey="Weekday" stroke="#ff3b5c" strokeWidth={2.4} fill="url(#wd)" />
          <Area type="monotone" dataKey="Weekend" stroke="#2ee08a" strokeWidth={2.4} fill="url(#we)" />
        </AreaChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

function Box({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip-box">
      <div style={{ marginBottom: 6, color: "#e8eefc" }}>{label}</div>
      {payload.map((p: any) => (
        <div className="t-row" key={p.name}>
          <span style={{ color: p.color }}>{p.name}</span>
          <span>{p.value}</span>
        </div>
      ))}
    </div>
  );
}
