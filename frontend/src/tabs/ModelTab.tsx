import { motion } from "framer-motion";
import {
  ResponsiveContainer, ComposedChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
} from "recharts";
import ModelComparison from "../components/ModelComparison";
import type { MetricsFile, LiveFile } from "../types";

export default function ModelTab({ metrics, live }: { metrics: MetricsFile; live: LiveFile }) {
  const solution = metrics.solution_model ?? live.solution_model;
  const v = live.validation;
  const data = live.hours.map((h, i) => ({
    hour: `${String(h).padStart(2, "0")}`, Actual: v.actual[i], Forecast: v.predicted[i],
  }));

  return (
    <div className="tab-stack">
      <div className="grid cols-2">
        <ModelComparison data={metrics} solution={solution} />

        <motion.div className="glass card"
          initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
          <h3>Validation — real day, actual vs forecast</h3>
          <div className="cap">
            Held-out test day {v.day} · network inflow · correlation <b style={{ color: "#2ee08a" }}>{v.corr.toFixed(2)}</b>.
            This is the model on <i>unseen real</i> taps (not a typical-day profile).
          </div>
          <ResponsiveContainer width="100%" height={272}>
            <ComposedChart data={data} margin={{ top: 8, right: 14, left: -6, bottom: 0 }}>
              <CartesianGrid stroke="rgba(125,155,220,0.10)" vertical={false} />
              <XAxis dataKey="hour" tick={{ fill: "#8da2c8", fontSize: 11 }} interval={1} />
              <YAxis tick={{ fill: "#8da2c8", fontSize: 11 }} />
              <Tooltip content={<Box />} />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line type="monotone" dataKey="Actual" stroke="#9cc0ff" strokeWidth={2.6} dot={{ r: 2.5 }} isAnimationActive={false} />
              <Line type="monotone" dataKey="Forecast" stroke="#e4002b" strokeWidth={2.6} strokeDasharray="6 4" dot={{ r: 2.5 }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      <motion.div className="glass card"
        initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ duration: 0.4 }}>
        <h3>How the model works</h3>
        <div className="method">
          <div className="m-step"><span className="m-n">1</span><div><b>Real, latest data.</b> Dubai Pulse RTA fare-collection taps, <b>2026 only</b> — the current operating regime — aggregated to hourly inflow per station (05:00–24:00).</div></div>
          <div className="m-step"><span className="m-n">2</span><div><b>Features.</b> Per-station scaling, a learned <b>station embedding</b>, a leakage-safe <b>climatology</b> prior (station × hour × weekend, train days only), today's <b>level</b> ratio, and cyclical time + UAE weekend (Sat/Sun).</div></div>
          <div className="m-step"><span className="m-n">3</span><div><b>The hybrid.</b> An <b>LSTM(64) → SimpleRNN(32)</b> encoder reads the last 6 hours; an embedding/context head predicts the next hour. Trained with Huber loss, early stopping, chronological split.</div></div>
          <div className="m-step"><span className="m-n">4</span><div><b>Result.</b> Best RMSE &amp; R² and tied-best MAE vs LSTM/RNN/GRU/CNN-LSTM, clearly beating the Naive and Climatology baselines — network tracking correlation {live.validation.corr.toFixed(2)}.</div></div>
        </div>
      </motion.div>
    </div>
  );
}

function Box({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip-box">
      <div style={{ marginBottom: 6 }}>{label}:00</div>
      {payload.map((p: any) => (
        <div className="t-row" key={p.name}><span style={{ color: p.color }}>{p.name}</span>
          <span>{Math.round(p.value).toLocaleString()}</span></div>
      ))}
    </div>
  );
}
