import { motion } from "framer-motion";
import {
  ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from "recharts";
import type { LiveFile, EdaFile } from "../types";
import type { DubaiClock } from "../useDubaiClock";
import { liveState, band, fmt } from "../live";

export default function LiveTab({ live, eda, clock }:
  { live: LiveFile; eda: EdaFile; clock: DubaiClock }) {
  const st = liveState(live, clock);
  const net = st.profile.network;
  const peak = Math.max(...net.forecast);

  // forecast for the upcoming hour
  const nextIdx = st.hours.indexOf(st.nextHour);
  const nextVal = nextIdx >= 0 ? net.forecast[nextIdx] : 0;
  const nowVal = st.nowIdx >= 0 ? net.forecast[st.nowIdx] : 0;
  const delta = nowVal > 0 ? Math.round(((nextVal - nowVal) / nowVal) * 100) : 0;
  const b = band(nextVal, peak);

  const data = st.hours.map((h, i) => ({
    hour: `${String(h).padStart(2, "0")}`,
    Typical: net.typical[i],
    Forecast: net.forecast[i],
    now: st.nowIdx === i ? net.forecast[i] : null,
  }));

  return (
    <div className="live-grid">
      {/* LEFT — the live forecast headline */}
      <motion.div className="glass live-hero"
        initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}>
        <div className="lh-top">
          <span className={`badge ${st.open ? "" : "off"}`}>
            <span className="dot" /> {st.open ? "LIVE · NEXT-HOUR FORECAST" : "METRO CLOSED"}
          </span>
          <div className="lh-clock">
            <span className="lh-time">{clock.hhmmss}</span>
            <span className="lh-day">{clock.weekdayName}{clock.isWeekend ? " · weekend" : ""} · Dubai GST</span>
          </div>
        </div>

        {st.open && !st.closing ? (
          <div className="lh-body">
            <div className="lh-label">
              Forecast network inflow for <b>{String(st.nextHour).padStart(2, "0")}:00</b>
            </div>
            <div className="lh-value" style={{ color: b.color }}>{fmt(nextVal)}</div>
            <div className="lh-sub">
              check-ins across all {st.profile.stations.length} stations ·
              <span style={{ color: b.color }}> {b.label} demand</span> ·
              <span style={{ color: delta >= 0 ? "#ff7d93" : "#2ee08a" }}> {delta >= 0 ? "▲" : "▼"} {Math.abs(delta)}%</span> vs this hour
            </div>
            <div className="lh-pillrow">
              <span className="pill">This hour ({String(st.nowHour).padStart(2, "0")}:00): <b>{fmt(nowVal)}</b></span>
              <span className="pill">Model: <b>{live.solution_model}</b></span>
              <span className="pill">R² <b>{(live.metrics.R2 * 100).toFixed(0)}%</b></span>
            </div>
          </div>
        ) : st.closing ? (
          <div className="lh-body">
            <div className="lh-label">Service winding down · last full hour</div>
            <div className="lh-value" style={{ color: "#f4b740" }}>{fmt(nowVal)}</div>
            <div className="lh-sub">
              forecast check-ins this hour ({String(st.nowHour).padStart(2, "0")}:00) across all
              {" "}{st.profile.stations.length} stations · last trains around midnight
            </div>
            <div className="lh-pillrow">
              <span className="pill">Model: <b>{live.solution_model}</b></span>
              <span className="pill">R² <b>{(live.metrics.R2 * 100).toFixed(0)}%</b></span>
            </div>
          </div>
        ) : (
          <div className="lh-body closed">
            <div className="lh-value closed">— —</div>
            <div className="lh-sub">
              Dubai Metro is not running now. First trains at <b>{String(st.openHour).padStart(2, "0")}:00</b>.
              Live next-hour forecasting resumes at service start.
            </div>
            <div className="lh-pillrow">
              <span className="pill">Opens {String(st.openHour).padStart(2, "0")}:00</span>
              <span className="pill">Closes 00:00</span>
            </div>
          </div>
        )}
      </motion.div>

      {/* RIGHT — today's curve with the now marker */}
      <motion.div className="glass live-chart"
        initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.08, duration: 0.5 }}>
        <div className="lc-head">
          <h3>Today's expected demand — network</h3>
          <span className="cap">{clock.isWeekend ? "Weekend" : "Weekday"} profile · the marker is now</span>
        </div>
        <ResponsiveContainer width="100%" height="78%">
          <ComposedChart data={data} margin={{ top: 8, right: 12, left: -10, bottom: 0 }}>
            <defs>
              <linearGradient id="lgTyp" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#6ea8ff" stopOpacity={0.28} />
                <stop offset="100%" stopColor="#6ea8ff" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(125,155,220,0.10)" vertical={false} />
            <XAxis dataKey="hour" tick={{ fill: "#8da2c8", fontSize: 10 }} interval={1} />
            <YAxis tick={{ fill: "#8da2c8", fontSize: 10 }} width={38} />
            <Tooltip content={<Box />} />
            {st.open && <ReferenceLine x={String(st.nowHour).padStart(2, "0")} stroke="#2ee08a"
              strokeDasharray="3 3" label={{ value: "now", fill: "#2ee08a", fontSize: 10, position: "top" }} />}
            <Area type="monotone" dataKey="Typical" stroke="#9cc0ff" strokeWidth={2} fill="url(#lgTyp)" dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="Forecast" stroke="#e4002b" strokeWidth={2.6} dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="now" stroke="#2ee08a" strokeWidth={0} dot={{ r: 5, fill: "#2ee08a" }} isAnimationActive={false} legendType="none" />
          </ComposedChart>
        </ResponsiveContainer>
      </motion.div>

      {/* BOTTOM — KPI strip */}
      <div className="live-kpis">
        {[
          { label: "Forecast accuracy (R²)", value: `${(live.metrics.R2 * 100).toFixed(1)}%`, color: "#2ee08a" },
          { label: "Network tracking (corr)", value: live.validation.corr.toFixed(2), color: "#9cc0ff" },
          { label: "Stations forecast", value: `${st.profile.stations.length}`, color: "#f4b740" },
          { label: "Real taps · 2026", value: `${(eda.n_taps / 1000).toFixed(0)}k`, color: "#ff7d93" },
        ].map((k, i) => (
          <motion.div key={k.label} className="glass kpi-sm"
            initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.12 + i * 0.06 }}>
            <div className="value" style={{ color: k.color }}>{k.value}</div>
            <div className="label">{k.label}</div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

function Box({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip-box">
      <div style={{ marginBottom: 6 }}>{label}:00</div>
      {payload.filter((p: any) => p.value != null && p.dataKey !== "now").map((p: any) => (
        <div className="t-row" key={p.name}><span style={{ color: p.color }}>{p.name}</span>
          <span>{Math.round(p.value).toLocaleString()}</span></div>
      ))}
    </div>
  );
}
