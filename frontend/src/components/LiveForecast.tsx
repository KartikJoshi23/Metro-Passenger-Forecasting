import { useState, useEffect, useMemo, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, Legend,
} from "recharts";
import type { LiveFile } from "../types";

const NETWORK = "Network · all Dubai";

export default function LiveForecast({ live }: { live: LiveFile }) {
  const hours = live.hours;
  const [now, setNow] = useState(2);            // current hour index
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1100);     // ms per hour
  const [target, setTarget] = useState(NETWORK);
  const timer = useRef<number | null>(null);

  // advance the simulated clock
  useEffect(() => {
    if (!playing) return;
    timer.current = window.setInterval(() => {
      setNow((n) => (n + 1) % hours.length);
    }, speed);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [playing, speed, hours.length]);

  const series = useMemo(() => {
    if (target === NETWORK) return { actual: live.network.actual, predicted: live.network.predicted };
    const s = live.stations.find((x) => x.station === target)!;
    return { actual: s.actual, predicted: s.predicted };
  }, [target, live]);

  const data = hours.map((h, i) => ({
    hour: `${String(h).padStart(2, "0")}:00`,
    Actual: i <= now ? series.actual[i] : null,            // revealed up to "now"
    Forecast: i <= now + 1 ? series.predicted[i] : null,   // one hour ahead = the live forecast
  }));

  const nowH = hours[now];
  const nextH = hours[Math.min(now + 1, hours.length - 1)];
  const nextPred = Math.round(series.predicted[Math.min(now + 1, hours.length - 1)]);
  const nowActual = Math.round(series.actual[now]);
  const peak = Math.max(...series.predicted);
  const surge = nextPred > 0.82 * peak;

  return (
    <motion.div className="glass card live"
      initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }} transition={{ duration: 0.5 }}>
      <div className="live-head">
        <div>
          <h3>
            <span className="live-pill"><span className="dot" /> LIVE</span>
            Next-hour forecast — replaying {live.day}
          </h3>
          <div className="cap">
            Our <b style={{ color: "#2ee08a" }}>{live.solution_model}</b> model · the clock ticks
            through the day; the dashed line is the forecast one hour ahead of <i>now</i>.
          </div>
        </div>
        <select className="sel" value={target} onChange={(e) => setTarget(e.target.value)}>
          <option>{NETWORK}</option>
          {live.stations.map((s) => (
            <option key={s.station} value={s.station}>{s.station} · {s.line.replace(" Metro Line", "")}</option>
          ))}
        </select>
      </div>

      {/* live readout */}
      <div className="live-readout">
        <div className="lr-clock">
          <span className="lr-time">{String(nowH).padStart(2, "0")}:00</span>
          <span className="lr-label">now · {target === NETWORK ? "network" : target}</span>
        </div>
        <div className="lr-now"><span className="v">{nowActual.toLocaleString()}</span><span className="l">check-ins this hour</span></div>
        <div className={`lr-fc ${surge ? "surge" : ""}`}>
          <span className="v">{nextPred.toLocaleString()}</span>
          <span className="l">forecast for {String(nextH).padStart(2, "0")}:00</span>
        </div>
        <AnimatePresence>
          {surge && (
            <motion.div className="lr-alert"
              initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}>
              ⚠ SURGE — add trains / staff
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={data} margin={{ top: 10, right: 14, left: -6, bottom: 0 }}>
          <defs>
            <linearGradient id="liveA" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#6ea8ff" stopOpacity={0.35} />
              <stop offset="100%" stopColor="#6ea8ff" stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="rgba(125,155,220,0.10)" vertical={false} />
          <XAxis dataKey="hour" tick={{ fill: "#8da2c8", fontSize: 11 }} interval={1} />
          <YAxis tick={{ fill: "#8da2c8", fontSize: 11 }} />
          <Tooltip content={<Box />} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <ReferenceLine x={`${String(nowH).padStart(2, "0")}:00`} stroke="#2ee08a"
            strokeDasharray="3 3" label={{ value: "now", fill: "#2ee08a", fontSize: 11, position: "top" }} />
          <Area type="monotone" dataKey="Actual" stroke="#9cc0ff" strokeWidth={2.6}
            fill="url(#liveA)" connectNulls dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="Forecast" stroke="#e4002b" strokeWidth={2.8}
            strokeDasharray="6 4" connectNulls dot={{ r: 2.5, fill: "#ff3b5c" }} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="live-ctrl">
        <button className="btn" onClick={() => setPlaying((p) => !p)}>{playing ? "❚❚ Pause" : "▶ Play"}</button>
        <input type="range" min={400} max={1800} step={100} value={1900 - speed}
          onChange={(e) => setSpeed(1900 - Number(e.target.value))} />
        <span className="spd">speed</span>
        <button className="btn ghost" onClick={() => setNow(0)}>⟲ Restart</button>
      </div>
    </motion.div>
  );
}

function Box({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip-box">
      <div style={{ marginBottom: 6 }}>{label}</div>
      {payload.filter((p: any) => p.value != null).map((p: any) => (
        <div className="t-row" key={p.name}>
          <span style={{ color: p.color }}>{p.name}</span>
          <span>{Math.round(p.value).toLocaleString()}</span>
        </div>
      ))}
    </div>
  );
}
