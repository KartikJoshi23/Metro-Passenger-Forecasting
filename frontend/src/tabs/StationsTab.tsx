import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import {
  ResponsiveContainer, ComposedChart, Area, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from "recharts";
import type { LiveFile } from "../types";
import type { DubaiClock } from "../useDubaiClock";
import { liveState, band, fmt } from "../live";

export default function StationsTab({ live, clock }: { live: LiveFile; clock: DubaiClock }) {
  const st = liveState(live, clock);
  const stations = st.profile.stations;
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(stations[0].station);

  const filtered = useMemo(
    () => stations.filter((s) => s.station.toLowerCase().includes(q.toLowerCase())),
    [stations, q]
  );
  const cur = stations.find((s) => s.station === sel) ?? stations[0];

  const nextIdx = st.hours.indexOf(st.nextHour);
  const nextVal = st.open && nextIdx >= 0 ? cur.forecast[nextIdx] : 0;
  const peak = Math.max(...cur.forecast);
  const b = band(nextVal, peak);

  const data = st.hours.map((h, i) => ({
    hour: String(h).padStart(2, "0"),
    Typical: cur.typical[i],
    Forecast: cur.forecast[i],
    now: st.nowIdx === i ? cur.forecast[i] : null,
  }));

  return (
    <div className="stations-grid">
      {/* station list */}
      <div className="glass st-list">
        <input className="sel full" placeholder="Search any station…" value={q} onChange={(e) => setQ(e.target.value)} />
        <div className="st-scroll">
          {filtered.map((s) => {
            const v = st.open && nextIdx >= 0 ? s.forecast[nextIdx] : 0;
            const sb = band(v, Math.max(...s.forecast));
            return (
              <button key={s.station} className={`st-item ${s.station === sel ? "on" : ""}`} onClick={() => setSel(s.station)}>
                <span className="dotline" style={{ background: s.line.includes("Red") ? "#ff6b82" : "#2ee08a" }} />
                <span className="st-name">{s.station}</span>
                <span className="st-val" style={{ color: sb.color }}>{st.open ? fmt(v) : "—"}</span>
              </button>
            );
          })}
          {filtered.length === 0 && <div className="cap" style={{ padding: 12 }}>No match for “{q}”.</div>}
        </div>
      </div>

      {/* selected station detail */}
      <motion.div className="glass st-detail" key={sel}
        initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }}>
        <div className="st-detail-head">
          <div>
            <h3>{cur.station} <span className="line-tag" style={{ color: cur.line.includes("Red") ? "#ff6b82" : "#2ee08a" }}>● {cur.line.replace(" Metro Line", "")}</span></h3>
            <div className="cap">
              {st.open
                ? <>Next-hour forecast for <b>{String(st.nextHour).padStart(2, "0")}:00</b> — anchored to live Dubai time {clock.hhmm}</>
                : <>Metro closed now · first trains {String(st.openHour).padStart(2, "0")}:00. Showing today's expected {clock.isWeekend ? "weekend" : "weekday"} profile.</>}
            </div>
          </div>
          <div className="st-big" style={{ color: st.open ? b.color : "#8da2c8" }}>
            <span className="v">{st.open ? fmt(nextVal) : "—"}</span>
            <span className="l">{st.open ? `${b.label} demand` : "service closed"}</span>
          </div>
        </div>
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={data} margin={{ top: 8, right: 14, left: -6, bottom: 0 }}>
            <defs>
              <linearGradient id="sgTyp" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#6ea8ff" stopOpacity={0.28} />
                <stop offset="100%" stopColor="#6ea8ff" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(125,155,220,0.10)" vertical={false} />
            <XAxis dataKey="hour" tick={{ fill: "#8da2c8", fontSize: 11 }} interval={1} />
            <YAxis tick={{ fill: "#8da2c8", fontSize: 11 }} />
            <Tooltip content={<Box />} />
            {st.open && <ReferenceLine x={String(st.nowHour).padStart(2, "0")} stroke="#2ee08a"
              strokeDasharray="3 3" label={{ value: "now", fill: "#2ee08a", fontSize: 11, position: "top" }} />}
            <Area type="monotone" dataKey="Typical" stroke="#9cc0ff" strokeWidth={2} fill="url(#sgTyp)" name="Typical demand" dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="Forecast" stroke="#e4002b" strokeWidth={2.6} name="Forecast" dot={false} isAnimationActive={false} />
            <Line type="monotone" dataKey="now" stroke="#2ee08a" strokeWidth={0} dot={{ r: 5, fill: "#2ee08a" }} name="now" isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </motion.div>
    </div>
  );
}

function Box({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tooltip-box">
      <div style={{ marginBottom: 6 }}>{label}:00</div>
      {payload.filter((p: any) => p.value != null && p.name !== "now").map((p: any) => (
        <div className="t-row" key={p.name}><span style={{ color: p.color }}>{p.name}</span>
          <span>{Math.round(p.value).toLocaleString()}</span></div>
      ))}
    </div>
  );
}
