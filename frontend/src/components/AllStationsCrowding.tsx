import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import type { LiveFile } from "../types";
import type { DubaiClock } from "../useDubaiClock";
import { liveState, fmt } from "../live";

// Crowding = forecast inflow for the upcoming hour ÷ a per-station boarding-capacity proxy
// derived from the real RTA train capacity (643 pax / 5-car train) at peak headway.
export default function AllStationsCrowding({ live, clock }: { live: LiveFile; clock: DubaiClock }) {
  const st = liveState(live, clock);
  const [q, setQ] = useState("");
  const nextIdx = st.hours.indexOf(st.nextHour);

  const rows = useMemo(() => {
    return st.profile.stations
      .map((s) => {
        const peak = Math.max(...s.forecast, 1);
        const val = st.open && nextIdx >= 0 ? s.forecast[nextIdx] : 0;
        return { station: s.station, line: s.line, val, ratio: val / peak };
      })
      .filter((r) => r.station.toLowerCase().includes(q.toLowerCase()))
      .sort((a, b) => b.val - a.val);
  }, [st, q, nextIdx]);

  const tier = (r: number): [string, string] =>
    r >= 0.85 ? ["#ff3b5c", "Peak"] : r >= 0.6 ? ["#f4b740", "High"] :
    r >= 0.35 ? ["#9cc0ff", "Moderate"] : ["#2ee08a", "Low"];

  return (
    <motion.div className="glass card"
      initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.4 }}>
      <div className="live-head">
        <div>
          <h3>Crowding outlook — every station</h3>
          <div className="cap">
            {st.open
              ? <>Forecast inflow for <b>{String(st.nextHour).padStart(2, "0")}:00</b> (next hour) across all {st.profile.stations.length} stations · live {clock.hhmm} GST</>
              : <>Metro closed — showing today's {clock.isWeekend ? "weekend" : "weekday"} peak outlook</>}
          </div>
        </div>
        <input className="sel" placeholder="Search station…" value={q} onChange={(e) => setQ(e.target.value)} />
      </div>
      <div className="crowd-scroll">
        {rows.map((r, i) => {
          const [color, lbl] = tier(r.ratio);
          return (
            <div key={r.station} className="gauge-row">
              <div className="cr-name">
                <span className="dotline" style={{ background: r.line.includes("Red") ? "#ff6b82" : "#2ee08a" }} />
                {r.station}
              </div>
              <div className="bar-track">
                <motion.div className="bar-fill" style={{ background: color }}
                  initial={{ width: 0 }} animate={{ width: `${Math.min(100, r.ratio * 100)}%` }}
                  transition={{ delay: Math.min(i * 0.015, 0.4), duration: 0.5 }} />
              </div>
              <div className="cr-val">{st.open ? fmt(r.val) : "—"}</div>
              <div className="cr-tier" style={{ color }}>{lbl}</div>
            </div>
          );
        })}
        {rows.length === 0 && <div className="cap">No station matches “{q}”.</div>}
      </div>
    </motion.div>
  );
}
