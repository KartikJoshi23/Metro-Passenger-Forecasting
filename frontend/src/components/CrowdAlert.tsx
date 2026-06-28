import { useEffect, useState, useRef } from "react";
import { motion } from "framer-motion";
import type { LiveFile } from "../types";
import type { DubaiClock } from "../useDubaiClock";
import { liveState, fmt } from "../live";

/** Global crowd alert. Whenever the (simulated or real) clock hour changes and the metro is open,
 *  it checks which stations are forecast near their daily peak for the UPCOMING hour and pops a
 *  dismissible toast — visible on every tab. Great with the Demo time-override during a talk. */
export default function CrowdAlert({ live, clock }: { live: LiveFile; clock: DubaiClock }) {
  const [alert, setAlert] = useState<{ hour: number; stations: { name: string; val: number; line: string }[] } | null>(null);
  const lastKey = useRef<string>("");

  // recompute on every hour / day-type change
  useEffect(() => {
    const st = liveState(live, clock);
    if (!st.open) { setAlert(null); return; }
    const nextIdx = st.hours.indexOf(st.nextHour);
    if (nextIdx < 0) { setAlert(null); return; }                 // closing hour → next hour shut

    const risk = st.profile.stations
      .map((s) => ({ name: s.station, line: s.line, val: s.forecast[nextIdx],
                     ratio: s.forecast[nextIdx] / Math.max(...s.forecast, 1) }))
      .filter((s) => s.ratio >= 0.7)                              // near that station's own peak
      .sort((a, b) => b.val - a.val)
      .slice(0, 3);

    const key = `${st.nextHour}-${clock.isWeekend}`;
    if (risk.length && key !== lastKey.current) {
      lastKey.current = key;
      setAlert({ hour: st.nextHour, stations: risk });
    } else if (!risk.length) {
      setAlert(null);
    }
  }, [clock.hour, clock.isWeekend, live]);

  // auto-dismiss after 10s
  useEffect(() => {
    if (!alert) return;
    const t = setTimeout(() => setAlert(null), 10000);
    return () => clearTimeout(t);
  }, [alert]);

  if (!alert) return null;
  const names = alert.stations.map((s) => s.name).join(", ");
  return (
    <motion.div className="crowd-alert" key={`${alert.hour}-${names}`}
      initial={{ opacity: 0, y: -24, scale: 0.96 }} animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ type: "spring", stiffness: 260, damping: 22 }}>
      <span className="ca-icon">⚠</span>
      <div className="ca-body">
        <div className="ca-title">
          Crowd alert · <b>{String(alert.hour).padStart(2, "0")}:00</b>
        </div>
        <div className="ca-text">
          High demand expected at <b>{names}</b> — consider adding trains / staff.
        </div>
        <div className="ca-chips">
          {alert.stations.map((s) => (
            <span key={s.name} className="ca-chip" style={{ borderColor: s.line.includes("Red") ? "#ff6b82" : "#2ee08a" }}>
              {s.name} · ~{fmt(s.val)}
            </span>
          ))}
        </div>
      </div>
      <button className="ca-close" onClick={() => setAlert(null)} aria-label="dismiss">×</button>
    </motion.div>
  );
}
