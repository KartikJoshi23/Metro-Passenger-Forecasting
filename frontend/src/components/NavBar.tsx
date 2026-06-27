import { useState } from "react";
import { motion } from "framer-motion";
import type { DubaiClock } from "../useDubaiClock";
import type { Demo } from "../App";

export type Tab = "live" | "stations" | "network" | "model";
const TABS: { id: Tab; label: string }[] = [
  { id: "live", label: "Live" },
  { id: "stations", label: "Stations" },
  { id: "network", label: "Network" },
  { id: "model", label: "Model" },
];

export default function NavBar({ tab, setTab, clock, open, demo, setDemo }:
  { tab: Tab; setTab: (t: Tab) => void; clock: DubaiClock; open: boolean;
    demo: Demo; setDemo: (d: Demo | ((d: Demo) => Demo)) => void }) {
  const [panel, setPanel] = useState(false);

  return (
    <nav className="nav">
      <div className="nav-brand">
        <span className="nav-logo"><span className="nl-red" /><span className="nl-green" /></span>
        <div className="nav-title">
          <b>Dubai Metro</b><span>Next-Hour Forecasting · LSTM + RNN</span>
        </div>
      </div>

      <div className="nav-tabs">
        {TABS.map((t) => (
          <button key={t.id} className={`nav-tab ${tab === t.id ? "on" : ""}`} onClick={() => setTab(t.id)}>
            {t.label}
            {tab === t.id && <motion.span layoutId="tab-underline" className="nav-underline" />}
          </button>
        ))}
      </div>

      <div className="nav-right">
        {/* presentation time-override: off by default → real time, zero effect */}
        <button className={`demo-btn ${demo.on ? "on" : ""}`} onClick={() => setPanel((p) => !p)}
          title="Demo time override for presentations">
          {demo.on ? "◷ Demo time" : "◷ Demo"}
        </button>

        <div className="nav-clock" title="Current Dubai time (GST, UTC+4)">
          <span className={`clk-dot ${open ? "live" : "closed"}`} />
          <div className="nav-clock-txt">
            <b>{clock.hhmm}</b>
            <span>{clock.isDemo ? "DEMO · simulated" : open ? "GST · live" : "service closed"}</span>
          </div>
        </div>

        {panel && (
            <motion.div key="demo-panel" className="demo-panel"
              initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }}>
              <div className="dp-head">
                <b>Demo time</b>
                <label className="dp-switch">
                  <input type="checkbox" checked={demo.on}
                    onChange={(e) => setDemo((d) => ({ ...d, on: e.target.checked }))} />
                  <span>{demo.on ? "On" : "Off"}</span>
                </label>
              </div>
              <div className="dp-row">
                <span>Hour</span>
                <input type="range" min={0} max={23} value={demo.hour} disabled={!demo.on}
                  onChange={(e) => setDemo((d) => ({ ...d, hour: Number(e.target.value) }))} />
                <b className="dp-hour">{String(demo.hour).padStart(2, "0")}:00</b>
              </div>
              <div className="dp-row">
                <span>Day type</span>
                <div className="dp-seg">
                  {([["Auto", null], ["Weekday", false], ["Weekend", true]] as const).map(([lbl, val]) => (
                    <button key={lbl} className={demo.weekend === val ? "on" : ""}
                      disabled={!demo.on}
                      onClick={() => setDemo((d) => ({ ...d, weekend: val }))}>{lbl}</button>
                  ))}
                </div>
              </div>
              <button className="dp-reset" onClick={() => { setDemo({ on: false, hour: 8, weekend: null }); setPanel(false); }}>
                ↺ Use live Dubai time
              </button>
            </motion.div>
          )}
      </div>
    </nav>
  );
}
