import { motion } from "framer-motion";
import type { DubaiClock } from "../useDubaiClock";

export type Tab = "live" | "stations" | "network" | "model";
const TABS: { id: Tab; label: string }[] = [
  { id: "live", label: "Live" },
  { id: "stations", label: "Stations" },
  { id: "network", label: "Network" },
  { id: "model", label: "Model" },
];

export default function NavBar({ tab, setTab, clock, open }:
  { tab: Tab; setTab: (t: Tab) => void; clock: DubaiClock; open: boolean }) {
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

      <div className="nav-clock" title="Current Dubai time (GST, UTC+4)">
        <span className={`clk-dot ${open ? "live" : "closed"}`} />
        <div className="nav-clock-txt">
          <b>{clock.hhmm}</b><span>{open ? "GST · live" : "service closed"}</span>
        </div>
      </div>
    </nav>
  );
}
