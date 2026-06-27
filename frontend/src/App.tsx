import { useState } from "react";
import Background from "./components/Background";
import NavBar, { type Tab } from "./components/NavBar";
import Footer from "./components/Footer";
import LiveTab from "./tabs/LiveTab";
import StationsTab from "./tabs/StationsTab";
import NetworkTab from "./tabs/NetworkTab";
import ModelTab from "./tabs/ModelTab";
import { useDubaiClock } from "./useDubaiClock";

import metrics from "./data/metrics.json";
import live from "./data/live_forecast.json";
import eda from "./data/eda.json";
import type { MetricsFile, LiveFile, EdaFile } from "./types";

const M = metrics as MetricsFile;
const L = live as unknown as LiveFile;
const E = eda as EdaFile;

export interface Demo { on: boolean; hour: number; weekend: boolean | null }

export default function App() {
  const [tab, setTab] = useState<Tab>("live");
  const [demo, setDemo] = useState<Demo>({ on: false, hour: 8, weekend: null });
  const clock = useDubaiClock(demo.on ? { hour: demo.hour, weekend: demo.weekend } : undefined);
  const open = clock.hour >= L.operating[0] && clock.hour < L.operating[1];

  return (
    <>
      <Background />
      <div className={`shell ${tab === "live" ? "no-scroll" : ""}`}>
        <NavBar tab={tab} setTab={setTab} clock={clock} open={open} demo={demo} setDemo={setDemo} />
        <main className="tab-body">
          {tab === "live" && <LiveTab live={L} eda={E} clock={clock} />}
          {tab === "stations" && <><StationsTab live={L} clock={clock} /><Footer /></>}
          {tab === "network" && <><NetworkTab live={L} eda={E} clock={clock} /><Footer /></>}
          {tab === "model" && <><ModelTab metrics={M} live={L} /><Footer /></>}
        </main>
        {tab === "live" && <Footer slim />}
      </div>
    </>
  );
}
