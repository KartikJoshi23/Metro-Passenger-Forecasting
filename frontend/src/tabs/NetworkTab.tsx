import AllStationsCrowding from "../components/AllStationsCrowding";
import IntradayChart from "../components/IntradayChart";
import TopStations from "../components/TopStations";
import type { LiveFile, EdaFile } from "../types";
import type { DubaiClock } from "../useDubaiClock";

export default function NetworkTab({ live, eda, clock }:
  { live: LiveFile; eda: EdaFile; clock: DubaiClock }) {
  return (
    <div className="tab-stack">
      <AllStationsCrowding live={live} clock={clock} />
      <div className="grid cols-2">
        <IntradayChart eda={eda} />
        <TopStations eda={eda} />
      </div>
    </div>
  );
}
