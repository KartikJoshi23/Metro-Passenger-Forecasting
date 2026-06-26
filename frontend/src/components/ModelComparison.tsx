import { motion } from "framer-motion";
import type { MetricsFile } from "../types";

export default function ModelComparison({ data, solution }: { data: MetricsFile; solution: string }) {
  const baseline = "Naive (persistence)";
  // Solution first, then other approaches sorted by MAE, baseline last.
  const rows = Object.entries(data.metrics).sort((a, b) => {
    if (a[0] === solution) return -1;
    if (b[0] === solution) return 1;
    if (a[0] === baseline) return 1;
    if (b[0] === baseline) return -1;
    return a[1].MAE - b[1].MAE;
  });
  return (
    <motion.div className="glass card"
      initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }} transition={{ duration: 0.5 }}>
      <h3>Approach comparison</h3>
      <div className="cap">
        Our <b style={{ color: "#2ee08a" }}>LSTM + RNN</b> solution vs other approaches ·
        {" "}{data.test_windows.toLocaleString()} held-out test windows · lower error = better
      </div>
      <table className="mtable">
        <thead>
          <tr><th>Approach</th><th>MAE</th><th>RMSE</th><th>MAPE %</th><th>R²</th></tr>
        </thead>
        <tbody>
          {rows.map(([name, m]) => {
            const isSol = name === solution;
            return (
              <tr key={name} style={isSol ? { background: "rgba(46,224,138,0.07)" } : undefined}>
                <td>
                  {name}
                  {isSol && <span className="tag-best">OUR SOLUTION</span>}
                  {name === baseline && <span className="tag-base">baseline</span>}
                </td>
                <td className={isSol ? "best" : ""}>{m.MAE}</td>
                <td>{m.RMSE}</td>
                <td>{m.MAPE ?? "—"}</td>
                <td className={isSol ? "best" : ""}>{m.R2}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </motion.div>
  );
}
