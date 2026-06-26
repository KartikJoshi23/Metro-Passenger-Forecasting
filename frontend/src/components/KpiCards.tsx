import { motion } from "framer-motion";
import { ResponsiveContainer, AreaChart, Area } from "recharts";

interface Kpi {
  label: string;
  value: string;
  sub: string;
  color: string;
  spark?: number[];
}

export default function KpiCards({ items }: { items: Kpi[] }) {
  return (
    <div className="grid cols-4">
      {items.map((k, i) => (
        <motion.div
          key={k.label}
          className="glass kpi"
          initial={{ opacity: 0, y: 22 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: i * 0.08, duration: 0.5 }}
        >
          <div className="label">{k.label}</div>
          <div className="value" style={{ color: k.color }}>{k.value}</div>
          <div className="sub">{k.sub}</div>
          {k.spark && (
            <div className="spark" style={{ height: 34 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={k.spark.map((v, x) => ({ x, v }))}>
                  <defs>
                    <linearGradient id={`sp${i}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor={k.color} stopOpacity={0.5} />
                      <stop offset="100%" stopColor={k.color} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="v" stroke={k.color} strokeWidth={2}
                    fill={`url(#sp${i})`} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </motion.div>
      ))}
    </div>
  );
}
