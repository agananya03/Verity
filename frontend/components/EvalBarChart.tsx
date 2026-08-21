"use client";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";

interface MetricData {
  metric: string;
  Baseline: number;
  Improved: number;
}

interface EvalBarChartProps {
  data: MetricData[];
}

export function EvalBarChart({ data }: EvalBarChartProps) {
  return (
    <div className="w-full h-80 sm:h-96 pt-4">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 20, right: 20, left: -10, bottom: 20 }}
          barGap={8}
          barSize={28}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
          <XAxis
            dataKey="metric"
            stroke="#94a3b8"
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            axisLine={{ stroke: "#334155" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1.0]}
            ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
            stroke="#94a3b8"
            tick={{ fill: "#94a3b8", fontSize: 12 }}
            axisLine={{ stroke: "#334155" }}
            tickLine={false}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="glass-panel p-3 rounded-xl border border-slate-800 text-xs space-y-1.5 shadow-xl">
                    <p className="font-semibold text-slate-200 border-b border-slate-800 pb-1">
                      {label}
                    </p>
                    {payload.map((entry, index) => (
                      <div key={index} className="flex items-center justify-between gap-4">
                        <span className="flex items-center gap-1.5 text-slate-300">
                          <span
                            className="w-2.5 h-2.5 rounded-full"
                            style={{ backgroundColor: entry.color }}
                          />
                          {entry.name}:
                        </span>
                        <span className="font-mono font-bold text-white">
                          {(Number(entry.value) * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                );
              }
              return null;
            }}
          />
          <Legend
            verticalAlign="top"
            align="right"
            wrapperStyle={{ paddingBottom: "12px", fontSize: "12px" }}
            formatter={(value) => (
              <span className="text-xs font-medium text-slate-300 ml-1">{value}</span>
            )}
          />
          <Bar
            dataKey="Baseline"
            fill="#64748b"
            radius={[6, 6, 0, 0]}
            name="Baseline Pipeline"
          />
          <Bar
            dataKey="Improved"
            fill="#10b981"
            radius={[6, 6, 0, 0]}
            name="Improved Verity Graph"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
