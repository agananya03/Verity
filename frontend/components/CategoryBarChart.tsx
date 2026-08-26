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

export interface CategoryChartData {
  category: string;
  displayCategory: string;
  Baseline: number;
  Improved: number;
}

interface CategoryBarChartProps {
  data: CategoryChartData[];
}

export function CategoryBarChart({ data }: CategoryBarChartProps) {
  return (
    <div className="w-full h-80 sm:h-96 pt-4 font-mono">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 20, right: 20, left: -10, bottom: 20 }}
          barGap={8}
          barSize={32}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#2A3138" vertical={false} />
          <XAxis
            dataKey="displayCategory"
            stroke="#8B95A1"
            tick={{ fill: "#8B95A1", fontSize: 11, fontFamily: "var(--font-mono)" }}
            axisLine={{ stroke: "#2A3138" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1.0]}
            ticks={[0, 0.2, 0.4, 0.6, 0.8, 1.0]}
            stroke="#8B95A1"
            tick={{ fill: "#8B95A1", fontSize: 11, fontFamily: "var(--font-mono)" }}
            axisLine={{ stroke: "#2A3138" }}
            tickLine={false}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          />
          <Tooltip
            content={({ active, payload, label }) => {
              if (active && payload && payload.length) {
                return (
                  <div className="bg-[#0F1419] p-3 rounded-sm border border-[#2A3138] text-xs font-mono space-y-1.5 shadow-xl">
                    <p className="font-semibold text-[#EDE7DA] border-b border-[#2A3138] pb-1">
                      {label}
                    </p>
                    {payload.map((entry, index) => (
                      <div key={index} className="flex items-center justify-between gap-4">
                        <span className="flex items-center gap-1.5 text-[#8B95A1]">
                          <span
                            className="w-2.5 h-2.5 rounded-none"
                            style={{ backgroundColor: entry.color }}
                          />
                          {entry.name}:
                        </span>
                        <span className="font-bold text-[#EDE7DA]">
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
            wrapperStyle={{ paddingBottom: "12px", fontSize: "11px", fontFamily: "var(--font-mono)" }}
            formatter={(value) => (
              <span className="text-xs font-mono text-[#8B95A1] ml-1">{value}</span>
            )}
          />
          <Bar
            dataKey="Baseline"
            fill="#2A3138"
            radius={[0, 0, 0, 0]}
            name="Baseline (Vector-Only)"
          />
          <Bar
            dataKey="Improved"
            fill="#E8A33D"
            radius={[0, 0, 0, 0]}
            name="Improved (Verity Graph)"
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
