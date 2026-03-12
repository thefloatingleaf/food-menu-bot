"use client";

import { Cell, Pie, PieChart, ResponsiveContainer, Tooltip } from "recharts";

import type { ResultPayload } from "@/lib/types";

const chartColors: Record<"V" | "P" | "K", string> = {
  V: "var(--chart-v)",
  P: "var(--chart-p)",
  K: "var(--chart-k)",
};

type PieResultsChartProps = {
  data: ResultPayload["charts"]["lifetime"];
  title: string;
};

export function PieResultsChart({ data, title }: PieResultsChartProps) {
  return (
    <div className="stack" aria-label={`${title} pie chart`}>
      <div style={{ width: "100%", height: 240 }}>
        <ResponsiveContainer>
          <PieChart>
            <Pie
              data={data}
              dataKey="value"
              nameKey="key"
              innerRadius={56}
              outerRadius={92}
              paddingAngle={4}
              stroke="rgba(255, 250, 242, 0.8)"
              strokeWidth={2}
            >
              {data.map((entry) => (
                <Cell key={entry.key} fill={chartColors[entry.key]} />
              ))}
            </Pie>
            <Tooltip
              formatter={(value, key) => [`${value ?? 0}`, `${String(key)} total`]}
              contentStyle={{
                borderRadius: 16,
                border: "1px solid rgba(129, 104, 73, 0.14)",
                background: "rgba(255, 252, 245, 0.98)",
              }}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="legend" aria-hidden="true">
        {data.map((entry) => (
          <span className="legend__item" key={entry.key}>
            <span className="legend__swatch" style={{ backgroundColor: chartColors[entry.key] }} />
            {entry.key} • {entry.value}
          </span>
        ))}
      </div>
    </div>
  );
}
