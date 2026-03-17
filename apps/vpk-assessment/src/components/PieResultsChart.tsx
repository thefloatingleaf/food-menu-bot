"use client";

import { Cell, Pie, PieChart, ResponsiveContainer } from "recharts";

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
    <div className="result-pie" aria-label={`${title} constitution profile`}>
      <div className="result-pie__frame">
        <div className="result-pie__pulse" />
        <div className="result-pie__chart">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={data}
                dataKey="value"
                nameKey="key"
                innerRadius={70}
                outerRadius={118}
                startAngle={110}
                endAngle={-250}
                paddingAngle={4}
                stroke="rgba(255, 250, 242, 0.95)"
                strokeWidth={2}
                isAnimationActive
                animationBegin={0}
                animationDuration={1800}
                animationEasing="ease-out"
              >
                {data.map((entry) => (
                  <Cell key={entry.key} fill={chartColors[entry.key]} />
                ))}
              </Pie>
              <Pie
                data={data}
                dataKey="value"
                nameKey="key"
                innerRadius={58}
                outerRadius={64}
                startAngle={110}
                endAngle={-250}
                paddingAngle={2}
                stroke="transparent"
                isAnimationActive
                animationBegin={260}
                animationDuration={1500}
                animationEasing="ease-out"
              >
                {data.map((entry) => (
                  <Cell key={`${entry.key}-inner`} fill={chartColors[entry.key]} fillOpacity={0.28} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>
        <div className="result-pie__core" />
      </div>
      <div className="result-pie__legend" aria-hidden="true">
        {data.map((entry) => (
          <span className="result-pie__legend-item" key={entry.key}>
            <span
              className="result-pie__swatch"
              style={{ backgroundColor: chartColors[entry.key] }}
            />
            {entry.key}
          </span>
        ))}
      </div>
    </div>
  );
}
