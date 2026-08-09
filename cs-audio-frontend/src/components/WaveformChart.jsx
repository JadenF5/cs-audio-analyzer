// WaveformChart.jsx
// Two-panel chart: original vs reconstructed waveform + reconstruction error.

import {
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
} from "recharts";

// Custom tooltip to keep it clean and dark-themed
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "#141920",
      border: "1px solid #252e3a",
      borderRadius: 6,
      padding: "8px 12px",
      fontFamily: "var(--font-mono)",
      fontSize: 11,
    }}>
      <div style={{ color: "#6b7a8f", marginBottom: 4 }}>
        t = {parseFloat(label).toFixed(4)}s
      </div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value.toFixed(4)}
        </div>
      ))}
    </div>
  );
};

export default function WaveformChart({ result }) {
  // Zip time, original, reconstructed → [{t, orig, rec, err}]
  const data = result.time_axis.map((t, i) => ({
    t: t.toFixed(4),
    Original:      result.original[i],
    Reconstructed: result.reconstructed[i],
    Error:         result.original[i] - result.reconstructed[i],
  }));

  const axisStyle = { fill: "#6b7a8f", fontSize: 10, fontFamily: "var(--font-mono)" };

  return (
    <div className="chart-card">
      <div className="chart-header">
        <span className="chart-title">Waveform — Original vs Reconstructed</span>
        <div className="chart-legend">
          <div className="legend-item">
            <div className="legend-dot" style={{ background: "#60a5fa" }} />
            Original
          </div>
          <div className="legend-item">
            <div className="legend-dot" style={{ background: "#34d399" }} />
            Reconstructed
          </div>
        </div>
      </div>

      {/* Overlay chart */}
      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -10, bottom: 4 }}>
          <CartesianGrid stroke="#1e2530" strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="t"
            tickFormatter={(v) => `${v}s`}
            tick={axisStyle}
            tickLine={false}
            axisLine={{ stroke: "#1e2530" }}
            interval="preserveStartEnd"
          />
          <YAxis
            tick={axisStyle}
            tickLine={false}
            axisLine={{ stroke: "#1e2530" }}
            width={40}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line
            type="monotone"
            dataKey="Original"
            stroke="#60a5fa"
            strokeWidth={2}
            dot={false}
            strokeOpacity={0.9}
          />
          <Line
            type="monotone"
            dataKey="Reconstructed"
            stroke="#34d399"
            strokeWidth={1.5}
            dot={false}
            strokeDasharray="4 2"
            strokeOpacity={0.85}
          />
        </LineChart>
      </ResponsiveContainer>

      {/* Error chart */}
      <div style={{ marginTop: 16 }}>
        <div style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "#6b7a8f",
          textTransform: "uppercase",
          letterSpacing: ".1em",
          marginBottom: 8,
        }}>
          Reconstruction Error (Original − Reconstructed)
        </div>
        <ResponsiveContainer width="100%" height={100}>
          <LineChart data={data} margin={{ top: 4, right: 8, left: -10, bottom: 4 }}>
            <CartesianGrid stroke="#1e2530" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="t"
              tick={axisStyle}
              tickLine={false}
              axisLine={{ stroke: "#1e2530" }}
              tickFormatter={(v) => `${v}s`}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={axisStyle}
              tickLine={false}
              axisLine={{ stroke: "#1e2530" }}
              width={40}
            />
            <ReferenceLine y={0} stroke="#252e3a" />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="Error"
              stroke="#f97066"
              strokeWidth={1.2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
