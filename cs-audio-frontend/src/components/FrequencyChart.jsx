// FrequencyChart.jsx
// Side-by-side FFT magnitude charts showing sparsity in frequency domain.

import {
  BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";

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
        Bin {label}
      </div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color }}>
          {p.name}: {p.value.toFixed(4)}
        </div>
      ))}
    </div>
  );
};

export default function FrequencyChart({ result }) {
  const axisStyle = { fill: "#6b7a8f", fontSize: 10, fontFamily: "var(--font-mono)" };

  // Build data for both charts — one entry per frequency bin
  const origData = result.fft_original.map((mag, i) => ({
    bin: i,
    Magnitude: mag,
  }));

  const recData = result.fft_reconstructed.map((mag, i) => ({
    bin: i,
    Magnitude: mag,
  }));

  // Find max for consistent y-axis scale
  const maxMag = Math.max(
    ...result.fft_original,
    ...result.fft_reconstructed,
    0.01
  );

  const sharedYDomain = [0, maxMag * 1.1];

  return (
    <div className="chart-card">
      <div className="chart-header">
        <span className="chart-title">Frequency Domain — Sparsity Visualization</span>
        <div style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--muted)",
        }}>
          k = {result.sparsity_k} nonzero bins of {result.fft_original.length}
          &nbsp;({(result.sparsity_k / result.fft_original.length * 100).toFixed(1)}%)
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
        {/* Original FFT */}
        <div>
          <div style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "#60a5fa",
            textTransform: "uppercase",
            letterSpacing: ".1em",
            marginBottom: 8,
          }}>
            Original
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart
              data={origData}
              margin={{ top: 4, right: 4, left: -10, bottom: 4 }}
              barCategoryGap={0}
            >
              <CartesianGrid stroke="#1e2530" vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="bin"
                tick={axisStyle}
                tickLine={false}
                axisLine={{ stroke: "#1e2530" }}
                interval={Math.floor(origData.length / 4)}
                label={{ value: "Frequency bin", position: "insideBottom", offset: -2, fill: "#3a4556", fontSize: 9 }}
              />
              <YAxis
                tick={axisStyle}
                tickLine={false}
                axisLine={{ stroke: "#1e2530" }}
                domain={sharedYDomain}
                width={40}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,.03)" }} />
              <Bar dataKey="Magnitude" fill="#60a5fa" fillOpacity={0.8} radius={[1,1,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Reconstructed FFT */}
        <div>
          <div style={{
            fontFamily: "var(--font-mono)",
            fontSize: 10,
            color: "#34d399",
            textTransform: "uppercase",
            letterSpacing: ".1em",
            marginBottom: 8,
          }}>
            Reconstructed
          </div>
          <ResponsiveContainer width="100%" height={180}>
            <BarChart
              data={recData}
              margin={{ top: 4, right: 4, left: -10, bottom: 4 }}
              barCategoryGap={0}
            >
              <CartesianGrid stroke="#1e2530" vertical={false} strokeDasharray="3 3" />
              <XAxis
                dataKey="bin"
                tick={axisStyle}
                tickLine={false}
                axisLine={{ stroke: "#1e2530" }}
                interval={Math.floor(recData.length / 4)}
                label={{ value: "Frequency bin", position: "insideBottom", offset: -2, fill: "#3a4556", fontSize: 9 }}
              />
              <YAxis
                tick={axisStyle}
                tickLine={false}
                axisLine={{ stroke: "#1e2530" }}
                domain={sharedYDomain}
                width={40}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,.03)" }} />
              <Bar dataKey="Magnitude" fill="#34d399" fillOpacity={0.8} radius={[1,1,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Sparsity callout */}
      <div style={{
        marginTop: 14,
        padding: "10px 14px",
        background: "var(--surface)",
        border: "1px solid var(--border)",
        borderLeft: "3px solid var(--yellow)",
        borderRadius: "0 var(--radius) var(--radius) 0",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        color: "var(--muted)",
        lineHeight: 1.7,
      }}>
        <span style={{ color: "var(--yellow)", fontWeight: 600 }}>CS assumption:</span>
        &nbsp; signal is k-sparse in the Fourier basis —
        only <strong style={{ color: "var(--text)" }}>{result.sparsity_k}</strong> of{" "}
        <strong style={{ color: "var(--text)" }}>{result.fft_original.length}</strong> frequency
        bins are significant. ℓ₁ minimization exploits this structure to recover
        the full signal from just{" "}
        <strong style={{ color: "var(--yellow)" }}>{result.n_measurements}</strong> measurements
        (theory requires ≥{" "}
        <strong style={{ color: "var(--yellow)" }}>{result.theory_min_n}</strong>).
      </div>
    </div>
  );
}
