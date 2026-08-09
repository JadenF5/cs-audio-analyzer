// MetricsPanel.jsx
// Top row of metric cards + CS parameter chips.

export default function MetricsPanel({ result }) {
  const snrColor =
    result.snr_db > 40 ? "green" :
    result.snr_db > 20 ? "yellow" : "red";

  const corrColor =
    result.correlation > 0.99 ? "green" :
    result.correlation > 0.9  ? "yellow" : "red";

  return (
    <div className="chart-card">
      <div className="chart-header">
        <span className="chart-title">Reconstruction Quality</span>
        <span style={{
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          padding: "3px 10px",
          borderRadius: 20,
          background: result.quality_label === "excellent"
            ? "var(--green-dim)" : result.quality_label === "good"
            ? "var(--yellow-dim)" : "var(--red-dim)",
          color: result.quality_label === "excellent"
            ? "var(--green)" : result.quality_label === "good"
            ? "var(--yellow)" : "var(--red)",
          border: `1px solid ${result.quality_label === "excellent"
            ? "rgba(52,211,153,.25)" : result.quality_label === "good"
            ? "rgba(251,191,36,.25)" : "rgba(249,112,102,.25)"}`,
          textTransform: "uppercase",
          letterSpacing: ".08em",
          fontWeight: 600,
        }}>
          {result.quality_label}
        </span>
      </div>

      {/* Metric cards */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">SNR</div>
          <div className={`metric-value ${snrColor}`}>
            {result.snr_db.toFixed(1)}
          </div>
          <div className="metric-sub">dB &nbsp;·&nbsp; &gt;40 = excellent</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Correlation</div>
          <div className={`metric-value ${corrColor}`}>
            {result.correlation.toFixed(4)}
          </div>
          <div className="metric-sub">1.0000 = perfect</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">MSE</div>
          <div className="metric-value blue">
            {result.mse < 1e-6
              ? result.mse.toExponential(1)
              : result.mse.toFixed(4)}
          </div>
          <div className="metric-sub">mean squared error</div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Compression</div>
          <div className="metric-value yellow">
            {Math.round(result.compression_ratio * 100)}%
          </div>
          <div className="metric-sub">
            {result.n_measurements} of {result.n_samples} samples
          </div>
        </div>
      </div>

      {/* CS parameter chips */}
      <div className="cs-params">
        <div className="param-chip">
          N = <strong>{result.n_samples}</strong>
        </div>
        <div className="param-chip">
          k = <strong>{result.sparsity_k}</strong> sparse bins
        </div>
        <div className="param-chip">
          n = <strong>{result.n_measurements}</strong> measurements
        </div>
        <div className="param-chip">
          theory min n ≥ <strong>{result.theory_min_n}</strong>
        </div>
        <div className={`param-chip ${result.cs_feasible ? "feasible" : "infeasible"}`}>
          {result.cs_feasible ? "✓ CS feasible" : "✗ under-sampled"}
        </div>
        <div className="param-chip">
          solver: <strong>{result.solver_status}</strong>
        </div>
      </div>
    </div>
  );
}
