// ControlPanel.jsx
// Left sidebar: signal source toggle, file upload, compression slider, run button.

import { useRef } from "react";

export default function ControlPanel({
  compression,
  setCompression,
  useDemo,
  setUseDemo,
  uploadInfo,
  onUpload,
  onReconstruct,
  loading,
  result,
}) {
  const fileRef = useRef(null);

  const handleFile = (e) => {
    const file = e.target.files[0];
    if (file) onUpload(file);
  };

  // Theoretical minimum measurements: k * log(N)
  const theoryMin = result
    ? result.theory_min_n
    : "—";

  const nMeas = result
    ? result.n_measurements
    : "—";

  return (
    <div>
      {/* ── Signal source ── */}
      <div className="card">
        <div className="card-label blue">Signal Source</div>

        <div className="source-toggle">
          <button
            className={`source-btn ${useDemo ? "active" : ""}`}
            onClick={() => { setUseDemo(true); }}
          >
            Demo
          </button>
          <button
            className={`source-btn ${!useDemo ? "active" : ""}`}
            onClick={() => setUseDemo(false)}
          >
            Upload
          </button>
        </div>

        {useDemo ? (
          <div className="theory-box">
            Uses a synthetic signal with <strong>k=4</strong> pure sine
            waves — exactly sparse in the Fourier basis. Ideal for
            demonstrating CS recovery.
          </div>
        ) : (
          <>
            <label className="upload-zone" htmlFor="audio-upload">
              <input
                id="audio-upload"
                ref={fileRef}
                type="file"
                accept=".wav,.mp3,.ogg,.flac"
                onChange={handleFile}
              />
              <div style={{ fontSize: 24, marginBottom: 6 }}>⊕</div>
              <div>Drop audio file here</div>
              <div style={{ fontSize: 10, marginTop: 4, opacity: .6 }}>
                .wav · .mp3 · .ogg · .flac
              </div>
            </label>

            {uploadInfo && (
              <div className="upload-success" style={{ marginTop: 10 }}>
                ✓ {uploadInfo.n_samples} samples · {uploadInfo.sample_rate} Hz
                · k={uploadInfo.sparsity_k} sparse bins
              </div>
            )}
          </>
        )}
      </div>

      {/* ── Compression ratio ── */}
      <div className="card">
        <div className="card-label yellow">Compression Ratio n/N</div>

        <div className="slider-container">
          <div className="slider-row">
            <input
              type="range"
              min={0.05}
              max={0.95}
              step={0.01}
              value={compression}
              onChange={(e) => setCompression(parseFloat(e.target.value))}
            />
            <span className="slider-value">{Math.round(compression * 100)}%</span>
          </div>
          <div className="slider-hints">
            <span>5% — max compression</span>
            <span>95% — near-full</span>
          </div>
        </div>

        <div className="theory-box" style={{ marginTop: 12 }}>
          <strong>Theory:</strong> need n ≥ k·log(N) measurements.<br />
          Current n = <strong>{nMeas}</strong> &nbsp;|&nbsp;
          Min = <strong>{theoryMin}</strong><br />
          {result && (
            <span style={{ color: result.cs_feasible ? "var(--green)" : "var(--red)" }}>
              {result.cs_feasible
                ? "✓ CS feasible — recovery expected"
                : "✗ Under-sampled — recovery may fail"}
            </span>
          )}
        </div>
      </div>

      {/* ── Run button ── */}
      <button
        className="run-btn"
        onClick={onReconstruct}
        disabled={loading || (!useDemo && !uploadInfo)}
      >
        {loading ? "Running..." : "▶  Run CS Pipeline"}
      </button>

      {!useDemo && !uploadInfo && (
        <p style={{
          fontFamily: "var(--font-mono)",
          fontSize: 10,
          color: "var(--muted)",
          textAlign: "center",
          marginTop: 8
        }}>
          Upload a file first, or switch to Demo
        </p>
      )}

      {/* ── CS pipeline recap ── */}
      <div className="card" style={{ marginTop: 12 }}>
        <div className="card-label purple">CS Pipeline</div>
        {[
          ["1", "Signal x  (N samples)"],
          ["2", "y = Φ · x  (n measurements)"],
          ["3", "A = Φ · Ψ⁻¹  (sensing matrix)"],
          ["4", "min ‖s‖₁  s.t.  A·s = y"],
          ["5", "x̂ = IFFT(s)  (reconstruct)"],
        ].map(([n, step]) => (
          <div key={n} style={{
            display: "flex", gap: 10,
            marginBottom: 6,
            fontFamily: "var(--font-mono)",
            fontSize: 11,
          }}>
            <span style={{
              color: "var(--purple)",
              fontWeight: 700,
              minWidth: 14,
            }}>{n}</span>
            <span style={{ color: "var(--muted)" }}>{step}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
