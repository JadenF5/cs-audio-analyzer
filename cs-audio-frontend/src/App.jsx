// App.jsx — Compressed Sensing Audio Analyzer
// Main app: manages state, API calls, and layout.

import { useState, useCallback, useEffect } from "react";
import axios from "axios";
import ControlPanel   from "./components/ControlPanel";
import WaveformChart  from "./components/WaveformChart";
import FrequencyChart from "./components/FrequencyChart";
import MetricsPanel   from "./components/MetricsPanel";
import ClassificationCard from "./components/ClassificationCard";
import { API_BASE_URL, friendlyError } from "./config";
import "./App.css";

const API = API_BASE_URL;

export default function App() {
  // ── State ──────────────────────────────────────────────
  const [result,      setResult]      = useState(null);   // reconstruction result from API
  const [uploadInfo,  setUploadInfo]  = useState(null);   // info from /upload
  const [loading,     setLoading]     = useState(false);  // spinner while CS runs
  const [error,       setError]       = useState(null);   // error message
  const [compression, setCompression] = useState(0.27);   // compression ratio slider
  const [useDemo,     setUseDemo]     = useState(true);   // demo vs uploaded file

  // ── Wake up the backend as soon as the page loads ────────
  // Render's free tier sleeps after ~15 min idle; the first request after
  // that can take 50+ seconds. Firing a harmless /health ping on mount
  // means the server is usually already warm by the time someone actually
  // clicks a button, instead of that delay landing on their first action.
  useEffect(() => {
    axios.get(`${API}/health`).catch(() => {
      // Ignore failures here — the real request below will surface any
      // actual problem with a proper error message.
    });
  }, []);

  // ── Upload handler ──────────────────────────────────────
  const handleUpload = useCallback(async (file) => {
    setError(null);
    setLoading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await axios.post(`${API}/upload`, form, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 90000,   // generous — Render free tier cold start can take 50s+
      });
      setUploadInfo(data);
      setUseDemo(false);
      setResult(null);
    } catch (e) {
      setError(friendlyError(e, "Upload failed."));
    } finally {
      setLoading(false);
    }
  }, []);

  // ── Reconstruct handler ─────────────────────────────────
  const handleReconstruct = useCallback(async () => {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      let data;
      if (useDemo) {
        // GET /demo — no body needed
        ({ data } = await axios.get(`${API}/demo`, {
          params: { compression_ratio: compression },
          timeout: 90000,
        }));
      } else {
        // POST /reconstruct — uses last uploaded signal
        ({ data } = await axios.post(`${API}/reconstruct`, {
          compression_ratio: compression,
          use_demo: false,
          seed: 42,
        }, { timeout: 90000 }));
      }
      setResult(data);
    } catch (e) {
      setError(friendlyError(e, "Reconstruction failed."));
    } finally {
      setLoading(false);
    }
  }, [compression, useDemo]);

  // ── Render ──────────────────────────────────────────────
  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="header-left">
            <div className="logo-tag">CS AUDIO</div>
            <div className="header-titles">
              <h1 className="header-title">Compressed Sensing<br/>Audio Analyzer</h1>
              <p className="header-sub">
                Reconstruct audio from sub-Nyquist measurements via ℓ₁ minimization
              </p>
            </div>
          </div>
          <div className="header-badges">
            <span className="badge blue">NumPy</span>
            <span className="badge green">CVXPY</span>
            <span className="badge purple">FastAPI</span>
            <span className="badge yellow">React</span>
          </div>
        </div>
      </header>

      {/* ── Main layout ── */}
      <main className="main">
        {/* Left: controls */}
        <aside className="sidebar">
          <ControlPanel
            compression={compression}
            setCompression={setCompression}
            useDemo={useDemo}
            setUseDemo={setUseDemo}
            uploadInfo={uploadInfo}
            onUpload={handleUpload}
            onReconstruct={handleReconstruct}
            loading={loading}
            result={result}
          />
        </aside>

        {/* Right: visualizations */}
        <section className="charts">
          {/* Error banner */}
          {error && (
            <div className="error-banner">
              <span className="error-icon">⚠</span>
              <span>{error}</span>
            </div>
          )}

          {/* Loading state */}
          {loading && (
            <div className="loading-card">
              <div className="spinner" />
              <div>
                <p className="loading-title">Running CS Pipeline...</p>
                <p className="loading-sub">
                  Building measurement matrix · Solving ℓ₁ minimization · Reconstructing signal
                </p>
              </div>
            </div>
          )}

          {/* Empty state */}
          {!loading && !result && !error && (
            <div className="empty-state">
              <div className="empty-icon">◈</div>
              <h2 className="empty-title">Ready to Compress</h2>
              <p className="empty-sub">
                Choose a compression ratio, then click <strong>Run CS Pipeline</strong> to see
                how ℓ₁ minimization reconstructs an audio signal from far fewer measurements
                than the Nyquist rate requires.
              </p>
              <div className="empty-math">
                min ‖s‖₁ &nbsp; subject to &nbsp; A·s = y
              </div>
            </div>
          )}

          {/* Results */}
          {!loading && result && (
            <div className="results-grid">
              <MetricsPanel result={result} />
              <WaveformChart result={result} />
              <FrequencyChart result={result} />
              <ClassificationCard useDemo={useDemo} />
            </div>
          )}
        </section>
      </main>
    </div>
  );
}