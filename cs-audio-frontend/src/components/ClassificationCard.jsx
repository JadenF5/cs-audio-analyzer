// ClassificationCard.jsx
// Shows ML classification result: predicted class, confidence bar,
// probability breakdown for all classes, and key audio features.

import { useState } from "react";
import axios from "axios";
import { API_BASE_URL, friendlyError } from "../config";

const API = API_BASE_URL;

const COLOR_MAP = {
  blue:   { bg: "var(--blue-dim)",   border: "rgba(96,165,250,.25)",   text: "var(--blue)"   },
  red:    { bg: "var(--red-dim)",    border: "rgba(249,112,102,.25)",  text: "var(--red)"    },
  green:  { bg: "var(--green-dim)",  border: "rgba(52,211,153,.25)",   text: "var(--green)"  },
  yellow: { bg: "var(--yellow-dim)", border: "rgba(251,191,36,.25)",   text: "var(--yellow)" },
};

// Horizontal confidence bar for a single class
function ScoreBar({ label, score, color, isWinner }) {
  const c = COLOR_MAP[color] ?? COLOR_MAP.blue;
  return (
    <div style={{ marginBottom: 8 }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        marginBottom: 4,
        color: isWinner ? c.text : "var(--muted)",
        fontWeight: isWinner ? 700 : 400,
      }}>
        <span>{label}</span>
        <span>{(score * 100).toFixed(1)}%</span>
      </div>
      <div style={{
        height: 6,
        background: "var(--border2)",
        borderRadius: 3,
        overflow: "hidden",
      }}>
        <div style={{
          height: "100%",
          width: `${score * 100}%`,
          background: isWinner ? c.text : "var(--dim)",
          borderRadius: 3,
          transition: "width .6s ease",
        }} />
      </div>
    </div>
  );
}

// One feature row in the feature summary table
function FeatureRow({ label, value, unit }) {
  return (
    <div style={{
      display: "flex",
      justifyContent: "space-between",
      alignItems: "center",
      padding: "7px 0",
      borderBottom: "1px solid var(--border)",
      fontFamily: "var(--font-mono)",
      fontSize: 11,
    }}>
      <span style={{ color: "var(--muted)" }}>{label}</span>
      <span style={{ color: "var(--text)", fontWeight: 600 }}>
        {value} <span style={{ color: "var(--dim)", fontWeight: 400 }}>{unit}</span>
      </span>
    </div>
  );
}

export default function ClassificationCard({ useDemo }) {
  const [result,  setResult]  = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);

  const handleClassify = async () => {
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const { data } = await axios.post(`${API}/classify`, {
        use_reconstructed: true,
        use_demo: useDemo,
      }, { timeout: 90000 });   // generous — Render free tier cold start can take 50s+
      setResult(data);
    } catch (e) {
      setError(friendlyError(e, "Classification failed."));
    } finally {
      setLoading(false);
    }
  };

  // Class display config (matches CLASS_INFO in classifier.py)
  const CLASS_CONFIG = {
    tone:   { color: "blue",   icon: "◎", label: "Pure Tone"  },
    noise:  { color: "red",    icon: "≋", label: "Noise"      },
    music:  { color: "green",  icon: "♪", label: "Music"      },
    speech: { color: "yellow", icon: "◉", label: "Speech"     },
  };

  const winnerColor = result
    ? (COLOR_MAP[result.class_color] ?? COLOR_MAP.blue)
    : null;

  return (
    <div className="chart-card">
      {/* Header */}
      <div className="chart-header">
        <span className="chart-title">ML Classification — Phase 4</span>
        <button
          onClick={handleClassify}
          disabled={loading}
          style={{
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            fontWeight: 700,
            padding: "6px 16px",
            background: loading ? "var(--surface)" : "var(--purple)",
            color: loading ? "var(--muted)" : "#080b10",
            border: "none",
            borderRadius: 6,
            cursor: loading ? "not-allowed" : "pointer",
            transition: "all .2s",
          }}
        >
          {loading ? "Classifying..." : "▶  Classify Signal"}
        </button>
      </div>

      {/* Description */}
      <div style={{
        fontFamily: "var(--font-mono)",
        fontSize: 11,
        color: "var(--muted)",
        marginBottom: 16,
        lineHeight: 1.6,
      }}>
        Extracts 30 audio features (MFCCs, spectral centroid, ZCR, rolloff, RMS)
        and feeds them into a RandomForest trained on synthetic data.
        Classifies into: <span style={{ color: "var(--blue)" }}>tone</span> ·{" "}
        <span style={{ color: "var(--red)" }}>noise</span> ·{" "}
        <span style={{ color: "var(--green)" }}>music</span> ·{" "}
        <span style={{ color: "var(--yellow)" }}>speech</span>
      </div>

      {/* Error */}
      {error && (
        <div className="error-banner" style={{ marginBottom: 16 }}>
          <span className="error-icon">⚠</span>
          <span>{error}</span>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: 14,
          padding: "20px 0",
        }}>
          <div className="spinner" style={{ width: 28, height: 28 }} />
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 12, color: "var(--text)" }}>
              Extracting features...
            </div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 10, color: "var(--muted)" }}>
              MFCCs · spectral centroid · ZCR · rolloff · RMS
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {!loading && result && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>

          {/* Left: prediction + confidence bars */}
          <div>
            {/* Big prediction result */}
            <div style={{
              background: winnerColor.bg,
              border: `1px solid ${winnerColor.border}`,
              borderRadius: 10,
              padding: "20px 24px",
              textAlign: "center",
              marginBottom: 16,
            }}>
              <div style={{ fontSize: 36, marginBottom: 8 }}>
                {result.class_icon}
              </div>
              <div style={{
                fontFamily: "var(--font-display)",
                fontSize: "1.4rem",
                fontWeight: 800,
                color: winnerColor.text,
                marginBottom: 4,
              }}>
                {result.class_label}
              </div>
              <div style={{
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: winnerColor.text,
                opacity: .8,
                marginBottom: 12,
              }}>
                {(result.confidence * 100).toFixed(1)}% confidence
              </div>
              <div style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                color: "var(--muted)",
                lineHeight: 1.6,
              }}>
                {result.class_description}
              </div>
            </div>

            {/* Confidence bars for all classes */}
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 10,
                          color: "var(--muted)", marginBottom: 8,
                          textTransform: "uppercase", letterSpacing: ".1em" }}>
              Class Probabilities
            </div>
            {Object.entries(result.all_scores)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, score]) => (
                <ScoreBar
                  key={cls}
                  label={CLASS_CONFIG[cls]?.label ?? cls}
                  score={score}
                  color={CLASS_CONFIG[cls]?.color ?? "blue"}
                  isWinner={cls === result.predicted_class}
                />
              ))
            }
          </div>

          {/* Right: feature summary */}
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontSize: 10,
                          color: "var(--muted)", marginBottom: 8,
                          textTransform: "uppercase", letterSpacing: ".1em" }}>
              Extracted Features
            </div>

            <div style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: "0 14px",
              marginBottom: 14,
            }}>
              <FeatureRow
                label="Spectral Centroid"
                value={result.feature_summary.spectral_centroid_hz?.toFixed(1)}
                unit="Hz"
              />
              <FeatureRow
                label="Spectral Rolloff"
                value={result.feature_summary.spectral_rolloff_hz?.toFixed(1)}
                unit="Hz"
              />
              <FeatureRow
                label="Zero Crossing Rate"
                value={result.feature_summary.zero_crossing_rate?.toFixed(4)}
                unit=""
              />
              <FeatureRow
                label="RMS Energy"
                value={result.feature_summary.rms_energy?.toFixed(4)}
                unit=""
              />
              <FeatureRow
                label="MFCC 1 (mean)"
                value={result.feature_summary.mfcc_1_mean?.toFixed(2)}
                unit=""
              />
              <FeatureRow
                label="MFCC 2 (mean)"
                value={result.feature_summary.mfcc_2_mean?.toFixed(2)}
                unit=""
              />
            </div>

            {/* Feature interpretation */}
            <div style={{
              background: "var(--surface)",
              border: "1px solid var(--border)",
              borderLeft: `3px solid ${winnerColor.text}`,
              borderRadius: "0 8px 8px 0",
              padding: "10px 14px",
              fontFamily: "var(--font-mono)",
              fontSize: 10,
              color: "var(--muted)",
              lineHeight: 1.7,
            }}>
              <span style={{ color: winnerColor.text, fontWeight: 600 }}>
                Why {result.class_label}?
              </span>
              {result.predicted_class === "tone" && (
                <span> Low ZCR, narrow spectral centroid, low energy variation — hallmarks of a pure periodic signal.</span>
              )}
              {result.predicted_class === "noise" && (
                <span> Very high ZCR, flat spectrum, high RMS variance — characteristic of white noise.</span>
              )}
              {result.predicted_class === "music" && (
                <span> Rich harmonic content, moderate ZCR, structured MFCC pattern — typical of musical tones.</span>
              )}
              {result.predicted_class === "speech" && (
                <span> Amplitude modulation, formant-like resonances, moderate ZCR — matches voiced speech pattern.</span>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!loading && !result && !error && (
        <div style={{
          textAlign: "center",
          padding: "30px 0",
          fontFamily: "var(--font-mono)",
          fontSize: 12,
          color: "var(--dim)",
        }}>
          Run CS Pipeline first, then click Classify Signal
        </div>
      )}
    </div>
  );
}