// config.js — Single source of truth for the backend API URL.
//
// Local dev: falls back to http://localhost:8000 automatically.
// Deployed:  set REACT_APP_API_URL in Vercel's project settings to your
//            Render backend URL, e.g. https://cs-audio-backend.onrender.com
//
// Create React App only exposes env vars prefixed with REACT_APP_, and
// only at BUILD time (not runtime) — so this must be set in Vercel's
// dashboard before/at build, not edited after deploy.

export const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:8000";

// Distinguish "server was asleep and we timed out waking it" (Render's
// free tier sleeps after ~15 min idle) from a genuine error, so the
// message doesn't wrongly imply something is broken.
export function friendlyError(e, fallback) {
  if (e.code === "ECONNABORTED") {
    return "The server was waking up from inactivity (free hosting tier) " +
           "and took longer than expected. Please try again — it should " +
           "be fast now that it's warm.";
  }
  return e.response?.data?.detail ?? `${fallback} Is the backend running?`;
}