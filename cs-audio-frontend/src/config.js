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
