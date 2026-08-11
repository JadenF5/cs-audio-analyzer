# CS Audio Analyzer

**Compressed Sensing meets Audio Classification**  
A full‑stack web application that reconstructs sparse audio signals from far fewer measurements than Nyquist requires, then classifies the recovered signal as **tone**, **noise**, or **music** using a custom‑trained ML model.

**Live Demo**: [cs-audio-analyzer.vercel.app](https://cs-audio-analyzer.vercel.app/)

## Features

- **Compressed Sensing Pipeline**  
  - Builds a random sensing matrix `Φ` with configurable compression ratio  
  - Recovers the signal via ℓ₁‑minimisation (basis pursuit)  
  - Visualises original vs. reconstructed waveforms and frequency spectra  
  - Reports SNR, MSE, correlation, and sparsity metrics

- **Built‑in Demo Signal**  
  Instantly test the pipeline on a synthetic 4‑tone signal (no upload needed).

- **Audio Upload**  
  Supports `.wav`, `.mp3`, `.ogg`, `.flac` files. Automatically trims silence and resamples to 8 kHz.

- **ML Classification**  
  After reconstruction, classify the output as **tone** / **noise** / **music** using a gradient‑boosted tree ensemble trained on real‑world UrbanSound8k data augmented with synthetic white and coloured noise.

- **Interactive Frontend**  
  - Real‑time waveform and spectrum plots  
  - Adjustable compression ratio slider  
  - Confidence scores and feature summaries  
  - Dark‑themed UI built with React

---

## How It Works

1. **Sparsity Analysis** – The audio signal is transformed to the frequency domain (FFT) and its sparsity `k` is estimated.
2. **Sensing** – A random Gaussian measurement matrix `Φ` of size `n × N` is generated, where `n = compression_ratio × N`.
3. **Recovery** – The under‑determined system `y = Φx` is solved via ℓ₁‑minimisation:  
   `min ||s||₁  subject to  ΦΨs = y`, where `Ψ` is the inverse Fourier basis.
4. **Reconstruction** – The recovered frequency‑domain vector is transformed back to the time domain.
5. **Classification** – 33 spectral/timbral features (MFCCs, flatness, bandwidth, centroid, etc.) are extracted and fed into a HistGradientBoosting classifier.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React, Chart.js, Axios |
| **Backend** | Python, FastAPI, NumPy, SciPy, scikit‑learn, librosa |
| **ML Model** | HistGradientBoostingClassifier (scikit‑learn) trained on UrbanSound8k + synthetic data |
| **Deployment** | Backend → [Render](https://render.com/), Frontend → [Vercel](https://vercel.com/) |

---

## Project Structure

```
backend/
├── main.py # FastAPI server
├── cs_engine.py # Compressed sensing algorithms
├── classifier.py # Audio feature extraction & ML classifier
├── evaluate_classifier.py # Model training & evaluation
├── audio_classifier_model.joblib # Pre‑trained model
└── requirements.txt
cs-audio-frontend/
├── src/
│ ├── App.jsx
│ ├── App.css
│ ├── config.js
│ ├── index.css
│ ├── index.js
│ └── components/
|   ├── ClassificationCard.jsx
|   ├── ControlPanel.jsx
|   ├── FrequencyChart.jsx
|   ├── MetricsPanel.jsx
|   └── WaveformChart.jsx
└── public/
```

## Notice

ML Classification might not work on the first try due to render being on the free tier, so you get a message, saying render will be waking up, so just rerun the ML Classification again to get the data after.